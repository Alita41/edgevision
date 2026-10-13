from __future__ import annotations
from collections import deque
from dataclasses import replace
import statistics
import time
import threading
import cv2
import numpy as np
from .types import Detection,Event
from .config import VisionConfig
from .classifier import Classifier, CandidateOnlyClassifier
from .tracking import Tracker
from .storage import EventStore
from .vision import proposals


def risk_tier(label:str,source:str)->str:
    if source.startswith("heuristic"):
        return "observation"
    normalized=label.lower()
    if any(s in normalized for s in ("stop", "no entry", "yield", "give way", "priority")):
        return "attention"
    if "speed limit" in normalized or "restriction" in normalized:
        return "notice"
    return "observation"


class VisionPipeline:
    def __init__(self,config:VisionConfig|None=None,classifier:Classifier|None=None,store:EventStore|None=None):
        self.config=(config or VisionConfig()).validate()
        self.classifier=classifier or CandidateOnlyClassifier()
        self.store=store or EventStore(limit=self.config.max_event_rows)
        self.tracker=Tracker(self.config)
        self.frames=0
        self.events=0
        self.latencies=deque(maxlen=300)
        self.first_wall=None
        self.last_frame_time=None
        self.last_detections=[]
        self.lock=threading.RLock()

    @property
    def mode(self)->str:
        return "heuristic demo" if isinstance(self.classifier,CandidateOnlyClassifier) else "TFLite model"

    def process(self,frame:np.ndarray,now:float|None=None,draw:bool=True)->tuple[np.ndarray,list[Event]]:
        with self.lock:
            started=time.perf_counter()
            now=time.time() if now is None else now
            detected=[]
            for candidate in proposals(frame,self.config):
                outcome=self.classifier.predict(frame,candidate,self.config)
                detected.append(Detection(candidate.box,outcome.label,outcome.confidence,outcome.margin,
                                          outcome.abstained,self.classifier.name))
            tracks=self.tracker.update(detected,now)
            fresh=[]
            for track in tracks:
                if not track.confirmed or track.misses or track.abstained:
                    continue
                if now-track.last_alert<self.config.alert_cooldown_s:
                    continue
                track.last_alert=now
                event=Event(track.id,track.label,track.confidence,risk_tier(track.label,self.classifier.name),
                            track.first_seen,now,track.box,self.classifier.name,self.classifier.name)
                self.store.add(event)
                fresh.append(event)
            self.frames+=1;self.events+=len(fresh)
            self.last_frame_time=now
            if self.first_wall is None:self.first_wall=now
            self.last_detections=[x.public() for x in tracks if x.misses==0]
            if draw:
                frame=self.annotate(frame,tracks)
            self.latencies.append((time.perf_counter()-started)*1000)
            return frame,fresh

    def annotate(self,frame:np.ndarray,tracks)->np.ndarray:
        result=frame.copy()
        h,w=result.shape[:2]
        for t in tracks:
            if t.misses:continue
            x1,y1,x2,y2=t.box
            color=(53,235,176) if t.confirmed else (95,197,241)
            if t.abstained:color=(155,155,155)
            cv2.rectangle(result,(x1,y1),(x2,y2),color,2)
            name=t.label.upper() if t.confirmed else "CANDIDATE / VERIFYING"
            # Do not call heuristic scores "classification confidence".
            label=f"ID {t.id:02d}  {name[:32]}" if self.mode=="heuristic demo" else f"ID {t.id:02d}  {name[:26]}  {t.confidence:.2f}"
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.42,1)
            by=max(y1-8,19)
            cv2.rectangle(result,(x1,by-th-8),(min(w-1,x1+tw+14),by+4),(15,24,29),-1)
            cv2.putText(result,label,(x1+7,by-2),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
        cv2.rectangle(result,(0,h-35),(w,h),(16,25,32),-1)
        cv2.putText(result,"EDGEVISION / "+self.mode.upper(),(18,h-12),cv2.FONT_HERSHEY_SIMPLEX,.45,(220,231,228),1,cv2.LINE_AA)
        return result

    def stats(self)->dict:
        with self.lock:
            lat=list(self.latencies)
            fps=0.0
            if self.first_wall and self.last_frame_time and self.frames>1:
                fps=(self.frames-1)/max(self.last_frame_time-self.first_wall,1e-6)
            return {"mode":self.mode,"model":self.classifier.name,"frames":self.frames,
                    "events":self.store.count(),"active_tracks":len([x for x in self.tracker.tracks.values() if not x.misses]),
                    "confirmed_tracks":len([x for x in self.tracker.tracks.values() if x.confirmed and not x.misses]),
                    "throughput_fps":round(fps,2),
                    "processing_ms_p50":round(float(np.percentile(lat,50)),2) if lat else 0.,
                    "processing_ms_p95":round(float(np.percentile(lat,95)),2) if lat else 0.,
                    "processing_ms_mean":round(statistics.mean(lat),2) if lat else 0.,
                    "detections":list(self.last_detections),"config":self.config.to_dict()}

    def set_config(self,**settings):
        with self.lock:
            self.config=self.config.updated(**settings)
            self.tracker.cfg=self.config
