"""Greedy IoU tracking with temporal consensus and explicit unknown suppression."""
from __future__ import annotations
from collections import Counter
import statistics
from .types import Detection, Track
from .vision import overlap
from .config import VisionConfig


class Tracker:
    def __init__(self, config: VisionConfig):
        self.cfg=config
        self.tracks:dict[int,Track]={}
        self.next_id=1

    def update(self,detections:list[Detection],now:float)->list[Track]:
        pairs=[]
        for ti,track in self.tracks.items():
            for di,det in enumerate(detections):
                score=overlap(track.box,det.box)
                if score>=self.cfg.match_iou_threshold:
                    pairs.append((score,ti,di))
        used_tracks,used_dets=set(),set()
        for score,ti,di in sorted(pairs,reverse=True):
            if ti in used_tracks or di in used_dets:continue
            used_tracks.add(ti);used_dets.add(di)
            t,d=self.tracks[ti],detections[di]
            t.box=d.box;t.last_seen=now;t.hits+=1;t.misses=0
            t.labels.append(d.label if not d.abstained else "__unknown__")
            t.scores.append(d.confidence)
            window=max(7,self.cfg.min_track_hits)
            t.labels=t.labels[-window:];t.scores=t.scores[-window:]
            # Never confirm a track based on a single high-confidence spike.
            count=Counter(x for x in t.labels if x!="__unknown__")
            label, votes = count.most_common(1)[0] if count else ("uncertain sign",0)
            t.label=label;t.confidence=float(statistics.median(t.scores))
            t.abstained=d.abstained or votes < self.cfg.min_track_hits
            t.confirmed=(votes>=self.cfg.min_track_hits and not t.abstained)
        for ti,t in list(self.tracks.items()):
            if ti not in used_tracks:
                t.misses+=1
                if t.misses>self.cfg.track_max_misses:
                    del self.tracks[ti]
        for di,d in enumerate(detections):
            if di in used_dets:continue
            ti=self.next_id;self.next_id+=1
            label="__unknown__" if d.abstained else d.label
            self.tracks[ti]=Track(ti,d.box,d.label,d.confidence,first_seen=now,last_seen=now,
                                  labels=[label],scores=[d.confidence],abstained=d.abstained,
                                  confirmed=(self.cfg.min_track_hits==1 and not d.abstained))
        return list(self.tracks.values())
