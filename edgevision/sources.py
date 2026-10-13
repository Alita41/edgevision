"""Real-camera, video-file, and deterministic synthetic sources."""
from __future__ import annotations
import time
from typing import Protocol
import cv2
import numpy as np


class FrameSource(Protocol):
    name: str
    fps_hint:float
    def read(self)->tuple[bool,np.ndarray|None]: ...
    def release(self)->None: ...


class SyntheticSource:
    """Generates a moving scene for repeatable architecture demos, not ML accuracy evaluation."""
    name="synthetic-demo"
    def __init__(self,width:int=960,height:int=540,fps:float=24.0,total_frames:int|None=None):
        self.width=width;self.height=height;self.fps_hint=fps;self.total_frames=total_frames;self.i=0

    def read(self):
        if self.total_frames is not None and self.i>=self.total_frames:
            return False,None
        i=self.i;self.i+=1
        w,h=self.width,self.height
        canvas=np.zeros((h,w,3),dtype=np.uint8)
        canvas[:]=(20,24,32)
        # Atmospheric road-scene illustration, visibly synthetic.
        cv2.rectangle(canvas,(0,int(h*.63)),(w,h),(26,34,42),-1)
        cv2.line(canvas,(int(w*.48),int(h*.63)),(int(w*.24),h),(64,75,83),2)
        cv2.line(canvas,(int(w*.52),int(h*.63)),(int(w*.76),h),(64,75,83),2)
        for k in range(7):
            y=int(h*.65)+k*52
            cv2.line(canvas,(w//2,y),(w//2,y+20),(106,118,123),2)
        cv2.rectangle(canvas,(0,int(h*.48)),(w,int(h*.63)),(24,32,37),-1)
        for k in range(15):
            x=k*80+30
            cv2.rectangle(canvas,(x,int(h*.45)-(k%3)*20),(x+48,int(h*.63)),(31,39,48),-1)
        x=int(w*.30+100*np.sin(i/32))
        y=int(h*.43+15*np.sin(i/61))
        radius=48
        # Red regulatory-sign-like octagon; does not assert true STOP recognition.
        coords=[]
        for k in range(8):
            a=np.deg2rad(22.5+45*k)
            coords.append((int(x+radius*np.cos(a)),int(y+radius*np.sin(a))))
        cv2.fillPoly(canvas,[np.asarray(coords,np.int32)],(32,36,210))
        cv2.polylines(canvas,[np.asarray(coords,np.int32)],True,(242,239,234),3)
        cv2.putText(canvas,"STOP",(x-34,y+8),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),2,cv2.LINE_AA)
        cv2.line(canvas,(x,y+radius),(x,int(h*.68)),(120,130,136),6)
        cv2.putText(canvas,"SYNTHETIC TEST SCENE",(26,38),cv2.FONT_HERSHEY_SIMPLEX,.55,(160,176,185),1,cv2.LINE_AA)
        cv2.putText(canvas,f"FRAME {i:06d}",(26,h-25),cv2.FONT_HERSHEY_SIMPLEX,.5,(148,163,176),1,cv2.LINE_AA)
        return True,canvas

    def release(self):pass


class OpenCVSource:
    def __init__(self,source:str|int,width:int=960,height:int=540):
        self.name=f"camera:{source}" if isinstance(source,int) else f"video:{source}"
        self.capture=cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video source: {source}")
        if isinstance(source,int):
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE,1)
        hint=float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps_hint=hint if 1 <=hint<=120 else 24.0

    def read(self):return self.capture.read()
    def release(self):self.capture.release()


def make_source(uri:str,width:int=960,height:int=540,fps:float=24.0)->FrameSource:
    if uri=="synthetic":return SyntheticSource(width,height,fps)
    if uri.startswith("camera:"):
        return OpenCVSource(int(uri.split(":",1)[1]),width,height)
    return OpenCVSource(uri,width,height)
