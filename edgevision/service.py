"""Non-blocking acquisition with explicit bounded backpressure.

Only the latest annotated JPEG is kept in memory. Source frames are never saved.
"""
from __future__ import annotations
import queue
import threading
import time
import cv2
from .pipeline import VisionPipeline
from .sources import FrameSource


class VisionService:
    def __init__(self,source:FrameSource,pipeline:VisionPipeline,pace:bool=True):
        self.source=source
        self.pipeline=pipeline
        self.pace=pace
        self._queue=queue.Queue(maxsize=pipeline.config.queue_capacity)
        self._stop=threading.Event()
        self._ended=threading.Event()
        self._threads=[]
        self._latest:bytes|None=None
        self._version=0
        self._condition=threading.Condition()
        self.dropped_frames=0
        self.captured_frames=0
        self.error:str|None=None
        self.started=False

    def start(self):
        if self.started:return
        self.started=True
        self._threads=[threading.Thread(target=self._acquire,name="ev-capture",daemon=True),
                       threading.Thread(target=self._infer,name="ev-infer",daemon=True)]
        for t in self._threads:t.start()

    def _acquire(self):
        try:
            fps=max(float(self.source.fps_hint),1.0)
            target=1.0/fps
            while not self._stop.is_set():
                start=time.monotonic()
                ok,frame=self.source.read()
                if not ok or frame is None:break
                self.captured_frames+=1
                try:self._queue.put_nowait(frame)
                except queue.Full:
                    try:self._queue.get_nowait();self._queue.task_done()
                    except queue.Empty:pass
                    self.dropped_frames+=1
                    try:self._queue.put_nowait(frame)
                    except queue.Full:self.dropped_frames+=1
                if self.pace and target>0:
                    self._stop.wait(max(0,target-(time.monotonic()-start)))
        except Exception as exc:
            self.error=f"Capture failed: {exc}"
        finally:
            self.source.release()
            self._ended.set()

    def _infer(self):
        while not self._stop.is_set():
            try:frame=self._queue.get(timeout=.15)
            except queue.Empty:
                if self._ended.is_set():break
                continue
            try:
                annotated,_=self.pipeline.process(frame)
                ok,encoded=cv2.imencode(".jpg",annotated,[cv2.IMWRITE_JPEG_QUALITY,self.pipeline.config.jpeg_quality])
                if ok:
                    with self._condition:
                        self._latest=encoded.tobytes()
                        self._version+=1
                        self._condition.notify_all()
            except Exception as exc:
                self.error=f"Processing failed: {exc}"
                self._stop.set()
            finally:self._queue.task_done()

    def get_jpeg(self,after_version:int=0,wait:float=0.5)->tuple[int,bytes|None]:
        with self._condition:
            if self._version<=after_version:
                self._condition.wait(timeout=wait)
            return self._version,self._latest

    def metrics(self)->dict:
        result=self.pipeline.stats()
        result.update({"source":self.source.name,"captured_frames":self.captured_frames,
                       "dropped_frames":self.dropped_frames,"queue_depth":self._queue.qsize(),
                       "stream_ended":self._ended.is_set() and self._queue.empty(),
                       "error":self.error})
        return result

    def stop(self):
        self._stop.set()
        with self._condition:self._condition.notify_all()
        for t in self._threads:
            if t.is_alive():t.join(timeout=2)
        self.source.release()
