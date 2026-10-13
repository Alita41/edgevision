"""Measure *synthetic* pipeline compute throughput and latency on local hardware."""
from __future__ import annotations
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
import numpy as np
from edgevision.pipeline import VisionPipeline
from edgevision.sources import SyntheticSource


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frames",type=int,default=300)
    ap.add_argument("--warmup",type=int,default=20)
    ap.add_argument("--width",type=int,default=960)
    ap.add_argument("--height",type=int,default=540)
    ap.add_argument("--out",default="artifacts/reference-pipeline.json")
    args=ap.parse_args()
    if args.frames<1 or args.warmup<0:raise SystemExit("invalid frame counts")
    src=SyntheticSource(args.width,args.height,total_frames=args.frames+args.warmup)
    pipeline=VisionPipeline()
    for _ in range(args.warmup):
        ok,frame=src.read()
        if ok:pipeline.process(frame,draw=False)
    dt=[]
    started=time.perf_counter()
    for _ in range(args.frames):
        ok,frame=src.read()
        if not ok:break
        t=time.perf_counter();pipeline.process(frame,draw=False)
        dt.append((time.perf_counter()-t)*1000)
    elapsed=time.perf_counter()-started
    result={"type":"synthetic computational benchmark; NOT a traffic-sign accuracy/FPS-on-Raspberry-Pi claim",
            "backend":"color-shape candidate proposals, no neural model",
            "frames":len(dt),"width":args.width,"height":args.height,
            "elapsed_seconds":round(elapsed,3),"throughput_fps":round(len(dt)/elapsed,2),
            "processing_ms":{"mean":round(statistics.mean(dt),3),"p50":round(float(np.percentile(dt,50)),3),
                             "p95":round(float(np.percentile(dt,95)),3),"p99":round(float(np.percentile(dt,99)),3)},
            "events":pipeline.store.count(),
            "environment":{"machine":platform.machine(),"cpu_count":os.cpu_count(),"python":platform.python_version(),
                           "opencv":__import__('cv2').__version__}}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    pipeline.store.close()

if __name__=="__main__":main()
