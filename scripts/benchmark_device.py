"""Benchmarks a running service on the actual device, not a synthetic claim.

Examples: start EdgeVision serve --source camera:0 --model ... first.
"""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
from urllib.request import urlopen


def get_json(base):
    with urlopen(base.rstrip('/')+'/api/v1/telemetry',timeout=4) as r:
        return json.load(r)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--url",default="http://127.0.0.1:8080")
    ap.add_argument("--seconds",type=float,default=30.)
    ap.add_argument("--out",default="artifacts/device-benchmark.json")
    args=ap.parse_args()
    if args.seconds<2:raise SystemExit("--seconds must be >=2")
    start=get_json(args.url)
    t0=time.monotonic()
    time.sleep(args.seconds)
    t1=time.monotonic()
    end=get_json(args.url)
    frames=max(0,end["frames"]-start["frames"])
    drops=max(0,end["dropped_frames"]-start["dropped_frames"])
    result={"type":"runtime device measurement", "mode":end["mode"],"source":end["source"],
            "model":end["model"],"duration_seconds":round(t1-t0,3),
            "frames_processed":frames,"frames_dropped":drops,
            "observed_processing_fps":round(frames/(t1-t0),2),
            "latest_processing_ms_p50":end["processing_ms_p50"],
            "latest_processing_ms_p95":end["processing_ms_p95"],
            "device":{"machine":platform.machine(),"system":platform.system(),"release":platform.release()},
            "note":"No accuracy figure is implied; evaluate model predictions against labels separately."}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
