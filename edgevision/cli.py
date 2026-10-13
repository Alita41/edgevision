from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import cv2
from .classifier import CandidateOnlyClassifier,TFLiteClassifier
from .config import VisionConfig
from .pipeline import VisionPipeline
from .storage import EventStore
from .sources import SyntheticSource, make_source
from .service import VisionService


def parser():
    a=argparse.ArgumentParser(prog="edgevision",description="Local-first edge perception and event monitoring")
    sub=a.add_subparsers(dest="command",required=True)
    serve=sub.add_parser("serve",help="Run live monitoring dashboard")
    serve.add_argument("--source",default="synthetic",help="synthetic, camera:0, or path/to/video.mp4")
    serve.add_argument("--host",default="127.0.0.1",help="Bind loopback by default; do not expose cameras on public interfaces")
    serve.add_argument("--port",type=int,default=8080)
    serve.add_argument("--model",help="Optional trained .tflite classifier path")
    serve.add_argument("--labels",help="JSON list of classes matching trained model")
    serve.add_argument("--db",default="data/events.sqlite3")
    serve.add_argument("--config",default="configs/demo.json")
    serve.add_argument("--width",type=int,default=960)
    serve.add_argument("--height",type=int,default=540)
    demo=sub.add_parser("smoke",help="Headless deterministic end-to-end synthetic smoke test")
    demo.add_argument("--frames",type=int,default=90)
    demo.add_argument("--output",default="artifacts/smoke.json")
    demo.add_argument("--image",default="artifacts/smoke-last.jpg")
    return a


def main(argv:list[str]|None=None):
    args=parser().parse_args(argv)
    if args.command=="serve":
        if bool(args.model)!=bool(args.labels):raise SystemExit("--model and --labels must be supplied together")
        if args.port<=0 or args.port>65535:raise SystemExit("invalid port")
        config=VisionConfig.from_json(args.config) if Path(args.config).exists() else VisionConfig()
        model=TFLiteClassifier(args.model,args.labels) if args.model else CandidateOnlyClassifier()
        pipeline=VisionPipeline(config,model,EventStore(args.db,config.max_event_rows))
        source=make_source(args.source,args.width,args.height)
        service=VisionService(source,pipeline)
        from .api import create_app
        import uvicorn
        print(f"EdgeVision: http://{args.host}:{args.port} | {pipeline.mode} | source={source.name}")
        uvicorn.run(create_app(service),host=args.host,port=args.port)
    else:
        if args.frames<=0:raise SystemExit("--frames must be positive")
        source=SyntheticSource(total_frames=args.frames)
        pipeline=VisionPipeline()
        last=None
        for i in range(args.frames):
            ok,frame=source.read()
            if not ok:break
            last,_=pipeline.process(frame,now=1730000000+i/24)
        result=pipeline.stats()
        result.update({"source":source.name,"benchmark_type":"synthetic-smoke-not-model-accuracy",
                       "events_sample":pipeline.store.recent(5)})
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2),encoding="utf-8")
        if last is not None:
            image=Path(args.image);image.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(image),last)
        print(json.dumps(result,indent=2))
        pipeline.store.close()


if __name__=="__main__":main()
