"""Read-only monitoring and explicitly scoped parameter tuning; no image uploads."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel, Field, ConfigDict
from .service import VisionService

WEB=Path(__file__).resolve().parent/"web"

class TuningRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    confidence_threshold:float|None=Field(default=None,ge=0,le=1)
    margin_threshold:float|None=Field(default=None,ge=0,le=1)
    min_track_hits:int|None=Field(default=None,ge=1,le=30)
    alert_cooldown_s:float|None=Field(default=None,ge=0,le=3600)


def create_app(service:VisionService)->FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        service.start()
        try:yield
        finally:
            service.stop()
            service.pipeline.store.close()
    app=FastAPI(title="EdgeVision",version="1.0.0",lifespan=lifespan,
                description="Local-first edge perception and metadata-only monitoring")

    @app.get("/",include_in_schema=False)
    def home():return FileResponse(WEB/"index.html")

    @app.get("/style.css",include_in_schema=False)
    def css():return FileResponse(WEB/"style.css",media_type="text/css")

    @app.get("/app.js",include_in_schema=False)
    def js():return FileResponse(WEB/"app.js",media_type="text/javascript")

    @app.get("/favicon.svg",include_in_schema=False)
    def favicon():return FileResponse(WEB/"favicon.svg",media_type="image/svg+xml")

    @app.get("/api/v1/health")
    def health():
        info=service.metrics()
        return {"status":"degraded" if info["error"] else "ok",
                "mode":info["mode"],"source":info["source"],"error":info["error"]}

    @app.get("/api/v1/telemetry")
    def telemetry():return service.metrics()

    @app.get("/api/v1/events")
    def events(limit:int=Query(30,ge=1,le=200),severity:str|None=None):
        if severity and severity not in {"observation","notice","attention"}:
            raise HTTPException(400,"unknown severity")
        return {"events":service.pipeline.store.recent(limit,severity),"total":service.pipeline.store.count()}

    @app.get("/api/v1/config")
    def config():return service.pipeline.config.to_dict()

    @app.patch("/api/v1/config")
    def tune(body:TuningRequest):
        changes=body.model_dump(exclude_none=True)
        if not changes:raise HTTPException(400,"no settings provided")
        try:service.pipeline.set_config(**changes)
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
        return {"config":service.pipeline.config.to_dict(),"updated":sorted(changes)}

    @app.get("/frame.jpg",include_in_schema=False)
    def frame():
        _,jpeg=service.get_jpeg(wait=.15)
        if jpeg is None:raise HTTPException(503,"waiting for first frame")
        return Response(jpeg,media_type="image/jpeg",headers={"Cache-Control":"no-store"})

    @app.get("/stream.mjpg",include_in_schema=False)
    def stream():
        def frames():
            version=0
            while not service._stop.is_set():
                new_version,jpeg=service.get_jpeg(version,.8)
                if new_version==version:
                    if service._ended.is_set() and service._queue.empty():break
                    continue
                version=new_version
                if jpeg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "+
                           str(len(jpeg)).encode()+b"\r\n\r\n"+jpeg+b"\r\n")
        return StreamingResponse(frames(),media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control":"no-cache, no-store","X-Content-Type-Options":"nosniff"})
    return app
