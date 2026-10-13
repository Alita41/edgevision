import time
from fastapi.testclient import TestClient
from edgevision.sources import SyntheticSource
from edgevision.pipeline import VisionPipeline
from edgevision.service import VisionService
from edgevision.api import create_app
from edgevision.config import VisionConfig
from edgevision.storage import EventStore


def test_service_end_to_end_and_api(tmp_path):
    source=SyntheticSource(width=640,height=360,fps=24.0,total_frames=12)
    pipe=VisionPipeline(VisionConfig(min_track_hits=2),store=EventStore(tmp_path/"test.sqlite3"))
    service=VisionService(source,pipe,pace=False)
    app=create_app(service)
    with TestClient(app) as client:
        for _ in range(200):
            if service._ended.is_set() and service._queue.empty() and pipe.frames>=1:break
            time.sleep(.02)
        t=client.get("/api/v1/telemetry")
        assert t.status_code==200
        assert t.json()["frames"]>=1
        assert t.json()["mode"]=="heuristic demo"
        assert client.get("/").status_code==200
        assert "EdgeVision" in client.get("/").text
        assert client.get("/style.css").status_code==200
        assert client.get("/app.js").status_code==200
        assert client.get("/api/v1/events").status_code==200
        assert client.patch("/api/v1/config",json={"min_track_hits":4}).json()["config"]["min_track_hits"]==4
        assert client.patch("/api/v1/config",json={"not_a_setting":1}).status_code==422
        assert client.get("/api/v1/events?limit=900").status_code==422
