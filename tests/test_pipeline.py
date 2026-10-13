from edgevision.sources import SyntheticSource
from edgevision.pipeline import VisionPipeline,risk_tier
from edgevision.config import VisionConfig
from edgevision.storage import EventStore


def test_end_to_end_alert_once_per_cooldown():
    p=VisionPipeline(VisionConfig(min_track_hits=3,alert_cooldown_s=8),store=EventStore())
    source=SyntheticSource(total_frames=20)
    for i in range(20):
        ok,frame=source.read();assert ok
        annotated,events=p.process(frame,now=i/24)
        assert annotated.shape==frame.shape
    assert p.store.count()==1
    assert p.store.recent()[0]["severity"]=="observation"
    assert p.stats()["confirmed_tracks"]>=1
    p.store.close()


def test_risk_tier():
    assert risk_tier("stop","tflite-traffic-model")=="attention"
    assert risk_tier("red sign-like candidate","heuristic-candidates")=="observation"
    assert risk_tier("speed limit 50 km/h","trained-model")=="notice"
