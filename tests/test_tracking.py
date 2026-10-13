from edgevision.config import VisionConfig
from edgevision.tracking import Tracker
from edgevision.types import Detection


def det(label="stop",conf=.95,abstained=False,box=(12,12,90,90)):
    return Detection(box,label,conf,.4,abstained,"test")


def test_confirmation_and_id_stability():
    t=Tracker(VisionConfig(min_track_hits=3))
    assert not t.update([det()],1)[0].confirmed
    assert not t.update([det(box=(13,12,91,90))],2)[0].confirmed
    out=t.update([det(box=(14,12,92,90))],3)
    assert len(out)==1 and out[0].id==1 and out[0].confirmed
    assert out[0].hits==3


def test_unknown_suppressed_and_track_expires():
    t=Tracker(VisionConfig(min_track_hits=2,track_max_misses=1))
    t.update([det(abstained=True)],1)
    assert not t.update([det(abstained=True)],2)[0].confirmed
    t.update([],3)
    assert t.update([],4)==[]


def test_conflicting_labels_require_majority():
    t=Tracker(VisionConfig(min_track_hits=3))
    t.update([det("stop")],1)
    t.update([det("yield")],2)
    assert not t.update([det("stop")],3)[0].confirmed
    assert t.update([det("stop")],4)[0].confirmed


def test_configurable_long_confirmation_window():
    t=Tracker(VisionConfig(min_track_hits=10))
    for i in range(9):
        assert not t.update([det()],i)[0].confirmed
    assert t.update([det()],9)[0].confirmed
