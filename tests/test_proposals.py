import cv2
import numpy as np
from edgevision.config import VisionConfig
from edgevision.sources import SyntheticSource
from edgevision.vision import proposals,overlap,crop_rgb


def test_red_proposal_in_synthetic_frame():
    ok,frame=SyntheticSource().read()
    assert ok
    result=proposals(frame,VisionConfig())
    assert any(p.color=="red" and p.score>0.6 for p in result)
    p=result[0]
    cropped=crop_rgb(frame,p.box,(64,64))
    assert cropped.shape==(64,64,3)


def test_blank_no_proposal():
    assert not proposals(np.zeros((180,320,3),dtype=np.uint8),VisionConfig())


def test_iou_and_bad_shapes():
    assert overlap((0,0,10,10),(0,0,10,10))==1.0
    assert overlap((0,0,10,10),(10,10,20,20))==0.0
    import pytest
    with pytest.raises(ValueError):proposals(np.zeros((10,10),dtype=np.uint8),VisionConfig())
