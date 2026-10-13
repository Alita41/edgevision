import pytest
from edgevision.evaluation import LabeledBox,evaluate_boxes,calibration_ece


def test_box_metrics_and_abstention():
    gt=[LabeledBox("a","stop",(0,0,20,20)),LabeledBox("b","yield",(0,0,20,20))]
    preds=[LabeledBox("a","stop",(0,0,20,20),.91),LabeledBox("b","yield",(0,0,20,20),.46,True)]
    r=evaluate_boxes(gt,preds)
    assert r["micro_precision"]==1.0
    assert r["micro_recall"]==.5
    assert r["abstention_rate"]==.5
    assert r["per_class"]["yield"]["fn"]==1


def test_calibration_ece():
    assert calibration_ece([1,1],[1,1])==0
    assert calibration_ece([],[])==0
    with pytest.raises(ValueError):calibration_ece([.5],[1,0])
