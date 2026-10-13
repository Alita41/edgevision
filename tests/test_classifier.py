import numpy as np
from edgevision.classifier import TFLiteClassifier, CandidateOnlyClassifier
from edgevision.types import Proposal
from edgevision.config import VisionConfig


def test_quantization_and_dequantization():
    tensor={"dtype":np.int8,"quantization":(.01,-128)}
    v=np.array([0.0,.5,1.0],dtype=np.float32)
    encoded=TFLiteClassifier._quantize(v,tensor)
    restored=TFLiteClassifier._dequantize(encoded,tensor)
    assert encoded.dtype==np.int8
    assert np.max(np.abs(restored-v))<.011


def test_heuristic_is_explicitly_not_model():
    c=CandidateOnlyClassifier()
    out=c.predict(np.zeros((20,20,3),dtype=np.uint8),Proposal((0,0,19,19),"red",.83),VisionConfig())
    assert "candidate" in out.label
    assert "no ML" in c.name


def test_fake_interpreter_uncertainty_gate():
    class FakeInterpreter:
        def __init__(self,output):self.output=np.asarray([output],dtype=np.float32)
        def set_tensor(self,index,value):self.input=value
        def invoke(self):pass
        def get_tensor(self,index):return self.output
    # Bypass optional native dependency and exercise the real prediction path.
    c=TFLiteClassifier.__new__(TFLiteClassifier)
    c.labels=['stop','yield','other'];c.shape=[1,32,32,3]
    c.input={'index':0,'dtype':np.float32,'quantization':(0.,0)}
    c.output={'index':0,'dtype':np.float32,'quantization':(0.,0)}
    c.interpreter=FakeInterpreter([.86,.09,.05]);c.name='fake-model'
    frame=np.zeros((40,40,3),dtype=np.uint8)
    region=Proposal((1,1,39,39),'red',.8)
    confident=c.predict(frame,region,VisionConfig())
    assert not confident.abstained and confident.label=='stop'
    c.interpreter=FakeInterpreter([.39,.37,.24])
    uncertain=c.predict(frame,region,VisionConfig())
    assert uncertain.abstained and uncertain.label=='uncertain sign'
