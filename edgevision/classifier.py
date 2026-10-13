"""Runtime classifiers. TFLite is optional and imported only when used.

Never interpret heuristic candidate scores as calibrated neural probabilities.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol
import numpy as np
from .config import VisionConfig
from .types import Classification, Proposal
from .vision import crop_rgb


class Classifier(Protocol):
    name: str
    def predict(self, bgr: np.ndarray, candidate: Proposal, cfg: VisionConfig) -> Classification: ...


class CandidateOnlyClassifier:
    """Demonstration of tracking/alerts without pretending to recognize road signs."""
    name = "heuristic-candidates (no ML model)"

    def predict(self, bgr: np.ndarray, candidate: Proposal, cfg: VisionConfig) -> Classification:
        del bgr, cfg
        return Classification(
            label=f"{candidate.color} sign-like candidate", confidence=candidate.score,
            margin=0.0, entropy=1.0, abstained=False,
        )


class TFLiteClassifier:
    def __init__(self, model: str | Path, labels: str | Path, num_threads: int = 2):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite import Interpreter
            except ImportError as exc:
                raise RuntimeError("Install tflite-runtime (Pi) or TensorFlow to run real neural inference") from exc
        self.interpreter = Interpreter(model_path=str(model), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        with Path(labels).open(encoding="utf-8") as fh:
            data = json.load(fh)
        self.labels = data["labels"] if isinstance(data,dict) else data
        fingerprint=hashlib.sha256(Path(model).read_bytes()).hexdigest()[:10]
        self.name = f"{Path(model).name}@{fingerprint}"
        self.shape = self.input["shape"]
        if len(self.shape) != 4 or int(self.shape[0]) != 1 or int(self.shape[-1]) != 3:
            raise ValueError(f"expected NHWC RGB classifier, got {self.shape}")

    @staticmethod
    def _quantize(x: np.ndarray, detail: dict) -> np.ndarray:
        dtype = detail["dtype"]
        if np.issubdtype(dtype,np.floating):
            return x.astype(dtype)
        scale, zero = detail["quantization"]
        if not scale:
            raise ValueError("quantized tensor missing scale")
        info = np.iinfo(dtype)
        return np.clip(np.round(x/scale + zero),info.min,info.max).astype(dtype)

    @staticmethod
    def _dequantize(x: np.ndarray, detail: dict) -> np.ndarray:
        scale, zero=detail["quantization"]
        return (x.astype(np.float32)-zero)*scale if scale else x.astype(np.float32)

    def predict(self,bgr:np.ndarray,candidate:Proposal,cfg:VisionConfig)->Classification:
        height,width=int(self.shape[1]),int(self.shape[2])
        rgb=crop_rgb(bgr,candidate.box,(width,height)).astype(np.float32)/255.0
        self.interpreter.set_tensor(self.input["index"],self._quantize(rgb[None,...],self.input))
        self.interpreter.invoke()
        raw=self._dequantize(self.interpreter.get_tensor(self.output["index"])[0],self.output)
        if len(raw)!=len(self.labels):
            raise ValueError("model outputs and label count differ")
        if not np.all(np.isfinite(raw)):
            raise ValueError("non-finite model output")
        if np.any(raw<0) or not 0.99<=float(raw.sum())<=1.01:
            x=raw-np.max(raw)
            exp=np.exp(x)
            prob=exp/exp.sum()
        else:
            prob=raw/np.sum(raw)
        order=np.argsort(-prob)
        top,second=int(order[0]),int(order[1]) if len(order)>1 else int(order[0])
        conf=float(prob[top])
        margin=float(conf-prob[second]) if len(order)>1 else conf
        ent=float(-np.sum(prob*np.log(prob+1e-12))/max(math.log(len(prob)),1e-9))
        abstain=conf<cfg.confidence_threshold or margin<cfg.margin_threshold
        return Classification(self.labels[top] if not abstain else "uncertain sign",conf,margin,ent,abstain)
