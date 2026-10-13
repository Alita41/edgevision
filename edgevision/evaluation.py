"""Detection, abstention, and calibration metrics for *labeled* held-out data.

The class identifier is part of each labeled bounding box. Evaluating classification
alone cannot stand in for detection mAP or live Pi FPS.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from .vision import overlap


@dataclass(frozen=True)
class LabeledBox:
    frame_id:str
    label:str
    box:tuple[int,int,int,int]
    confidence:float=1.0
    abstained:bool=False


def evaluate_boxes(truth:list[LabeledBox], predictions:list[LabeledBox], iou_threshold:float=0.5, bins:int=10)->dict:
    if not 0 < iou_threshold <=1:raise ValueError("invalid IoU threshold")
    if bins<2:raise ValueError("bins must be >=2")
    labels=sorted({x.label for x in truth}|{x.label for x in predictions if not x.abstained})
    rows={}
    accepted=[x for x in predictions if not x.abstained]
    for label in labels:
        gt=defaultdict(list);pr=defaultdict(list)
        for x in truth:
            if x.label==label:gt[x.frame_id].append(x)
        for x in accepted:
            if x.label==label:pr[x.frame_id].append(x)
        matched=[];tp=fp=fn=0
        for frame in sorted(set(gt)|set(pr)):
            g=gt[frame];p=sorted(pr[frame],key=lambda x:x.confidence,reverse=True)
            used=set()
            for pred in p:
                candidates=[(overlap(pred.box,real.box),i) for i,real in enumerate(g) if i not in used]
                best=max(candidates,default=(0,-1))
                success=best[0]>=iou_threshold
                matched.append((float(pred.confidence),int(success)))
                if success:used.add(best[1]);tp+=1
                else:fp+=1
            fn+=len(g)-len(used)
        precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
        f1=2*precision*recall/max(precision+recall,1e-12)
        rows[label]={"tp":tp,"fp":fp,"fn":fn,"precision":round(precision,4),"recall":round(recall,4),"f1":round(f1,4)}
    total_tp=sum(x["tp"] for x in rows.values())
    total_fp=sum(x["fp"] for x in rows.values())
    total_fn=sum(x["fn"] for x in rows.values())
    p=total_tp/max(total_tp+total_fp,1);r=total_tp/max(total_tp+total_fn,1)
    return {"ground_truth_boxes":len(truth),"prediction_boxes":len(predictions),
            "abstained_boxes":len(predictions)-len(accepted),
            "abstention_rate":round((len(predictions)-len(accepted))/max(len(predictions),1),4),
            "iou_threshold":iou_threshold,
            "micro_precision":round(p,4),"micro_recall":round(r,4),
            "micro_f1":round(2*p*r/max(p+r,1e-12),4),"per_class":rows}


def calibration_ece(confidences:list[float],correctness:list[int],bins:int=10)->float:
    if len(confidences)!=len(correctness):raise ValueError("lengths differ")
    if not confidences:return 0.0
    if any(x<0 or x>1 for x in confidences):raise ValueError("probabilities out of range")
    conf=np.array(confidences,dtype=np.float64);correct=np.array(correctness,dtype=np.float64)
    err=0.0
    for i in range(bins):
        lo,hi=i/bins,(i+1)/bins
        mask=(conf>=lo)&(conf<hi if i<bins-1 else conf<=hi)
        if mask.any():err+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
    return round(float(err),6)
