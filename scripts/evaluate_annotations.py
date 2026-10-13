"""Evaluate held-out scene annotations, never synthetic demo frames as ground truth.

Input JSONL records: {"frame_id":"f01", "label":"stop", "box":[10,20,80,90],
                      "confidence":0.92, "abstained":false}
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from edgevision.evaluation import LabeledBox,evaluate_boxes


def load(path:str):
    out=[]
    with Path(path).open(encoding="utf-8") as fh:
        for n,line in enumerate(fh,1):
            if not line.strip():continue
            obj=json.loads(line)
            try:out.append(LabeledBox(str(obj["frame_id"]),str(obj["label"]),tuple(map(int,obj["box"])),float(obj.get("confidence",1)),bool(obj.get("abstained",False))))
            except (KeyError,ValueError,TypeError) as e:raise ValueError(f"{path}:{n}: {e}") from e
    return out


def main():
    ap=argparse.ArgumentParser(description="Evaluate scene-level labeled bounding boxes at IoU 0.5")
    ap.add_argument("--ground-truth",required=True)
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--out",default=None)
    args=ap.parse_args()
    result=evaluate_boxes(load(args.ground_truth),load(args.predictions))
    txt=json.dumps(result,indent=2)
    print(txt)
    if args.out:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        Path(args.out).write_text(txt+"\n",encoding="utf-8")


if __name__=="__main__":main()
