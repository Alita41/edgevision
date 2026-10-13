"""Color/geometry region proposals, not a learned traffic-sign detector.

This stage finds candidate red/blue sign-like regions; TFLite optionally classifies
crops. The heuristic fallback only labels them *candidates*.
"""
from __future__ import annotations

import cv2
import numpy as np
from .config import VisionConfig
from .types import Proposal


def overlap(a: tuple[int,int,int,int], b: tuple[int,int,int,int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    area = max(x2-x1, 0) * max(y2-y1, 0)
    if not area:
        return 0.0
    aa = max(a[2]-a[0],0)*max(a[3]-a[1],0)
    ab = max(b[2]-b[0],0)*max(b[3]-b[1],0)
    return area / float(aa + ab - area)


def proposals(frame: np.ndarray, cfg: VisionConfig, limit: int = 16) -> list[Proposal]:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected a BGR frame with three channels")
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = {
        "red": (cv2.inRange(hsv, np.array([0, 90, 70]), np.array([13, 255, 255])) |
                cv2.inRange(hsv, np.array([165, 90, 70]), np.array([179, 255, 255]))),
        "blue": cv2.inRange(hsv, np.array([92, 90, 60]), np.array([132, 255, 255])),
    }
    found: list[Proposal] = []
    kernel = np.ones((3,3), dtype=np.uint8)
    for color, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < cfg.proposal_min_area or area > w*h*cfg.proposal_max_area_fraction:
                continue
            x,y,bw,bh = cv2.boundingRect(c)
            aspect = bw / float(max(bh,1))
            if not 0.63 <= aspect <= 1.55:
                continue
            fill = area / float(bw*bh)
            if fill < cfg.proposal_min_fill:
                continue
            # Shape quality is only a heuristic, never an ML confidence.
            roundness = 4 * np.pi * area / (cv2.arcLength(c, True)**2 + 1e-6)
            score = float(np.clip(0.48 * min(fill/0.8,1) + 0.36*min(roundness,1) + 0.16*(1-abs(1-aspect)),0,1))
            pad = max(3, int(min(bw,bh)*0.08))
            box = (max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad))
            found.append(Proposal(box,color,score))
    selected: list[Proposal] = []
    for item in sorted(found,key=lambda d:d.score,reverse=True):
        if any(overlap(item.box, old.box) > 0.35 for old in selected):
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def crop_rgb(frame: np.ndarray, box: tuple[int,int,int,int], size: tuple[int,int]) -> np.ndarray:
    x1,y1,x2,y2=box
    part=frame[y1:y2, x1:x2]
    if part.size == 0:
        raise ValueError("empty crop")
    return cv2.resize(cv2.cvtColor(part,cv2.COLOR_BGR2RGB),size,interpolation=cv2.INTER_AREA)
