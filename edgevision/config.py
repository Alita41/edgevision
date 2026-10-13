from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class VisionConfig:
    # Values are conservative *demo defaults*, not calibrated operating points.
    proposal_min_area: int = 180
    proposal_max_area_fraction: float = 0.20
    proposal_min_fill: float = 0.24
    confidence_threshold: float = 0.70
    margin_threshold: float = 0.13
    min_track_hits: int = 3
    track_max_misses: int = 6
    match_iou_threshold: float = 0.18
    alert_cooldown_s: float = 8.0
    queue_capacity: int = 2
    max_event_rows: int = 10000
    jpeg_quality: int = 77

    def validate(self) -> "VisionConfig":
        if self.proposal_min_area < 1 or not 0 < self.proposal_max_area_fraction <= 1:
            raise ValueError("invalid proposal area thresholds")
        for name in ("proposal_min_fill", "confidence_threshold", "margin_threshold", "match_iou_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_track_hits < 1 or self.track_max_misses < 0 or self.queue_capacity < 1:
            raise ValueError("invalid tracker/queue settings")
        if self.alert_cooldown_s < 0 or self.max_event_rows < 1 or not 30 <= self.jpeg_quality <= 95:
            raise ValueError("invalid cooldown, event limit, or JPEG quality")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "VisionConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = set(cls.__dataclass_fields__)
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        return cls(**data).validate()

    def updated(self, **kwargs) -> "VisionConfig":
        return replace(self, **kwargs).validate()
