from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Proposal:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2; exclusive upper bound
    color: str
    score: float


@dataclass(frozen=True)
class Classification:
    label: str
    confidence: float
    margin: float
    entropy: float
    abstained: bool


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    label: str
    confidence: float
    margin: float
    abstained: bool
    source: str


@dataclass
class Track:
    id: int
    box: tuple[int, int, int, int]
    label: str
    confidence: float
    hits: int = 1
    misses: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_alert: float = -1e12
    labels: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    confirmed: bool = False
    abstained: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id, "box": self.box, "label": self.label,
            "confidence": round(self.confidence, 4), "hits": self.hits,
            "confirmed": self.confirmed, "abstained": self.abstained,
            "age_s": round(self.last_seen - self.first_seen, 3),
        }


@dataclass(frozen=True)
class Event:
    track_id: int
    label: str
    confidence: float
    severity: str
    first_seen: float
    confirmed_at: float
    bbox: tuple[int, int, int, int]
    source: str
    model_version: str
    pipeline_version: str = "1.0.0"

    def public(self) -> dict:
        item = asdict(self)
        item["bbox"] = list(self.bbox)
        item["confidence"] = round(item["confidence"], 4)
        return item
