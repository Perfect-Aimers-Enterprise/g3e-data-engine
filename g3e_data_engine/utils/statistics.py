"""
Dataset statistics — tells you where the pipeline is losing images and
whether any class is underrepresented after filtering.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class RejectionCounts:
    total_seen: int = 0
    accepted: int = 0
    rejected_blurry: int = 0
    rejected_dark_or_bright: int = 0
    rejected_low_res: int = 0
    rejected_corrupted: int = 0
    rejected_duplicate: int = 0

    @property
    def rejected(self) -> int:
        return self.total_seen - self.accepted


@dataclass
class DatasetStats:
    rejection: RejectionCounts
    class_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "images": {
                "total_seen": self.rejection.total_seen,
                "accepted": self.rejection.accepted,
                "rejected": self.rejection.rejected,
                "rejected_breakdown": {
                    "blurry": self.rejection.rejected_blurry,
                    "dark_or_bright": self.rejection.rejected_dark_or_bright,
                    "low_resolution": self.rejection.rejected_low_res,
                    "corrupted": self.rejection.rejected_corrupted,
                    "duplicate": self.rejection.rejected_duplicate,
                },
            },
            "class_counts": self.class_counts,
            "split_counts": self.split_counts,
        }


def build_class_counts(records: list[dict]) -> dict[str, int]:
    """
    records: list of metadata dicts like {"classes": ["person", "car"], ...}
    """
    counter: Counter[str] = Counter()
    for r in records:
        for cls in r.get("classes", []):
            counter[cls] += 1
    return dict(sorted(counter.items(), key=lambda kv: -kv[1]))
