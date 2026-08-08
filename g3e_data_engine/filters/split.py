"""
Deterministic train/val/test split.

Splits by *image id*, not by annotation, so all boxes for one image always
land in the same split. Uses a seeded RNG so re-running the pipeline on the
same input set reproduces the same split.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from g3e_data_engine.core.config import SplitConfig


@dataclass
class SplitResult:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def split_ids(ids: list[str], config: SplitConfig) -> SplitResult:
    rng = random.Random(config.seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * config.train)
    n_val = int(n * config.val)
    # remainder goes to test to avoid dropping items to rounding
    n_test = n - n_train - n_val

    return SplitResult(
        train=shuffled[:n_train],
        val=shuffled[n_train : n_train + n_val],
        test=shuffled[n_train + n_val : n_train + n_val + n_test],
    )
