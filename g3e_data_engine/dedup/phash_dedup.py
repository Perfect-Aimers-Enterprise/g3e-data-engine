"""
Duplicate detection via perceptual hashing (pHash).

Near-duplicates (recompressed, slightly cropped, resized copies of the same
image) are a common source of leaked train/val/test contamination in
scraped datasets, so this runs before the split stage, not after.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

from g3e_data_engine.core.config import DuplicatesConfig


@dataclass
class DedupResult:
    kept: list[str]
    duplicates: list[str]
    duplicate_of: dict[str, str]  # duplicate_path -> original_path it matched


def find_duplicates(paths: list[str | Path], config: DuplicatesConfig) -> DedupResult:
    if not config.enabled:
        return DedupResult(kept=[str(p) for p in paths], duplicates=[], duplicate_of={})

    hashes: list[tuple[str, imagehash.ImageHash]] = []
    kept: list[str] = []
    duplicates: list[str] = []
    duplicate_of: dict[str, str] = {}

    for p in paths:
        p = str(p)
        try:
            with Image.open(p) as img:
                h = imagehash.phash(img)
        except Exception:
            # Unreadable images are handled by the validator stage, not here.
            kept.append(p)
            continue

        match = None
        for existing_path, existing_hash in hashes:
            if h - existing_hash <= config.hamming_distance_threshold:
                match = existing_path
                break

        if match is not None:
            duplicates.append(p)
            duplicate_of[p] = match
        else:
            hashes.append((p, h))
            kept.append(p)

    return DedupResult(kept=kept, duplicates=duplicates, duplicate_of=duplicate_of)
