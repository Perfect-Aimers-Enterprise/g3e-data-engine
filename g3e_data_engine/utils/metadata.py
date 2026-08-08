"""
Per-image metadata records + versions.json bookkeeping.

Schema mirrors metadata/metadata.json in DATASET_SPEC.md:
{
  "id": "000001",
  "dataset": "coco",
  "source": "huggingface",
  "classes": ["person", "car"],
  "width": 1280,
  "height": 720,
  "split": "train"
}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ImageMetadata:
    id: str
    dataset: str
    source: str
    classes: list[str]
    width: int
    height: int
    split: str = "unassigned"


def write_metadata(records: list[ImageMetadata], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)


def bump_version(versions_path: str | Path, notes: str, image_count: int) -> dict:
    """
    Append a new entry to metadata/versions.json. Called once per completed
    pipeline run, never mid-run — see DATASET_SPEC.md "Versioning policy".
    """
    versions_path = Path(versions_path)
    versions_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    if versions_path.exists():
        with open(versions_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    next_minor = len(history) + 1
    entry = {
        "version": f"1.{next_minor}.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image_count": image_count,
        "notes": notes,
    }
    history.append(entry)

    with open(versions_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return entry
