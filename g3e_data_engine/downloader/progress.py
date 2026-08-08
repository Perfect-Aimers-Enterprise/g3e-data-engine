"""
Per-source download progress/resume manifest.

Every downloader calls into this so the on-disk manifest shape (and the
resume behavior it enables) stays identical across huggingface/roboflow/
future downloader kinds, instead of each one inventing its own format.

The manifest lives at `<dest_dir>/_progress.json` and is rewritten
(atomically — write to a temp file, then rename over the real one) after
every image a downloader saves. This is the mechanism that answers "don't
wait for the whole source to finish before writing to disk" — each image is
written to disk and recorded in the manifest immediately, so a crash mid-run
loses at most the single in-flight image, not the whole source's progress:

  - Restart the same pipeline run afterward, and each downloader reloads
    this manifest, skips classes that are already satisfied, and (for
    streaming sources) resumes from the last processed row instead of
    starting over.
  - If a source's manifest is missing or corrupt (e.g. the process was
    killed mid-write), `load_progress` returns `{}` — the downloader just
    treats that source as starting fresh. Never a crash on resume.
"""
from __future__ import annotations

import json
from pathlib import Path


def progress_path(dest_dir: str | Path) -> Path:
    return Path(dest_dir) / "_progress.json"


def load_progress(dest_dir: str | Path) -> dict:
    path = progress_path(dest_dir)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A prior crash mid-write is exactly the case this needs to survive
        # — treat a corrupt manifest as "nothing recorded yet" rather than
        # raising and taking down the whole pipeline over a resume hint.
        return {}


def save_progress(dest_dir: str | Path, **fields) -> None:
    """
    Merge `fields` into whatever's already recorded and write it back.
    Called after every single image a downloader saves — cheap enough at
    v1's image-count scale (thousands, not millions) to prioritize "never
    lose more than one image's worth of progress" over write throughput.
    """
    path = progress_path(dest_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = load_progress(dest_dir)
    data.update(fields)

    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp_path.replace(path)  # atomic on POSIX — no partially-written manifest is ever visible
