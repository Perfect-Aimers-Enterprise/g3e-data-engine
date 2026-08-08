"""
Packages a processed dataset (images/ + labels/ + metadata/) into a single
versioned release folder/zip under datasets/releases/.

Uploading that release to Hugging Face Hub is a separate, explicit step
(scripts/upload_hf.py) — this module never pushes anything over the
network, so "export" always stays a safe, local, offline operation.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def export_release(
    processed_dir: str | Path,
    metadata_dir: str | Path,
    releases_dir: str | Path,
    version: str,
) -> Path:
    processed_dir = Path(processed_dir)
    metadata_dir = Path(metadata_dir)
    releases_dir = Path(releases_dir)

    release_root = releases_dir / f"g3e-vision-dataset-v{version}"
    if release_root.exists():
        raise FileExistsError(
            f"Release {release_root} already exists. Versions are immutable — "
            "bump the version instead of overwriting."
        )
    release_root.mkdir(parents=True)

    for sub in ("images", "labels"):
        src = processed_dir / sub
        if src.exists():
            shutil.copytree(src, release_root / sub)

    meta_dest = release_root / "metadata"
    meta_dest.mkdir(exist_ok=True)
    for f in ("classes.json", "metadata.json", "stats.json", "versions.json"):
        src = metadata_dir / f
        if src.exists():
            shutil.copy2(src, meta_dest / f)

    archive_path = shutil.make_archive(
        base_name=str(release_root), format="zip", root_dir=release_root
    )
    return Path(archive_path)
