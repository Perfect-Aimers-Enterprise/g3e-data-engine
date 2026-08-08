"""
HuggingFace `datasets`-backed downloader.

This is the reference implementation for `kind: huggingface` sources in
configs/datasets.yaml. It streams a dataset (so it never buffers the whole
remote dataset in memory), filters rows down to the target classes, and
stops as soon as each class has hit its per-run budget — which is exactly
what keeps v1 from pulling down more than it needs.

NOTE: this module imports `datasets`/`huggingface_hub` lazily (inside the
method, not at module import time) so the rest of the engine — config
loading, priority allocation, the FastAPI app, tests — works fully offline
without requiring those (larger) packages or network access at all.
"""
from __future__ import annotations

import os
from pathlib import Path

from g3e_data_engine.downloader.base import (
    Downloader,
    DownloadRequest,
    DownloadedImage,
    register,
)


@register("huggingface")
class HFDownloader(Downloader):
    def download(self, request: DownloadRequest) -> list[DownloadedImage]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' package is required for huggingface sources. "
                "Install it with `pip install datasets huggingface_hub`."
            ) from exc

        from g3e_data_engine.core.config import load_engine_config

        cfg = load_engine_config()
        source = cfg.datasets.sources[request.source_name]
        if not source.hf_repo:
            raise ValueError(
                f"Source '{request.source_name}' has no hf_repo set in "
                "configs/datasets.yaml — fill it in before enabling this source."
            )

        dest = Path(request.dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        remaining = dict(request.target_classes)  # class -> images still needed
        results: list[DownloadedImage] = []

        # streaming=True: rows are pulled lazily, one at a time, off the wire.
        # This is what stops v1 from ever materializing a full remote dataset
        # on disk just to grab a few thousand images from it.
        ds = load_dataset(source.hf_repo, split="train", streaming=True)

        for i, row in enumerate(ds):
            if all(v <= 0 for v in remaining.values()):
                break  # every target class already satisfied

            row_classes = _extract_class_names(row, source.classes)
            useful = [c for c in row_classes if remaining.get(c, 0) > 0]
            if not useful:
                continue

            img = row.get("image")
            if img is None:
                continue

            out_path = dest / f"{request.source_name}_{i:07d}.jpg"
            try:
                img.convert("RGB").save(out_path, format="JPEG", quality=95)
            except Exception:
                continue  # unreadable/corrupt row — skip rather than fail the whole run

            for c in useful:
                remaining[c] = max(0, remaining[c] - 1)

            results.append(
                DownloadedImage(
                    local_path=str(out_path),
                    source_name=request.source_name,
                    classes_present=row_classes,
                    raw_annotations=row.get("objects", row.get("annotations", [])) or [],
                )
            )

            if len(results) >= source.max_images:
                break  # hard per-source cap from datasets.yaml, belt-and-suspenders

        return results


def _extract_class_names(row: dict, allowed: list[str]) -> list[str]:
    """
    Best-effort extraction of which of `allowed` classes appear in a HF
    dataset row. Different HF vision datasets use different schemas
    (categories, objects.category, label, etc.) — extend this function, not
    the downloader class, when wiring up a new dataset's schema.
    """
    found: set[str] = set()

    label = row.get("label")
    if isinstance(label, str) and label in allowed:
        found.add(label)

    for key in ("objects", "annotations"):
        objs = row.get(key)
        if isinstance(objs, dict) and "category" in objs:
            for cat in objs["category"]:
                if isinstance(cat, str) and cat in allowed:
                    found.add(cat)
        elif isinstance(objs, list):
            for obj in objs:
                cat = obj.get("category") if isinstance(obj, dict) else None
                if isinstance(cat, str) and cat in allowed:
                    found.add(cat)

    return sorted(found)
