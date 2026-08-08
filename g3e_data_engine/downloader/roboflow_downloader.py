"""
Roboflow Universe-backed downloader.

Reference implementation for `kind: roboflow` sources in
configs/datasets.yaml — proof that the downloader registry (base.py) is
genuinely source-agnostic: the rest of the pipeline (validators, dedup,
split, stats, export) doesn't know or care that this source isn't
Hugging Face.

`project` + `version` must be pinned exactly (e.g.
"fire-rqbio/fire-and-smoke-yikzn", version 3) so a future update to the
Roboflow project doesn't silently change what's in an already-released
G3E dataset version.

Like hf_downloader.py, `roboflow` is imported lazily inside `download()` so
the rest of the engine works fully offline without it installed — install
with `pip install "g3e-data-engine[roboflow]"` when you actually need this
source.
"""
from __future__ import annotations

import glob
import shutil
from pathlib import Path

import yaml

from g3e_data_engine.downloader.base import (
    Downloader,
    DownloadRequest,
    DownloadedImage,
    register,
)
from g3e_data_engine.core.credentials import require_token


@register("roboflow")
class RoboflowDownloader(Downloader):
    def download(self, request: DownloadRequest) -> list[DownloadedImage]:
        try:
            from roboflow import Roboflow
        except ImportError as exc:
            raise RuntimeError(
                "The 'roboflow' package is required for roboflow sources. "
                'Install it with `pip install "g3e-data-engine[roboflow]"`.'
            ) from exc

        from g3e_data_engine.core.config import load_engine_config

        cfg = load_engine_config()
        source = cfg.datasets.sources[request.source_name]
        if not source.project or source.version is None:
            raise ValueError(
                f"Source '{request.source_name}' has no project/version pinned in "
                "configs/datasets.yaml — fill both in before enabling this source."
            )
        if "/" not in source.project:
            raise ValueError(
                f"Source '{request.source_name}'.project must be 'workspace/project-slug', "
                f"got {source.project!r}."
            )

        # Roboflow has no anonymous/public read path in its API client —
        # unlike Hugging Face, a key is always required here.
        api_key = require_token("roboflow", source)

        workspace_slug, project_slug = source.project.split("/", 1)
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace_slug).project(project_slug)
        version = project.version(source.version)

        dest = Path(request.dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        # YOLOv8 export gives us: <dest>/{train,valid,test}/images/*.jpg and
        # matching .../labels/*.txt, plus a data.yaml with the class name list.
        downloaded = version.download("yolov8", location=str(dest))
        export_root = Path(downloaded.location)

        class_names = _read_class_names(export_root / "data.yaml")

        remaining = dict(request.target_classes)
        results: list[DownloadedImage] = []

        for split_dir in ("train", "valid", "test"):
            images_dir = export_root / split_dir / "images"
            labels_dir = export_root / split_dir / "labels"
            if not images_dir.exists():
                continue

            for image_path in sorted(glob.glob(str(images_dir / "*.*"))):
                if all(v <= 0 for v in remaining.values()):
                    break

                label_path = labels_dir / (Path(image_path).stem + ".txt")
                row_classes = _classes_in_label_file(
                    label_path, class_names, source.classes, source.class_map
                )
                useful = [c for c in row_classes if remaining.get(c, 0) > 0]
                if not useful:
                    continue

                out_path = dest / f"{request.source_name}_{len(results):07d}{Path(image_path).suffix}"
                shutil.copy2(image_path, out_path)

                for c in useful:
                    remaining[c] = max(0, remaining[c] - 1)

                results.append(
                    DownloadedImage(
                        local_path=str(out_path),
                        source_name=request.source_name,
                        classes_present=row_classes,
                        raw_annotations=[{"label_file": str(label_path)}] if label_path.exists() else [],
                    )
                )

                if len(results) >= source.max_images:
                    return results

        return results


def _read_class_names(data_yaml_path: Path) -> list[str]:
    if not data_yaml_path.exists():
        return []
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", [])
    if isinstance(names, dict):  # some exports use {0: "fire", 1: "smoke"}
        names = [names[i] for i in sorted(names)]
    return list(names)


def _classes_in_label_file(
    label_path: Path,
    class_names: list[str],
    allowed: list[str],
    class_map: dict[str, str] | None,
) -> list[str]:
    """
    Reads a YOLO-format label .txt (class_id x y w h per line) and returns
    which of `allowed` (G3E) classes are present, translating via
    `class_map` (source label -> G3E label) if the source doesn't already
    use G3E's exact class names.
    """
    if not label_path.exists() or not class_names:
        return []

    class_map = class_map or {}
    found: set[str] = set()

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                raw_name = class_names[class_id]
            except (ValueError, IndexError):
                continue
            translated = class_map.get(raw_name, raw_name)
            if translated in allowed:
                found.add(translated)

    return sorted(found)
