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

Resume behavior: unlike the HF downloader (which streams row-by-row),
Roboflow's SDK exports the whole matched dataset in one call — so what's
resumable here is the *copy + classify* work after that export, not the
export itself. Each image we accept is copied and recorded in
`_progress.json` immediately; on a re-run, filenames already recorded are
skipped rather than re-copied/re-classified.

Like hf_downloader.py, `roboflow` is imported lazily inside `download()` so
the rest of the engine works fully offline without it installed — install
with `pip install "g3e-data-engine[roboflow]"` when you actually need this
source. Preflight (core/preflight.py) checks for this import BEFORE any
pipeline run reaches this code, so a missing install is caught in
milliseconds instead of after other sources have already downloaded.
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
from g3e_data_engine.downloader.progress import load_progress, save_progress


class _NullBar:
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


def _tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm as _real_tqdm

        return _real_tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


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

        dest = Path(request.dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        progress = load_progress(dest)
        results: list[DownloadedImage] = [
            DownloadedImage(local_path=r["path"], source_name=request.source_name, classes_present=r["classes"])
            for r in progress.get("images", [])
            if Path(r["path"]).exists()
        ]
        per_class_counts: dict[str, int] = dict(progress.get("per_class_counts", {}))
        processed_filenames: set[str] = set(progress.get("processed_filenames", []))

        if results:
            print(
                f"  [resume] {request.source_name}: found {len(results)} previously downloaded "
                "image(s) on disk — continuing instead of restarting."
            )

        remaining = {
            c: max(0, target - per_class_counts.get(c, 0)) for c, target in request.target_classes.items()
        }
        if all(v <= 0 for v in remaining.values()):
            print(f"  [{request.source_name}] already satisfied from a previous run — nothing to download.")
            return results

        workspace_slug, project_slug = source.project.split("/", 1)
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace_slug).project(project_slug)
        version = project.version(source.version)

        # YOLOv8 export gives us: <export_root>/{train,valid,test}/images/*.jpg
        # and matching .../labels/*.txt, plus a data.yaml with the class list.
        downloaded = version.download("yolov8", location=str(dest / "_roboflow_export"))
        export_root = Path(downloaded.location)
        class_names = _read_class_names(export_root / "data.yaml")

        image_paths: list[str] = []
        for split_dir in ("train", "valid", "test"):
            images_dir = export_root / split_dir / "images"
            if images_dir.exists():
                image_paths.extend(sorted(glob.glob(str(images_dir / "*.*"))))

        try:
            for image_path in _tqdm(image_paths, desc=request.source_name, unit="img"):
                if all(v <= 0 for v in remaining.values()):
                    break

                fname = Path(image_path).name
                if fname in processed_filenames:
                    continue  # already handled by a previous, interrupted run

                split_dir_name = Path(image_path).parents[1].name
                label_path = export_root / split_dir_name / "labels" / (Path(image_path).stem + ".txt")
                row_classes = _classes_in_label_file(label_path, class_names, source.classes, source.class_map)
                useful = [c for c in row_classes if remaining.get(c, 0) > 0]
                processed_filenames.add(fname)

                if not useful:
                    save_progress(
                        dest,
                        per_class_counts=per_class_counts,
                        images=[{"path": r.local_path, "classes": r.classes_present} for r in results],
                        processed_filenames=list(processed_filenames),
                    )
                    continue

                out_path = dest / f"{request.source_name}_{len(results):07d}{Path(image_path).suffix}"
                shutil.copy2(image_path, out_path)

                for c in useful:
                    remaining[c] = max(0, remaining[c] - 1)
                    per_class_counts[c] = per_class_counts.get(c, 0) + 1

                results.append(
                    DownloadedImage(
                        local_path=str(out_path),
                        source_name=request.source_name,
                        classes_present=row_classes,
                        raw_annotations=[{"label_file": str(label_path)}] if label_path.exists() else [],
                    )
                )

                # Written after EVERY copied image — see progress.py.
                save_progress(
                    dest,
                    per_class_counts=per_class_counts,
                    images=[{"path": r.local_path, "classes": r.classes_present} for r in results],
                    processed_filenames=list(processed_filenames),
                )

                if len(results) >= source.max_images:
                    break

        except Exception as exc:
            print(
                f"  [{request.source_name}] interrupted after {len(results)} image(s): {exc}. "
                "Everything copied so far is saved to disk and recorded — re-run to resume "
                "from here rather than starting over."
            )
            return results

        save_progress(
            dest,
            per_class_counts=per_class_counts,
            images=[{"path": r.local_path, "classes": r.classes_present} for r in results],
            processed_filenames=list(processed_filenames),
            completed=True,
        )
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
