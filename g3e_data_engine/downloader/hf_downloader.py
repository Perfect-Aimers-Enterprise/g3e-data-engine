"""
HuggingFace `datasets`-backed downloader.

This is the reference implementation for `kind: huggingface` sources in
configs/datasets.yaml. It streams a dataset (so it never buffers the whole
remote dataset in memory), filters rows down to the target classes, and
stops as soon as each class has hit its per-run budget — which is exactly
what keeps v1 from pulling down more than it needs.

Streaming + resume behavior:
  - Every accepted image is saved to disk AND recorded in
    `<dest_dir>/_progress.json` (see downloader/progress.py) immediately —
    never buffered until the whole source finishes. A crash loses at most
    the single in-flight image.
  - If a source's manifest already has progress in it (from an earlier,
    interrupted run), this downloader picks up from `row_offset` (via the
    HF `datasets` streaming API's own `.skip()`) instead of re-scanning
    rows it already looked at, and reduces its per-class targets by
    whatever's already been satisfied.
  - If the streaming iterator itself raises mid-run (a dropped connection,
    a transient HF Hub error, etc.), this returns whatever was downloaded
    so far rather than raising — the caller (Pipeline._download_stage) can
    still use those images, and the source can simply be re-run later to
    pick up where it left off.

NOTE: this module imports `datasets`/`huggingface_hub` lazily (inside the
method, not at module import time) so the rest of the engine — config
loading, priority allocation, the FastAPI app, tests — works fully offline
without requiring those (larger) packages or network access at all.
"""
from __future__ import annotations

from pathlib import Path

from g3e_data_engine.downloader.base import (
    Downloader,
    DownloadRequest,
    DownloadedImage,
    register,
)
from g3e_data_engine.core.credentials import get_token, require_token
from g3e_data_engine.downloader.progress import load_progress, save_progress


class _NullBar:
    """Minimal stand-in for a tqdm progress bar if tqdm truly can't be imported."""

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


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

        try:
            from tqdm import tqdm
        except ImportError:  # tqdm is a core dependency, but never let a missing
            def tqdm(*args, **kwargs):  # progress bar be a hard failure
                return _NullBar()

        from g3e_data_engine.core.config import load_engine_config

        cfg = load_engine_config()
        source = cfg.datasets.sources[request.source_name]
        if not source.hf_repo:
            raise ValueError(
                f"Source '{request.source_name}' has no hf_repo set in "
                "configs/datasets.yaml — fill it in before enabling this source."
            )

        # HF_TOKEN (or whatever `auth.token_env` points to) is optional for
        # public datasets — without it you just get the standard
        # "unauthenticated requests" warning + lower rate limits from HF's
        # own client. It becomes mandatory only if this source's config sets
        # `auth.required: true` (e.g. a gated/private dataset).
        token = require_token("huggingface", source) if source.auth.required else get_token("huggingface", source)

        dest = Path(request.dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        progress = load_progress(dest)
        results: list[DownloadedImage] = [
            DownloadedImage(local_path=r["path"], source_name=request.source_name, classes_present=r["classes"])
            for r in progress.get("images", [])
            if Path(r["path"]).exists()  # tolerate a file the user deleted between runs
        ]
        per_class_counts: dict[str, int] = dict(progress.get("per_class_counts", {}))
        row_offset: int = progress.get("row_offset", 0)

        if results:
            print(
                f"  [resume] {request.source_name}: found {len(results)} previously downloaded "
                f"image(s) on disk (row_offset={row_offset}) — continuing instead of restarting."
            )

        remaining = {
            c: max(0, target - per_class_counts.get(c, 0)) for c, target in request.target_classes.items()
        }
        if all(v <= 0 for v in remaining.values()):
            print(f"  [{request.source_name}] already satisfied from a previous run — nothing to download.")
            return results

        # streaming=True: rows are pulled lazily, one at a time, off the wire.
        # This is what stops v1 from ever materializing a full remote dataset
        # on disk just to grab a few thousand images from it.
        ds = load_dataset(source.hf_repo, split="train", streaming=True, token=token)
        if row_offset:
            ds = ds.skip(row_offset)

        rows_seen_this_call = 0
        bar = tqdm(desc=f"{request.source_name}", unit="img", initial=len(results), leave=True)

        try:
            for row in ds:
                rows_seen_this_call += 1
                if all(v <= 0 for v in remaining.values()):
                    break  # every target class already satisfied

                row_classes = _extract_class_names(row, source.classes, source.class_map)
                useful = [c for c in row_classes if remaining.get(c, 0) > 0]
                if not useful:
                    continue

                img = row.get("image")
                if img is None:
                    continue

                out_path = dest / f"{request.source_name}_{row_offset + rows_seen_this_call:07d}.jpg"
                try:
                    img.convert("RGB").save(out_path, format="JPEG", quality=95)
                except Exception:
                    continue  # unreadable/corrupt row — skip rather than fail the whole run

                for c in useful:
                    remaining[c] = max(0, remaining[c] - 1)
                    per_class_counts[c] = per_class_counts.get(c, 0) + 1

                results.append(
                    DownloadedImage(
                        local_path=str(out_path),
                        source_name=request.source_name,
                        classes_present=row_classes,
                        raw_annotations=row.get("objects", row.get("annotations", [])) or [],
                    )
                )
                bar.update(1)

                # Written after EVERY image — not buffered until the source
                # finishes. A crash on the very next line loses only this
                # one image's worth of progress.
                save_progress(
                    dest,
                    row_offset=row_offset + rows_seen_this_call,
                    per_class_counts=per_class_counts,
                    images=[{"path": r.local_path, "classes": r.classes_present} for r in results],
                )

                if len(results) >= source.max_images:
                    break

        except Exception as exc:
            bar.close()
            print(
                f"  [{request.source_name}] streaming interrupted after {len(results)} image(s): {exc}. "
                "Everything downloaded so far is saved to disk and recorded — re-run to resume "
                "from here rather than starting over."
            )
            return results

        bar.close()
        save_progress(
            dest,
            row_offset=row_offset + rows_seen_this_call,
            per_class_counts=per_class_counts,
            images=[{"path": r.local_path, "classes": r.classes_present} for r in results],
            completed=True,
        )
        return results


def _extract_class_names(row: dict, allowed: list[str], class_map: dict[str, str] | None = None) -> list[str]:
    """
    Best-effort extraction of which of `allowed` (G3E) classes appear in a HF
    dataset row. Different HF vision datasets use different schemas
    (categories, objects.category, label, etc.) — extend this function, not
    the downloader class, when wiring up a new dataset's schema.

    `class_map` translates a source's own label spelling (e.g. "GUN") to the
    G3E class name (e.g. "gun") — see configs/datasets.yaml -> sources.*.class_map.
    Sources that already use G3E's exact class names can leave this empty.
    """
    class_map = class_map or {}

    def _translate(raw: str) -> str:
        return class_map.get(raw, raw)

    found: set[str] = set()

    label = row.get("label")
    if isinstance(label, str):
        translated = _translate(label)
        if translated in allowed:
            found.add(translated)

    for key in ("objects", "annotations"):
        objs = row.get(key)
        if isinstance(objs, dict) and "category" in objs:
            for cat in objs["category"]:
                if isinstance(cat, str):
                    translated = _translate(cat)
                    if translated in allowed:
                        found.add(translated)
        elif isinstance(objs, list):
            for obj in objs:
                cat = obj.get("category") if isinstance(obj, dict) else None
                if isinstance(cat, str):
                    translated = _translate(cat)
                    if translated in allowed:
                        found.add(translated)

    return sorted(found)
