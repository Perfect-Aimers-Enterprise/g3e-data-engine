"""
Pipeline orchestrator.

    Download -> Validate -> Convert -> Quality Filter -> Deduplicate
    -> Generate Metadata -> Split -> Statistics -> Export

Each stage is implemented in its own module (downloader/, validators/,
converters/, dedup/, filters/, utils/statistics.py, exporters/) — this file
just sequences them and carries state between stages. If you need to change
*what* a stage does, edit that stage's module. Only edit this file if you're
changing the *order* of stages or adding a brand-new stage.

This is the object both scripts/run_pipeline.py and the FastAPI routes call
into, so behavior stays identical whether the engine is driven from the CLI
or from HTTP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from g3e_data_engine.core.config import EngineConfig, load_engine_config
from g3e_data_engine.core.priority import PriorityAllocator, AllocationResult
from g3e_data_engine.validators.image_quality import validate_batch, ValidationResult
from g3e_data_engine.dedup.phash_dedup import find_duplicates
from g3e_data_engine.filters.split import split_ids
from g3e_data_engine.utils.statistics import DatasetStats, RejectionCounts, build_class_counts
from g3e_data_engine.utils.metadata import ImageMetadata, write_metadata, bump_version


def _rejection_breakdown(rejected: list[ValidationResult]) -> dict[str, int]:
    """
    Quick, human-readable tally of WHY images were rejected, printed live
    during the Validate stage rather than only being discoverable later by
    reading metadata/stats.json. A real production run once silently
    dropped ~93% of downloaded images to an overly strict resolution check
    with no visibility into why until someone went digging — this is the
    fix for the "why" being invisible, independent of what the actual
    threshold values are set to.
    """
    counts: dict[str, int] = {}
    for v in rejected:
        for reason in v.reasons:
            if "resolution" in reason:
                key = "low_resolution"
            elif "blurry" in reason:
                key = "blurry"
            elif "dark" in reason or "bright" in reason:
                key = "dark_or_bright"
            elif "corrupted" in reason:
                key = "corrupted"
            else:
                key = "other"
            counts[key] = counts.get(key, 0) + 1
    return counts


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(pct * (len(s) - 1)))))
    return s[idx]


def _measured_diagnostics(validations: list[ValidationResult], thresholds) -> str:
    """
    Actual MEASURED distributions of shorter-side and blur score across this
    batch — accepted vs rejected — printed alongside the breakdown counts.

    This exists because guessing at a threshold from first principles (as
    an earlier version of this engine did — twice) isn't a substitute for
    looking at what a real source's images actually look like. These
    numbers are what should drive any further tuning of
    configs/processing.yaml, not another guess.
    """
    shorter_sides = [min(v.width, v.height) for v in validations if v.width and v.height]
    blur_scores = [v.blur_score for v in validations if v.width and v.height]

    if not shorter_sides:
        return "    (no readable images to measure)"

    lines = [
        f"    measured shorter-side: p10={_percentile(shorter_sides, 0.10):.0f} "
        f"median={_percentile(shorter_sides, 0.50):.0f} p90={_percentile(shorter_sides, 0.90):.0f} "
        f"(threshold: min_shorter_side={thresholds.min_shorter_side})",
        f"    measured blur score:   p10={_percentile(blur_scores, 0.10):.1f} "
        f"median={_percentile(blur_scores, 0.50):.1f} p90={_percentile(blur_scores, 0.90):.1f} "
        f"(threshold: blur_threshold={thresholds.blur_threshold})",
    ]
    return "\n".join(lines)


@dataclass
class PipelineRunResult:
    allocation: AllocationResult
    validations: list[ValidationResult] = field(default_factory=list)
    duplicates_removed: int = 0
    metadata_records: list[ImageMetadata] = field(default_factory=list)
    split: dict[str, list[str]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    dry_run: bool = True
    notes: str = ""
    upload_url: str | None = None
    failed_sources: dict[str, str] = field(default_factory=dict)


class Pipeline:
    """
    Usage (library):

        from g3e_data_engine import load_engine_config, Pipeline

        cfg = load_engine_config()
        pipeline = Pipeline(cfg)

        # Plan only — no downloads, no disk writes. Good for CI / sanity checks
        # and for answering "what would this run fetch?" before committing to it.
        result = pipeline.run(dry_run=True)

        # Real run: downloads (respecting the priority budget), validates,
        # dedups, writes metadata + splits + stats, and (optionally) exports.
        result = pipeline.run(
            dry_run=False,
            total_images=3000,
            priority_overrides={"fire": 2.0},
            export=True,
        )
    """

    def __init__(self, config: EngineConfig | None = None):
        self.config = config or load_engine_config()
        self.allocator = PriorityAllocator(self.config)

    def run(
        self,
        dry_run: bool = True,
        total_images: int | None = None,
        priority_overrides: dict[str, float] | None = None,
        available_by_class: dict[str, int] | None = None,
        export: bool = False,
        upload_to_hf: str | None = None,
        hf_private: bool = True,
        work_dir: str | Path = "datasets",
    ) -> PipelineRunResult:
        from g3e_data_engine import __version__

        print(f"g3e-data-engine v{__version__}")

        allocation = self.allocator.allocate(
            total_images=total_images,
            overrides=priority_overrides,
            available_by_class=available_by_class,
        )
        if allocation.notes:
            print(f"[priority] {allocation.notes}")

        if dry_run:
            return PipelineRunResult(
                allocation=allocation,
                dry_run=True,
                notes=(
                    "Dry run — no images were downloaded. This is the per-class "
                    "download plan the engine would execute."
                ),
            )

        # Preflight: refuse to download anything if any enabled source is
        # missing its downloader dependency (e.g. `roboflow` not installed),
        # missing its repo/project reference, or has an unverified license.
        # Runs in milliseconds, no network calls — this is what stops a run
        # from burning minutes/hours on other sources before discovering a
        # missing package on the LAST source it reaches. This check is
        # deliberately NOT part of load_engine_config() (which also backs
        # dry_run and /pipeline/allocate) — those should keep working even
        # while a source is still mid-setup. See config.py -> validate_sources_ready.
        self.config.validate_sources_ready()

        work_dir = Path(work_dir)
        raw_dir = work_dir / "raw"
        processed_dir = work_dir / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        downloaded_paths, class_by_path, source_by_path, failed_sources = self._download_stage(
            allocation, raw_dir
        )

        print(f"\n=== STAGE: Validate ({len(downloaded_paths)} downloaded image(s)) ===")
        validations = validate_batch(downloaded_paths, self.config.processing.image)
        accepted_paths = [v.path for v in validations if v.accepted]
        print(f"    accepted: {len(accepted_paths)} / {len(validations)}")
        if validations:
            print(_measured_diagnostics(validations, self.config.processing.image))
        rejected_now = [v for v in validations if not v.accepted]
        if rejected_now:
            breakdown = _rejection_breakdown(rejected_now)
            print(f"    rejected breakdown: {breakdown}")
            if len(accepted_paths) < 0.5 * len(validations):
                print(
                    "    [warning] most downloaded images were rejected — compare the measured "
                    "percentiles above against configs/processing.yaml's thresholds to see which "
                    "check is actually doing the rejecting before changing anything."
                )

        print("\n=== STAGE: Deduplicate ===")
        dedup_result = find_duplicates(accepted_paths, self.config.processing.duplicates)
        final_paths = dedup_result.kept
        print(f"    removed {len(dedup_result.duplicates)} duplicate(s); {len(final_paths)} unique image(s) remain")

        print("\n=== STAGE: Metadata & Split ===")
        metadata_records = [
            ImageMetadata(
                id=Path(p).stem,
                dataset=source_by_path.get(p, "unknown"),
                source=source_by_path.get(p, "unknown"),
                classes=class_by_path.get(p, []),
                width=next((v.width for v in validations if v.path == p), 0),
                height=next((v.height for v in validations if v.path == p), 0),
            )
            for p in final_paths
        ]

        split_result = split_ids([m.id for m in metadata_records], self.config.processing.split)
        id_to_split = {i: "train" for i in split_result.train}
        id_to_split.update({i: "val" for i in split_result.val})
        id_to_split.update({i: "test" for i in split_result.test})
        for m in metadata_records:
            m.split = id_to_split.get(m.id, "unassigned")
        print(
            f"    split -> train={len(split_result.train)} "
            f"val={len(split_result.val)} test={len(split_result.test)}"
        )

        print("\n=== STAGE: Statistics ===")
        rejection = RejectionCounts(
            total_seen=len(validations),
            accepted=len(final_paths),
            rejected_blurry=sum(1 for v in validations if any("blurry" in r for r in v.reasons)),
            rejected_dark_or_bright=sum(
                1 for v in validations if any(("dark" in r or "bright" in r) for r in v.reasons)
            ),
            rejected_low_res=sum(1 for v in validations if any("resolution" in r for r in v.reasons)),
            rejected_corrupted=sum(1 for v in validations if any("corrupted" in r for r in v.reasons)),
            rejected_duplicate=len(dedup_result.duplicates),
        )
        stats = DatasetStats(
            rejection=rejection,
            class_counts=build_class_counts([{"classes": m.classes} for m in metadata_records]),
            split_counts={
                "train": len(split_result.train),
                "val": len(split_result.val),
                "test": len(split_result.test),
            },
        )

        metadata_dir = Path("metadata")
        write_metadata(metadata_records, metadata_dir / "metadata.json")
        import json

        with open(metadata_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2)
        print(f"    {len(metadata_records)} image(s) accepted into the dataset")

        if failed_sources:
            print(
                f"\n[warning] {len(failed_sources)} source(s) failed and were skipped "
                f"(their partial progress is preserved on disk — re-run to resume them): "
                f"{list(failed_sources)}"
            )

        result = PipelineRunResult(
            allocation=allocation,
            validations=validations,
            duplicates_removed=len(dedup_result.duplicates),
            metadata_records=metadata_records,
            split=split_result.as_dict(),
            stats=stats.to_dict(),
            dry_run=False,
            failed_sources=failed_sources,
        )

        if export:
            print("\n=== STAGE: Export ===")
            entry = bump_version(
                metadata_dir / "versions.json",
                notes=f"Pipeline run: {len(metadata_records)} images",
                image_count=len(metadata_records),
            )
            from g3e_data_engine.exporters.release_exporter import export_release

            release_zip = export_release(
                processed_dir=processed_dir,
                metadata_dir=metadata_dir,
                releases_dir=work_dir / "releases",
                version=entry["version"].lstrip("1."),
            )
            print(f"    wrote {release_zip}")

            if upload_to_hf:
                # Uploading is opt-in per call (upload_to_hf must be passed
                # explicitly) — export never publishes anything on its own.
                print(f"\n=== STAGE: Upload to Hugging Face ({upload_to_hf}) ===")
                from g3e_data_engine.exporters.hf_uploader import upload_release_to_hf

                release_folder = release_zip.parent / release_zip.stem
                result.upload_url = upload_release_to_hf(
                    folder=release_folder,
                    repo_id=upload_to_hf,
                    private=hf_private,
                )
                print(f"    uploaded -> {result.upload_url}")

        return result

    def _download_stage(self, allocation: AllocationResult, raw_dir: Path):
        """
        Fans the per-class allocation out across every enabled source that
        can supply that class, respecting each source's own max_images cap.

        Each source's `downloader.download(...)` call is wrapped in its own
        try/except: if one source fails (network error, an issue the
        downloader itself didn't swallow, etc.), that failure is recorded
        and the loop moves on to the remaining sources instead of aborting
        the whole run — whatever that source already saved to disk (via its
        own incremental progress manifest, see downloader/progress.py) is
        kept, and the source can simply be re-run later to resume.

        Returns (all_paths, path->classes, path->source_name, failed_sources).
        """
        from g3e_data_engine.downloader import get_downloader, DownloadRequest

        remaining_by_class = allocation.as_dict()
        all_paths: list[str] = []
        class_by_path: dict[str, list[str]] = {}
        source_by_path: dict[str, str] = {}
        failed_sources: dict[str, str] = {}

        enabled = self.config.datasets.enabled_sources()
        print(f"\n=== STAGE: Download ({len(enabled)} enabled source(s)) ===")

        for source_name, source in enabled.items():
            target_for_source = {
                c: min(remaining_by_class.get(c, 0), source.max_images)
                for c in source.classes
                if remaining_by_class.get(c, 0) > 0
            }
            if not target_for_source:
                print(f"-> {source_name}: skipped (no remaining budget for its classes)")
                continue

            print(f"-> {source_name}: starting (kind={source.kind}, target={target_for_source})")
            try:
                downloader = get_downloader(source.kind)
                request = DownloadRequest(
                    source_name=source_name,
                    target_classes=target_for_source,
                    dest_dir=str(raw_dir / source_name),
                )
                images = downloader.download(request)
            except Exception as exc:
                failed_sources[source_name] = str(exc)
                print(f"x  {source_name}: FAILED — {exc}")
                print(
                    f"   Anything already saved for '{source_name}' before the failure is kept "
                    f"on disk (see {raw_dir / source_name}/_progress.json) — re-run to resume it."
                )
                continue

            for img in images:
                all_paths.append(img.local_path)
                class_by_path[img.local_path] = img.classes_present
                source_by_path[img.local_path] = source_name
                for c in img.classes_present:
                    if c in remaining_by_class:
                        remaining_by_class[c] = max(0, remaining_by_class[c] - 1)

            print(f"<- {source_name}: done — {len(images)} image(s) downloaded")

        return all_paths, class_by_path, source_by_path, failed_sources
