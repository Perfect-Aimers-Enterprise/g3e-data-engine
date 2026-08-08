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
        allocation = self.allocator.allocate(
            total_images=total_images,
            overrides=priority_overrides,
            available_by_class=available_by_class,
        )

        if dry_run:
            return PipelineRunResult(
                allocation=allocation,
                dry_run=True,
                notes=(
                    "Dry run — no images were downloaded. This is the per-class "
                    "download plan the engine would execute."
                ),
            )

        # Refuse to download anything if any enabled source is missing its
        # repo/project reference or has an unverified license. This check is
        # deliberately NOT part of load_engine_config() (which also backs
        # dry_run and /pipeline/allocate) — those should keep working even
        # while a source is still mid-setup. See config.py -> validate_sources_ready.
        self.config.validate_sources_ready()

        work_dir = Path(work_dir)
        raw_dir = work_dir / "raw"
        processed_dir = work_dir / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        downloaded_paths, class_by_path, source_by_path = self._download_stage(
            allocation, raw_dir
        )

        validations = validate_batch(downloaded_paths, self.config.processing.image)
        accepted_paths = [v.path for v in validations if v.accepted]

        dedup_result = find_duplicates(accepted_paths, self.config.processing.duplicates)
        final_paths = dedup_result.kept

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

        result = PipelineRunResult(
            allocation=allocation,
            validations=validations,
            duplicates_removed=len(dedup_result.duplicates),
            metadata_records=metadata_records,
            split=split_result.as_dict(),
            stats=stats.to_dict(),
            dry_run=False,
        )

        if export:
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

            if upload_to_hf:
                # Uploading is opt-in per call (upload_to_hf must be passed
                # explicitly) — export never publishes anything on its own.
                from g3e_data_engine.exporters.hf_uploader import upload_release_to_hf

                release_folder = release_zip.parent / release_zip.stem
                result.upload_url = upload_release_to_hf(
                    folder=release_folder,
                    repo_id=upload_to_hf,
                    private=hf_private,
                )

        return result

    def _download_stage(self, allocation: AllocationResult, raw_dir: Path):
        """
        Fans the per-class allocation out across every enabled source that
        can supply that class, respecting each source's own max_images cap.
        Returns (all_paths, path->classes, path->source_name).
        """
        from g3e_data_engine.downloader import get_downloader, DownloadRequest

        remaining_by_class = allocation.as_dict()
        all_paths: list[str] = []
        class_by_path: dict[str, list[str]] = {}
        source_by_path: dict[str, str] = {}

        for source_name, source in self.config.datasets.enabled_sources().items():
            target_for_source = {
                c: min(remaining_by_class.get(c, 0), source.max_images)
                for c in source.classes
                if remaining_by_class.get(c, 0) > 0
            }
            if not target_for_source:
                continue

            downloader = get_downloader(source.kind)
            request = DownloadRequest(
                source_name=source_name,
                target_classes=target_for_source,
                dest_dir=str(raw_dir / source_name),
            )
            images = downloader.download(request)

            for img in images:
                all_paths.append(img.local_path)
                class_by_path[img.local_path] = img.classes_present
                source_by_path[img.local_path] = source_name
                for c in img.classes_present:
                    if c in remaining_by_class:
                        remaining_by_class[c] = max(0, remaining_by_class[c] - 1)

        return all_paths, class_by_path, source_by_path
