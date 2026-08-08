# DATASET_SPEC.md

This is the contract every part of `g3e-data-engine` follows. If code and
this document ever disagree, that's a bug — in one of the two. Update both
together.

## 1. Supported image formats

- Input (from sources): whatever the source provides; converted to `jpg` on ingest.
- Stored/exported: `.jpg` (quality 95) for images. `.png` is accepted as input
  during validation (see `scripts/validate.py`) but the pipeline normalizes to jpg.

## 2. Target annotation format

- **YOLO** (`class_id x_center y_center width height`, all normalized to
  `[0, 1]`, one `.txt` file per image, same basename as the image).
- Source annotations are typically COCO-style boxes
  (`[x_min, y_min, width, height]`, absolute pixels) and are converted via
  `g3e_data_engine/converters/coco_to_yolo.py`.
- If you need a second target format (e.g. native COCO JSON for a specific
  trainer), add a new sibling converter module — do not overload this one.

## 3. Class definitions and IDs (v1 — frozen)

| id | name    | priority tier | rationale |
|----|---------|:-:|-----------|
| 0  | person  | 1 | safety-critical |
| 1  | fire    | 1 | safety-critical, rare in generic datasets |
| 2  | gun     | 1 | safety-critical, rare in generic datasets |
| 3  | smoke   | 2 | co-occurs with fire |
| 4  | knife   | 2 | safety-relevant |
| 5  | car     | 3 | abundant in generic datasets |
| 6  | dog     | 4 | abundant, non-safety-critical |
| 7  | cat     | 4 | abundant, non-safety-critical |

**Rule: never reorder or reuse an existing `id`.** Downstream YOLO labels
bake in the `id`, not the name — changing what id 3 means silently corrupts
every already-exported release. To add a class, append it with the next free
`id` (8, 9, ...) in `configs/classes.yaml`. See `docs/FUTURE_UPDATES.md`.

## 4. Image quality requirements (v1 defaults — `configs/processing.yaml`)

| Check       | Default | Rejected when |
|-------------|---------|---------------|
| Resolution  | `min_width=640`, `min_height=640` | either dimension is smaller |
| Blur        | `blur_threshold=90` (Laplacian variance) | score is below the threshold |
| Brightness  | `min_brightness=35`, `max_brightness=250` | mean grayscale value outside range |
| Corruption  | n/a | file fails to open/decode |

Rejected images are counted (see `metadata/stats.json`) but not silently
dropped without a trace — every rejection has a `reason` string attached
(`ValidationResult.reasons`).

## 5. Duplicate handling

Perceptual hash (`imagehash.phash`), Hamming distance threshold `5` by
default (`configs/processing.yaml -> duplicates`). Runs **after** validation
and **before** the split, so no near-duplicate can end up split across
train/val/test (which would leak signal between them).

## 6. Dataset split policy

Deterministic, seeded (`configs/processing.yaml -> split.seed`, default
`42`), by **image id**, default ratios `80/10/10` for train/val/test. Same
input set + same seed = same split, always — this is what makes re-running
the pipeline reproducible.

## 7. Metadata schema

`metadata/metadata.json` — one record per accepted, deduplicated image:

```json
{
  "id": "000001",
  "dataset": "coco",
  "source": "coco",
  "classes": ["person", "car"],
  "width": 1280,
  "height": 720,
  "split": "train"
}
```

`metadata/classes.json` — the frozen v1 class list (id + name only, no
priority info — priority is a *download-time* concept, not a dataset
property, so it deliberately doesn't leak into the released metadata).

`metadata/stats.json` — see `g3e_data_engine/utils/statistics.py`
`DatasetStats.to_dict()` for the exact shape (rejection breakdown, class
counts, split counts).

`metadata/versions.json` — append-only log, one entry per completed +
exported run:

```json
{
  "version": "1.1.0",
  "created_at": "2026-08-08T12:00:00+00:00",
  "image_count": 5893,
  "notes": "Pipeline run: 5893 images"
}
```

## 8. Naming conventions

- Image files: `<source_name>_<7-digit index>.jpg` (e.g. `coco_0000042.jpg`).
- Label files: same basename, `.txt` extension, same directory structure
  mirrored under `labels/` instead of `images/`.
- Release folders: `g3e-vision-dataset-v<version>/` under `datasets/releases/`.

## 9. Versioning policy

- Semantic-ish: `1.<minor>.0`. Minor bumps on every completed, exported
  pipeline run (`bump_version()` in `g3e_data_engine/utils/metadata.py`).
- **Releases are immutable.** `export_release()` refuses to overwrite an
  existing version folder — if you need to redo a release, bump the version
  instead of deleting and re-exporting the same one. This keeps
  `versions.json` an honest, append-only history.
- A version bump happens **once per run, at the end** — never mid-run —
  so `versions.json` never records a run that didn't finish.

## 10. Acceptance / rejection rules

An image is accepted into the final dataset only if, in order:

1. It downloads and decodes successfully (not corrupted).
2. It passes resolution, blur, and brightness thresholds (§4).
3. It is not a near-duplicate of an already-accepted image (§5).

Rejected images are never written into `metadata/metadata.json` — only
counted in `metadata/stats.json`'s rejection breakdown.

## 11. Licensing requirements for imported sources

Every entry in `configs/datasets.yaml -> sources.<name>` must set `license`
to the source's actual license before `enabled: true` ships in a real
release — `"TBD — verify before enabling"` is a placeholder, not a license,
and is a signal that source still needs a human to check its terms
(redistribution rights in particular) before it's used in anything public.
