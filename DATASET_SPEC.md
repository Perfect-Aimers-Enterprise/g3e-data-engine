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
| Resolution  | `min_shorter_side=416` | `min(width, height)` is smaller — checked against the SHORTER side, not width/height independently |
| Blur        | `blur_threshold=90` (Laplacian variance) | score is below the threshold |
| Brightness  | `min_brightness=35`, `max_brightness=250` | mean grayscale value outside range |
| Corruption  | n/a | file fails to open/decode |

**Why shorter-side, not width-AND-height:** an earlier version checked
`width < 640 or height < 640`, which effectively requires *both* dimensions
to clear the same bar — only near-square images satisfy that. A completely
ordinary 640×480 or 480×640 photo (the single most common shape in
datasets like COCO) failed it, which in production silently rejected the
large majority of otherwise-valid downloaded images with no visibility into
why until someone went digging through `metadata/stats.json`. Checking
`min(width, height)` against one threshold is aspect-ratio-independent and
matches how "is this image high-enough-resolution to be useful" is actually
judged. The Validate stage now also prints a live rejection breakdown
(`low_resolution` / `blurry` / `dark_or_bright` / `corrupted` counts) **and
the measured p10/median/p90 of shorter-side and blur score for the actual
batch**, printed next to each threshold, as soon as it runs — rather than
only being discoverable after the fact via `metadata/stats.json`, and
rather than requiring another guess at what the right threshold should be.
Tune thresholds from those measured numbers for a specific source, not from
a number that sounds reasonable in the abstract — that's how this engine
ended up with an overly strict default more than once.

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

`license` is a machine-readable object, not a free-text string:

```yaml
license:
  name: "CC BY 4.0"
  verified: true
  url: "https://huggingface.co/datasets/detection-datasets/coco"
```

**`verified: false` is the default and is never inferred automatically** —
not from the source's own claims, not from a similar/related dataset's
license, not from where it's hosted. A human reviews the actual terms and
flips it to `true`. Until then, `EngineConfig.validate_sources_ready()`
refuses to download from that source at all — see README.md "Source
readiness" for the exact failure mode. This is why v1 ships with `weapons`
and `fire_smoke` present in `configs/datasets.yaml` but not yet usable: their
`hf_repo`/`project` are filled in and correct, but nobody has independently
confirmed their redistribution terms yet.

Never copy a license string from one dataset onto a different one, even a
closely related one (e.g. a Roboflow-derived HF re-upload does not
automatically carry the same terms as the original Roboflow project).

## 12. Credentials

No token, API key, or other secret is ever stored in `configs/*.yaml`,
`metadata/*.json`, or anywhere else in this repo. Only *which environment
variable to read* is configurable (`SourceDef.auth.token_env`); the value
itself always comes from the process environment (optionally via a
gitignored local `.env` file). See `g3e_data_engine/core/credentials.py` and
README.md "Credentials."

A source's `auth.required: true` means the engine raises
`MissingCredentialError` immediately if that token is unset, rather than
letting an anonymous request fail confusingly partway through (this is the
default for `kind: roboflow`, which has no anonymous read path at all).

## 13. Preflight

Before `Pipeline.run(dry_run=False, ...)` downloads anything, it checks
every enabled source's downloader dependency (is the package importable?),
repository/project reference, and license status — all offline, no network
calls, milliseconds. See `g3e_data_engine/core/preflight.py` and README.md
"Preflight checks." This exists specifically to catch a missing downloader
package (e.g. `roboflow` not installed) immediately, rather than after
other sources have already spent real time downloading.

## 14. Category/label decoding

HF object-detection datasets frequently store `objects.category` (and
sometimes a bare `label`) as an **integer** `ClassLabel` id, not a string —
resolvable via that field's `Features` metadata (`.names`), not from row
data itself. `g3e_data_engine/downloader/hf_downloader.py` resolves this
mapping once per source, before scanning any rows, and applies it ahead of
`class_map` translation. A source whose categories are already plain
strings (no `ClassLabel`) works unchanged — decoding is a no-op in that
case. See `_resolve_label_names` / `_extract_class_names` in that module,
and `tests/test_hf_label_decoding.py` for the exact schema shapes handled.

If a source's names table can't be resolved at all, the downloader logs a
warning instead of silently scanning; a `max_rows_scanned` safety cap
(`SourceDef.max_rows_scanned`, default 50,000) also stops a scan that's
clearly not finding matches, rather than running indefinitely.

## 14a. Config caching

`load_engine_config()` caches the parsed config per `configs_dir`, but the
cache is fingerprinted on each of the four config files' modification time
and size — editing any of them and calling `load_engine_config()` again
always sees the edit, including within the same long-running process (a
notebook kernel, an API server). There is no manual cache-invalidation step
required for normal use; `clear_config_cache()` exists as an explicit
escape hatch (mainly useful in tests). See
`tests/test_config_cache_invalidation.py`.

## 15. Download progress and resumability

Every downloader writes each accepted image to disk and records it in a
per-source `<dest_dir>/_progress.json` manifest immediately after — not
batched until the source finishes. On a re-run, a downloader that finds an
existing manifest resumes from it (skipping already-satisfied classes, and
for streaming sources, already-scanned rows) instead of starting over. If a
source's download is interrupted, `Pipeline._download_stage()` records the
failure in `PipelineRunResult.failed_sources` and continues with the
remaining sources — one source failing never discards what other sources
already produced. See `g3e_data_engine/downloader/progress.py`.

## 16. Publishing releases

`g3e_data_engine/exporters/hf_uploader.py` (and `scripts/upload_hf.py`) push
a release folder to the Hugging Face Hub. Uploading is **never** automatic —
`Pipeline.run(export=True)` alone only produces a local zip under
`datasets/releases/`; publishing anywhere requires the caller to pass
`upload_to_hf=<repo_id>` explicitly (Python), `--upload-to-hf <repo_id>`
(CLI), or set `upload_to_hf` in the `/pipeline/run` request body (API).
Uploading always requires a Hugging Face token with write access — there is
no anonymous path, unlike downloading a public dataset.
