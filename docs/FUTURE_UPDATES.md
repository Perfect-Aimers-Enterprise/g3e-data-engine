# Future Updates (v1.1+)

This is the deliberate backlog for after v1. None of these are needed to use
v1 — they're here so "why isn't X done yet" has an answer, and so future-you
(or a contributor) doesn't have to guess whether something was forgotten or
intentionally deferred.

## Done since initial v1 draft

- [x] **Measured diagnostics in the Validate stage** — printing an
  accept/reject count alone wasn't enough to tune anything; the stage now
  prints the actual measured p10/median/p90 of shorter-side and blur score
  for the batch, right next to each threshold, so tuning
  `configs/processing.yaml` is based on a specific source's real numbers
  instead of another guess. Covered by
  `tests/test_pipeline.py::test_validate_stage_prints_measured_diagnostics`.
- [x] **Version banner** — every `Pipeline.run()` and `scripts/check_sources.py`
  invocation now prints `g3e-data-engine v{version}` as its first line, so
  a stale-cached-code situation (a real, repeatedly-hit Colab/Jupyter
  gotcha — replacing files on disk doesn't reload already-imported modules
  without a runtime restart) is immediately diagnosable instead of looking
  like a fix silently not working.
- [x] **Raised v1 capacity** — `global_max_images` 8,000 → 50,000 (the
  ceiling a single run can request; the default request size,
  `priority.yaml`'s `budget.total_images`, stays at a modest 6,000).
  Per-source `max_images` raised to match each source's actual known size
  (`coco`: 20,000 of ~117k rows; `weapons`: 7,000 of ~7,615 rows;
  `fire_smoke`: 3,500 of ~3,884 images) rather than the earlier, overly
  conservative caps. `priority.yaml`'s `max_per_class` raised 2,500 →
  20,000 so a large `total_images` request isn't needlessly clamped
  per-class.

- [x] **Resolution check fixed to use the shorter side, not width-AND-height**
  — the root cause of a real production run accepting only 48/714 (6.7%)
  of downloaded images. The old `min_width=640, min_height=640` check
  required BOTH dimensions to clear 640, which only near-square images
  satisfy — an ordinary 640x480 photo (COCO's most common shape) failed it.
  Replaced with a single `min_shorter_side` (default 416) checked against
  `min(width, height)`, which is aspect-ratio-independent. Also added a
  live rejection breakdown printed during the Validate stage (was
  previously only visible by reading `metadata/stats.json` after the
  fact), plus a warning when acceptance rate is under 50%. Covered by
  `tests/test_validators.py`.
- [x] **Friendlier Roboflow error messages** — an invalid/revoked API key
  used to surface as Roboflow's raw JSON error body
  (`{"error":{"message":"This API key does not exist..."}}`) with no
  indication of what to fix. `_friendly_roboflow_error()` translates
  recognizable cases (bad/revoked key, project/version not found) into a
  message naming the actual next step.
- [x] **Config cache invalidation fix** — the root cause of a real
  reported bug: `load_engine_config()` used a plain `functools.lru_cache`
  keyed only on the configs directory path, so editing
  `configs/datasets.yaml` in the same running process (e.g. flipping
  `license.verified: true` after actually reviewing a source, in the same
  notebook kernel that had already loaded the config once) kept silently
  returning the pre-edit config — the person would see a preflight failure
  quoting license text they'd already replaced. Fixed by fingerprinting
  each config file's `(mtime_ns, size)` and auto-invalidating on change;
  `clear_config_cache()` added as an explicit escape hatch. Covered by
  `tests/test_config_cache_invalidation.py`.

- [x] **Category/label decoding fix** — the root cause of a real production
  failure: HF object-detection datasets store `objects.category` (and
  sometimes `label`) as an integer `ClassLabel` id, not a string; comparing
  it directly against G3E's string class names silently matched nothing on
  every row, so a source would stream its entire dataset (30+ minutes, in
  the reported case) and accept zero images with no indication why. Fixed
  in `downloader/hf_downloader.py` (`_resolve_label_names` /
  `_extract_class_names`), covered by `tests/test_hf_label_decoding.py`
  using the real `datasets.Features`/`ClassLabel` classes. Also added a
  `max_rows_scanned` safety cap (default 50,000) and a rows-scanned
  progress bar (not just accepted-images) so a 0%-match-rate source is
  visible in seconds, not after 30+ minutes.
- [x] **Colab Secrets fallback for credentials** — root cause of a second
  reported issue ("I set the token but the library says it's missing"):
  Colab's Secrets panel doesn't export to `os.environ`. `get_token()` now
  falls back to `google.colab.userdata.get(...)` under the same variable
  name when the env var isn't set, with a specific hint if the secret
  exists but "Notebook access" isn't toggled on. Covered by
  `tests/test_colab_credentials.py` using a fake `google.colab` module
  (the real package only exists inside Colab).
- [x] **Preflight checks** (`core/preflight.py`) — dependency (is the
  package a source's `kind` needs actually importable?), repository
  (repo/project reference set?), and license (verified?) — checked in
  milliseconds, with no network calls, before `Pipeline.run(dry_run=False)`
  downloads anything. This is what catches a missing `roboflow` install
  immediately instead of after other sources have already spent
  minutes/hours downloading. Exposed standalone via `scripts/check_sources.py`.
- [x] **Download progress + incremental, resumable saves**
  (`downloader/progress.py`) — every downloader writes each accepted image
  to disk and records it in a per-source `_progress.json` manifest
  immediately (not batched), shows a live `tqdm` progress bar, and resumes
  from that manifest on a re-run instead of starting over (skipping
  satisfied classes, and for HF streaming sources, skipping already-scanned
  rows via `.skip()`).
- [x] **Per-source failure isolation** — `Pipeline._download_stage()` wraps
  each source's `downloader.download(...)` call individually; one source
  failing (network error, etc.) is recorded in `result.failed_sources` and
  the run continues with the remaining sources rather than aborting
  everything and losing already-downloaded data.
- [x] **Pipeline stage banners** — `Download`, `Validate`, `Deduplicate`,
  `Metadata & Split`, `Statistics`, `Export`, `Upload to Hugging Face` each
  print a start line + a short result line, so a long-running console or
  Colab session always shows exactly where the run is.
- [x] **Allocator transparency** — `AllocationResult.notes` explains it
  explicitly whenever `min_per_class`/`max_per_class` clamping makes the
  actual total allocated diverge noticeably from what was requested,
  instead of leaving that as a silent surprise in the raw numbers.
- [x] Credentials system (`core/credentials.py`) — env-var based token
  lookup (`HF_TOKEN`, `ROBOFLOW_API_KEY`, ...), optional `.env` loading,
  per-source `auth.token_env` override, `auth.required` fail-fast.
- [x] Machine-readable license schema (`license: {name, verified, url}`)
  + `EngineConfig.validate_sources_ready()` (now delegating to preflight).
- [x] Second downloader kind (`roboflow`), proving the registry is
  source-agnostic. Used for `fire_smoke`.
- [x] Per-source `class_map` for translating a source's native label
  spelling to g3e's class names.
- [x] HF uploader as a library function (`exporters/hf_uploader.py`),
  wired into `Pipeline.run(upload_to_hf=...)`, always opt-in.
- [x] Real sourcing decisions for weapons/fire_smoke (see README "Dataset
  sources") — both intentionally ship `license.verified: false` pending an
  actual human review of their terms.

## Near-term (v1.1)

- [ ] **License review** — actually check the terms of
  `Subh775/WeaponDetection_Grouped` and the pinned `fire-rqbio/fire-and-smoke-yikzn`
  v3 project, then flip `license.verified: true` for whichever pass review.
  This is the one blocker standing between v1's config and a real run.
- [ ] **`scripts/generate_classes_json.py`** — right now `metadata/classes.json`
  is hand-derived from `configs/classes.yaml`; write the one-line script that
  regenerates it, and call it from `scripts/run_pipeline.py` automatically.
- [ ] **Convert stage wiring in `Pipeline.run()`** — `converters/coco_to_yolo.py`
  exists and is tested, but `Pipeline._download_stage()` doesn't yet call it
  on `DownloadedImage.raw_annotations`; currently `Pipeline.run()` produces
  metadata + splits but not the `labels/*.txt` files. Wire this in once the
  weapons/fire_smoke sources are verified and their `raw_annotations` shape
  (COCO boxes vs. already-YOLO label files, per source) is confirmed end-to-end.
- [ ] **Progress reporting over the API** — `POST /pipeline/run` still blocks
  until the whole run finishes (the console/CLI path now has live stage +
  per-source progress; the HTTP path doesn't yet). Add either a
  background-task + polling endpoint, or server-sent events, that streams
  the same per-stage/per-source lines the console gets.
- [ ] **`_progress.json` write cost at scale** — `save_progress()` currently
  rewrites the *entire* per-source manifest (including the full `images`
  list) after every single image, which is O(n²) total I/O across a
  source's run. Fine at v1's few-thousand-images-per-source scale; revisit
  (e.g. append-only log + periodic compaction) before pushing per-source
  budgets much higher.

## Medium-term (v1.2+)

- [ ] **New downloader kinds**: Kaggle, GitHub releases, plain HTTP/zip,
  local-folder import (for hand-collected CCTV footage from `g3e-app`). See
  `docs/ARCHITECTURE.md` → "Adding a brand-new kind of source" — the
  huggingface/roboflow pair already proves the pattern holds up.
- [ ] **Raise `global_max_images`** once training on v1's ~6k images shows
  which classes need more data — do this deliberately, class-by-class via
  `configs/priority.yaml` overrides first, before raising the global ceiling.
- [ ] **New classes**: `helmet`, `mask`, `crowd_density` (a continuous quantity
  rather than a box — would need a schema change, not just a new class id).
- [ ] **Active-learning style prioritization** — instead of static tier
  weights, use a first-round trained model's confusion matrix to re-weight
  `configs/priority.yaml` for round 2 (boost the classes the model gets
  wrong most).
- [ ] **Credential rotation / multiple accounts** — `auth.token_env` already
  supports pointing a source at a non-default variable; a future update
  could add token *rotation* (fall back to a second variable if the first
  hits a rate limit) if that becomes a real bottleneck.

## Long-term / exploratory

- [ ] **Multi-label co-occurrence balancing** — v1's `PriorityAllocator`
  allocates per-class independently; it doesn't yet account for the fact
  that `smoke` and `fire` mostly co-occur in the same images, so boosting one
  effectively boosts both. Not a bug, but worth modeling explicitly if
  budgets get tight.
- [ ] **Automatic license-compatibility checks** across sources before export
  (today this is a manual checklist item in `DATASET_SPEC.md` §11).
- [ ] **Streaming export directly to HF Hub** without an intermediate local
  zip, for very large future releases.

## Explicitly NOT planned for v1.x (revisit only if requirements change)

- Video/frame-extraction ingestion (CCTV streams) — that's `g3e-app`'s
  territory until there's a concrete need for the *dataset engine itself* to
  ingest video.
- Multi-node/distributed downloading — v1's budgets are small enough that a
  single cloud GPU box is sufficient (see the README's cloud-first workflow).
