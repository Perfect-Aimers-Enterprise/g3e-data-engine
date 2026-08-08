# Future Updates (v1.1+)

This is the deliberate backlog for after v1. None of these are needed to use
v1 — they're here so "why isn't X done yet" has an answer, and so future-you
(or a contributor) doesn't have to guess whether something was forgotten or
intentionally deferred.

## Done since initial v1 draft

- [x] **Credentials system** (`core/credentials.py`) — env-var based token
  lookup (`HF_TOKEN`, `ROBOFLOW_API_KEY`, ...), optional `.env` loading,
  per-source `auth.token_env` override, `auth.required` fail-fast.
- [x] **Machine-readable license schema** (`license: {name, verified, url}`)
  + `EngineConfig.validate_sources_ready()` — an enabled source with a
  missing repo/project reference or `verified: false` aborts the whole run
  before any download happens, with a clear per-source error message.
- [x] **Second downloader kind (`roboflow`)** — proves the registry really
  is source-agnostic, not just "huggingface with extra steps." Used for the
  `fire_smoke` source (`fire-rqbio/fire-and-smoke-yikzn`, version 3 pinned).
- [x] **Per-source `class_map`** — lets a source use its own label spelling
  (e.g. `weapons`' `GUN`/`KNIFE`/`PERSON`) and have it translated to g3e's
  class names, instead of forcing every upstream dataset to already match.
- [x] **HF uploader as a library function**, not just a script
  (`exporters/hf_uploader.py` → `upload_release_to_hf()`), wired into
  `Pipeline.run(upload_to_hf=...)` and `POST /pipeline/run`, always opt-in.
- [x] **Real sourcing decisions for weapons/fire_smoke** — see README
  "Dataset sources." Both currently ship `license.verified: false` pending
  an actual human review of their terms; that's the intended state, not a
  bug — flip it once reviewed.

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
- [ ] **Progress reporting** — `POST /pipeline/run` currently blocks until the
  whole run finishes. For a 6,000-image run this is minutes, not seconds;
  add either a background-task + polling endpoint, or server-sent events.

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
