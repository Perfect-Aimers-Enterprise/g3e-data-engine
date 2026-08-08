# Future Updates (v1.1+)

This is the deliberate backlog for after v1. None of these are needed to use
v1 — they're here so "why isn't X done yet" has an answer, and so future-you
(or a contributor) doesn't have to guess whether something was forgotten or
intentionally deferred.

## Near-term (v1.1)

- [ ] **`scripts/generate_classes_json.py`** — right now `metadata/classes.json`
  is hand-derived from `configs/classes.yaml`; write the one-line script that
  regenerates it, and call it from `scripts/run_pipeline.py` automatically.
- [ ] **Real `hf_repo` values** for the `weapons` and `fire_smoke` sources in
  `configs/datasets.yaml` — currently placeholders. Needs a human to pick and
  license-check actual HF datasets (or another source kind) for `gun`,
  `knife`, `fire`, `smoke`.
- [ ] **Convert stage wiring in `Pipeline.run()`** — `converters/coco_to_yolo.py`
  exists and is tested, but `Pipeline._download_stage()` doesn't yet call it
  on `DownloadedImage.raw_annotations`; currently `Pipeline.run()` produces
  metadata + splits but not the `labels/*.txt` files. Wire this in once a real
  source's `raw_annotations` schema is confirmed (schemas vary per HF dataset).
- [ ] **Progress reporting** — `POST /pipeline/run` currently blocks until the
  whole run finishes. For a 6,000-image run this is minutes, not seconds;
  add either a background-task + polling endpoint, or server-sent events.

## Medium-term (v1.2+)

- [ ] **New downloader kinds**: raw HTTP/zip source, Roboflow, local-folder
  import (for hand-collected CCTV footage from `g3e-app`). See
  `docs/ARCHITECTURE.md` → "Adding a brand-new kind of source".
- [ ] **Raise `global_max_images`** once training on v1's ~6k images shows
  which classes need more data — do this deliberately, class-by-class via
  `configs/priority.yaml` overrides first, before raising the global ceiling.
- [ ] **New classes**: `helmet`, `mask`, `crowd_density` (a continuous quantity
  rather than a box — would need a schema change, not just a new class id).
- [ ] **Active-learning style prioritization** — instead of static tier
  weights, use a first-round trained model's confusion matrix to re-weight
  `configs/priority.yaml` for round 2 (boost the classes the model gets
  wrong most).

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
