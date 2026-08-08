# g3e-data-engine

Reusable, configuration-driven dataset engine for the G3E vision pipeline.

This repo is **code + config only** — it never stores raw images, labels, or
model weights. It downloads (in bounded, priority-ordered amounts), validates,
converts, deduplicates, splits, and packages vision datasets for G3E's
detection classes: `person`, `fire`, `gun`, `smoke`, `knife`, `car`, `dog`, `cat`.

It's built as an installable Python library (`g3e_data_engine`) with a thin
FastAPI wrapper on top, so it can be:

1. **Imported directly** in another repo/script (`from g3e_data_engine import Pipeline`)
2. **Run from the CLI** (`python scripts/run_pipeline.py`)
3. **Called over HTTP** (`uvicorn g3e_data_engine.api.main:app`) — useful if
   `g3e-app`'s backend or a notebook on another machine wants to trigger/inspect
   a run without a local Python environment.

> This is v1. It is deliberately capped so a first run never accidentally
> downloads a huge dataset onto your laptop. See "Why is everything capped?"
> below and `configs/priority.yaml` / `configs/datasets.yaml`.

---

## Where this fits in the G3E project

```
g3e-data-engine     <- YOU ARE HERE. Reusable dataset processing framework/library.
g3e-vision-dataset  <- The actual processed, versioned dataset + release configs
                        (depends on g3e-data-engine; is what gets published to HF Hub).
g3e-vision          <- Training, evaluation, inference for G3E-1, G3E-2, ...
g3e-app             <- Mobile app, backend, notifications, CCTV integration.
```

`g3e-data-engine` should have **no knowledge** of training or the app — it
only knows how to turn dataset sources into a clean, labeled, versioned
release folder.

---

## Quickstart

```bash
git clone <this-repo>
cd g3e-data-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. See the download plan WITHOUT downloading anything
python scripts/run_pipeline.py --dry-run

# 2. Run the tests (all offline, no network needed)
pytest -q

# 3. Start the API (optional)
uvicorn g3e_data_engine.api.main:app --reload
# -> open http://localhost:8000/docs
```

### As a library

```python
from g3e_data_engine import load_engine_config, Pipeline, PriorityAllocator

cfg = load_engine_config()

# Just want the download plan for a given budget/priorities?
allocator = PriorityAllocator(cfg)
plan = allocator.allocate(total_images=3000, overrides={"fire": 2.0, "gun": 2.0})
print(plan.as_dict())
# {'person': 750, 'fire': 900, 'gun': 900, 'smoke': ...}

# Want to actually run the whole pipeline, and publish the result to Hugging Face?
pipeline = Pipeline(cfg)
result = pipeline.run(
    dry_run=False,
    total_images=3000,
    export=True,
    upload_to_hf="your-org/g3e-vision-dataset",  # opt-in, requires a write-access HF token — see "Credentials" below
)
print(result.upload_url)
```

`upload_to_hf` is entirely optional — omit it and `export=True` just produces a
local, versioned release zip under `datasets/releases/` with nothing sent
anywhere. Uploading is always a separate, explicit opt-in, never an implicit
side effect of running the pipeline.

### As an HTTP service

```bash
uvicorn g3e_data_engine.api.main:app --port 8000
```

| Endpoint                  | What it does                                              |
|----------------------------|------------------------------------------------------------|
| `GET /health`             | Liveness check                                             |
| `GET /classes`            | v1 class list + priority tiers                             |
| `GET /sources`            | Enabled dataset sources + their caps                       |
| `GET /stats`              | Latest `metadata/stats.json` (404 until a real run happens)|
| `POST /pipeline/allocate` | Compute a download plan — no downloading, no disk writes   |
| `POST /pipeline/run`      | Run the pipeline (`dry_run` defaults to `true`); accepts `upload_to_hf` |

`POST /pipeline/run` with `dry_run: false` returns `422 {"error": "source_not_ready", ...}`
if any enabled source isn't ready (see "Source readiness" below) — not a 500.
A missing/invalid credential returns `401 {"error": "missing_credential", ...}`.

---

## Credentials (HF_TOKEN, ROBOFLOW_API_KEY, ...)

Downloaders and the HF uploader read credentials from environment variables
— **never** from any yaml/json config file. The easiest way to set them
locally:

```bash
cp .env.example .env
# edit .env and fill in the tokens you actually need
```

`g3e-data-engine` loads `.env` automatically (via `python-dotenv`) if it's
present in the repo root or your current working directory — or just export
the variables normally, both work identically.

| Env var             | Used by                          | Required? |
|----------------------|-----------------------------------|-----------|
| `HF_TOKEN`           | `kind: huggingface` sources, `scripts/upload_hf.py` / `upload_release_to_hf()` | Optional for public HF datasets (avoids the "unauthenticated requests, please set a HF_TOKEN" rate-limit warning); **required** to upload anything, and required for gated/private HF datasets |
| `ROBOFLOW_API_KEY`   | `kind: roboflow` sources          | **Required** — Roboflow's API has no anonymous read path |
| `KAGGLE_KEY`         | reserved for a future `kind: kaggle` downloader | n/a yet |

If a source needs to read a *different* env var than the default (e.g. you
have two HF accounts), set it per-source in `configs/datasets.yaml`:

```yaml
sources:
  weapons:
    ...
    auth:
      token_env: "HF_WEAPONS_TOKEN"   # instead of the default HF_TOKEN
      required: false                  # true = fail fast if the token is missing
```

`auth.required: true` is how a source declares "I cannot function without a
token" (Roboflow sources set this by default) — the engine raises
`MissingCredentialError` immediately rather than letting the download fail
confusingly partway through.

---

## Source readiness (the "don't download from a half-configured or unlicensed source" gate)

Before the pipeline downloads a single image, `EngineConfig.validate_sources_ready()`
checks every **enabled** source in `configs/datasets.yaml` and refuses the
entire run if any of them:

1. Is `kind: huggingface` with no `hf_repo` set, or `kind: roboflow` with no
   `project`/`version` pinned.
2. Has `license.verified: false`.

This check does **not** run on every config load — `load_engine_config()`,
dry runs, and `/pipeline/allocate` all keep working even while a source is
still mid-setup. It runs once, right before `Pipeline.run(dry_run=False, ...)`
starts downloading, so a run never dies partway through having already
pulled data from source A only to fail on source B:

```
G3E DATA ENGINE

✗ weapons
  Dataset license has not been verified (name='UNKNOWN — derivative of a
  Roboflow dataset, terms not yet reviewed'). Set license.verified: true
  in configs/datasets.yaml after reviewing the actual terms.

✗ fire_smoke
  Dataset license has not been verified (name='CC BY 4.0 (per project
  listing — verify on the pinned version page before enabling)').
  Set license.verified: true in configs/datasets.yaml after reviewing
  the actual terms.

Processing aborted. No data was downloaded.
```

Check readiness any time without running the pipeline:

```bash
python scripts/check_sources.py
```

**v1 ships with `weapons` and `fire_smoke` deliberately unverified** — see
"Dataset sources" below. Flip `license.verified: true` in
`configs/datasets.yaml` only after you've personally checked the actual
license terms; don't assume a Roboflow-derived HF dataset inherited its
source project's license, and don't copy a license string from one dataset
onto another.

---

## Why is everything capped? (the "don't download a huge dataset" ask)

Three independent caps exist, on purpose, so you have to *deliberately* raise
more than one of them before a run gets big:

1. **Per-source cap** (`configs/datasets.yaml -> sources.<name>.max_images`) —
   how many images a single source (e.g. COCO) will ever contribute in one run.
2. **Global cap** (`configs/datasets.yaml -> global_max_images`) — hard ceiling
   across *all* sources combined.
3. **Priority budget** (`configs/priority.yaml -> budget.total_images`) — the
   number actually requested for a given run; must be `<= global_max_images`
   (enforced at config-load time — the engine refuses to start otherwise).

v1 ships with a **6,000 image budget** against an **8,000 image global cap** —
small enough to run comfortably from a laptop-triggered cloud job, big enough
to be a genuinely useful first training set.

## Dataset sources (v1)

The downloader is **source-agnostic** — `kind` is a registry key
(`g3e_data_engine/downloader/`), not "must be Hugging Face." v1 ships two
kinds, proving the abstraction actually works across different APIs:

| Source       | kind          | Provides            | Status |
|--------------|---------------|----------------------|--------|
| `coco`       | `huggingface` | person, car, dog, cat | ✅ verified, ready |
| `weapons`    | `huggingface` | gun, knife, person    | ⏳ needs a license review — see below |
| `fire_smoke` | `roboflow`    | fire, smoke           | ⏳ needs a license review — see below |

- **`coco`** → [`detection-datasets/coco`](https://huggingface.co/datasets/detection-datasets/coco)
  on Hugging Face, CC BY 4.0, verified. The repo is ~20GB total — `max_images: 4000`
  is load-bearing, don't remove it.
- **`weapons`** → [`Subh775/WeaponDetection_Grouped`](https://huggingface.co/datasets/Subh775/WeaponDetection_Grouped),
  ~7,615 rows already consolidated into `GUN`/`KNIFE`/`PERSON` (mapped to
  g3e's `gun`/`knife`/`person` via `class_map` in `configs/datasets.yaml` —
  its `person` images get merged with `coco`'s `person` images downstream).
  It's a derivative of a Roboflow dataset; its own redistribution terms
  haven't been independently confirmed, so it ships `license.verified: false`
  until someone actually checks.
- **`fire_smoke`** → Roboflow Universe project
  [`fire-rqbio/fire-and-smoke-yikzn`](https://universe.roboflow.com/fire-rqbio/fire-and-smoke-yikzn),
  version **3** (pinned — a project update shouldn't silently change an
  already-released g3e-vision-dataset version), ~3,884 images, listed CC BY 4.0.
  Ships `license.verified: false` until that listing is checked on the pinned
  version page itself (Roboflow project licenses can change over time).

We deliberately did **not** wire up a fire/smoke source that turned out to be
a bad fit: `UniDataPro/fire-and-smoke-dataset` on HF is a video dataset under
a commercial license, not an image object-detection set — wrong format and
wrong license for v1. A much larger (~65k image) Roboflow fire/smoke project
exists too, but v1 intentionally doesn't take it wholesale; let the quality
pipeline select a good subset from a smaller, better-scoped source instead of
downloading tens of thousands of images "just in case."

**Adding a new source type** (Kaggle, GitHub releases, a direct URL/zip, your
own collected data) means writing one new downloader module + one registry
import — see `docs/ARCHITECTURE.md` → "Adding a brand-new kind of source."
Nothing else in the pipeline (validators, dedup, split, stats, export) needs
to know or care where an image came from.

## Priority-based downloading

Not all classes are equally easy to find or equally important. `person`,
`fire`, and `gun` are safety-critical *and* rare in generic datasets, so they're
tier 1 and get boosted; `dog`/`cat` are abundant and lower priority for G3E's
use case, so they're tier 4. See `configs/priority.yaml` for the tier weights,
and `g3e_data_engine/core/priority.py` for the allocator itself. You can
override this per run — from Python, the CLI, or the API — without editing
any config file:

```bash
python scripts/run_pipeline.py --no-dry-run --total-images 3000 \
    --override fire=2.0 --override gun=2.0 --export
```

---

## Repository layout

```text
g3e-data-engine/
├── configs/                 # All tunables live here — see below
├── g3e_data_engine/         # The actual importable library
│   ├── api/                 # FastAPI app + routes + schemas
│   ├── core/                # config loading, priority allocator, pipeline orchestrator
│   ├── downloader/           # Source-specific downloaders (HF, etc.) + registry
│   ├── validators/           # Image quality checks (resolution/blur/brightness)
│   ├── dedup/                # Perceptual-hash duplicate detection
│   ├── filters/              # train/val/test split
│   ├── converters/           # COCO -> YOLO (extend for other target formats)
│   ├── exporters/            # Packages a release into datasets/releases/
│   └── utils/                # metadata + statistics helpers
├── scripts/                  # Thin CLI wrappers around the library, one per stage
├── datasets/                  # raw/ processed/ releases/ — NEVER committed (see .gitignore)
├── metadata/                  # classes.json (committed), metadata/stats/versions.json (generated)
├── tests/                     # Fully offline pytest suite
└── docs/                      # Architecture + "what to touch" guides
```

See:
- **`DATASET_SPEC.md`** — the contract every script/class follows (formats, thresholds, versioning).
- **`docs/ARCHITECTURE.md`** — how the pieces fit together, and **which files to touch (or not)** for common changes.
- **`docs/FUTURE_UPDATES.md`** — the backlog for v1.1+ (bigger budgets, new sources, new classes, Roboflow/S3 downloaders, etc.).

---

## Running tests

```bash
pytest -q
```

All 27+ tests run fully offline — no network access, no real dataset
downloads. They cover config validation, the priority allocator, the
validators, dedup, the COCO->YOLO converter, the splitter, and the FastAPI
routes (via `TestClient`, using `dry_run=True` so nothing hits the network).

## License

Code: MIT (see `LICENSE`). Datasets fetched through this engine keep their
own upstream licenses — check `configs/datasets.yaml -> sources.<name>.license`
and `DATASET_SPEC.md -> Licensing` before redistributing anything.
