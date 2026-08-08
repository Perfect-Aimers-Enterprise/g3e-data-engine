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

# Want to actually run the whole pipeline?
pipeline = Pipeline(cfg)
result = pipeline.run(dry_run=False, total_images=3000, export=True)
```

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
| `POST /pipeline/run`      | Run the pipeline (`dry_run` defaults to `true`)             |

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
