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

# If you'll use any roboflow-kind source (e.g. fire_smoke in v1's shipped
# config), install its extra too — check_sources.py below tells you if you
# forgot:
pip install -e ".[dev,roboflow]"

# 1. Preflight: confirm every enabled source can actually run (dependency,
#    repo/project reference, license) — takes milliseconds, no network.
python scripts/check_sources.py

# 2. See the download plan WITHOUT downloading anything
python scripts/run_pipeline.py --dry-run

# 3. Run the tests (all offline, no network needed)
pytest -q

# 4. Start the API (optional)
uvicorn g3e_data_engine.api.main:app --reload
# -> open http://localhost:8000/docs
```

**Running in Colab or on a fresh cloud box?** Use the same `pip install -e
".[roboflow]"` extras form there too, then run `python scripts/check_sources.py`
*before* `run_pipeline.py` — that's what catches a missing downloader
dependency in milliseconds instead of after other sources have already
spent minutes/hours downloading.

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

### Running in Google Colab

**Colab's Secrets panel (the key icon in the left sidebar) does NOT
automatically export to `os.environ`.** It's a separate, per-notebook
secure store accessed via `google.colab.userdata.get(...)`, and each secret
has its own "Notebook access" toggle. This is the single most common cause
of "I definitely set the token and the library still says it's missing."

`get_token()` handles this automatically: if the environment variable isn't
set, it falls back to checking Colab's secret store under the *same
variable name*. So all you need is:

1. Open the key icon in the left sidebar.
2. Add a secret named exactly `HF_TOKEN` (and/or `ROBOFLOW_API_KEY`) with
   your token as the value.
3. Toggle **"Notebook access"** on for that secret — this step is easy to
   miss, and skipping it makes `userdata.get()` fail even though the secret
   exists. If this happens, the engine prints a specific hint pointing you
   back to this toggle rather than a generic "missing credential" error.

No extra code needed — you do **not** need to manually do
`os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")` yourself, though that
also works fine if you prefer it explicit.

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

## Preflight checks (catching a missing dependency in milliseconds, not hours)

Real failure this fixes: a run resolved its Hugging Face sources fine, then
crashed on `from roboflow import Roboflow` an hour in, because the
`roboflow` package simply wasn't installed in that environment. Before
`Pipeline.run(dry_run=False, ...)` downloads a single byte, it runs a
preflight (`core/preflight.py`) that checks, per enabled source:

1. **dependency** — is the package its `kind` needs actually importable?
2. **repository** — does it have a repo/project reference set?
3. **license** — has the license been verified?

All three are pure/offline checks (no network calls), so this takes
milliseconds. Any failure aborts the whole run before anything downloads:

```
G3E DATA ENGINE — PREFLIGHT

Sources
────────────────────────────
✓ coco
  huggingface dependency     ✓
  repository                 ✓
  license                    ✓

✗ weapons
  huggingface dependency     ✓
  repository                 ✓
  license                    ✗

✗ fire_smoke
  roboflow dependency        ✗

ERROR
✗ weapons
  Dataset license has not been verified (name='UNKNOWN — derivative of a
  Roboflow dataset, terms not yet reviewed'). Set license.verified: true
  in configs/datasets.yaml after reviewing the actual terms.

✗ fire_smoke
  Roboflow source is enabled but the `roboflow` package is missing.

  Install:
  pip install "g3e-data-engine[roboflow]"

Processing aborted. No data was downloaded.
```

Run this check on its own, any time:

```bash
python scripts/check_sources.py
```

On a fresh cloud box or Colab runtime, run this **before** launching the
full pipeline — if a downloader kind needs a package that isn't installed
yet (e.g. `pip install -e ".[roboflow]"` wasn't run), you find out
immediately instead of after other sources have already spent minutes
downloading.

### If you just edited `configs/datasets.yaml` and preflight still shows the old values

Config files are cached (so repeated calls don't re-parse YAML on every
request), but the cache automatically invalidates itself whenever any of
the four config files' modification time or size changes — so editing
`configs/datasets.yaml` and calling `load_engine_config()` again picks up
the edit immediately, even within the same running notebook kernel or API
server process. No restart, no manual cache-clearing needed.

If you're ever unsure, `from g3e_data_engine import clear_config_cache;
clear_config_cache()` forces a guaranteed-fresh read on the next call.

## Download progress, streaming saves, and resuming an interrupted run

Every downloader:

- Shows a live progress bar (via `tqdm`) while it downloads.
- Writes each accepted image to disk **immediately** — never buffers a
  source's images in memory until the whole thing finishes — and records it
  in a per-source `<dest>/_progress.json` manifest right after. A crash
  loses at most the one image that was in flight.
- **Resumes automatically** on the next run: if `_progress.json` already has
  progress in it, the downloader picks up from there — skipping classes
  already satisfied, and (for streaming HF sources) skipping past rows it
  already scanned via the `datasets` library's own `.skip()` — instead of
  starting that source over from scratch.
- If a source's download is interrupted (network drop, HF Hub hiccup,
  etc.), that source returns whatever it already has rather than crashing —
  **and it doesn't take the rest of the run down with it.** `Pipeline`
  catches failures per source, logs them, and continues with the remaining
  sources. Check `result.failed_sources` (a `{source_name: error}` dict) —
  or in the CLI/console output — to see what failed and re-run later to
  pick those back up:

```
=== STAGE: Download (3 enabled source(s)) ===
-> coco: starting (kind=huggingface, target={'person': 1091, 'car': 545, ...})
coco: 1091img [00:42, 25.9img/s]
<- coco: done — 1091 image(s) downloaded
-> weapons: starting (kind=huggingface, target={'gun': 1091, 'knife': 818, 'person': 1091})
x  weapons: FAILED — <network error>
   Anything already saved for 'weapons' before the failure is kept on disk
   (see datasets/raw/weapons/_progress.json) — re-run to resume it.

[warning] 1 source(s) failed and were skipped (their partial progress is
preserved on disk — re-run to resume them): ['weapons']
```

If a resume genuinely isn't worth chasing (e.g. you'd rather start clean),
just delete that source's folder under `datasets/raw/<source_name>/` and
re-run — there's no separate "reset" command needed.

Every pipeline stage prints a banner as it starts (`Download`, `Validate`,
`Deduplicate`, `Metadata & Split`, `Statistics`, `Export`, `Upload to
Hugging Face`) so a long-running console/Colab session always shows exactly
where the run is, not just a silent wait.

### Why images get rejected (and why the resolution check uses the shorter side)

The Validate stage prints a live breakdown of *why* images were rejected —
plus the **actual measured distribution** of shorter-side and blur score
across the batch, not just an accept/reject count — the moment it finishes:

```
=== STAGE: Validate (714 downloaded image(s)) ===
    accepted: 612 / 714
    measured shorter-side: p10=380 median=480 p90=640 (threshold: min_shorter_side=416)
    measured blur score:   p10=110.2 median=340.5 p90=920.1 (threshold: blur_threshold=90.0)
    rejected breakdown: {'low_resolution': 58, 'blurry': 31, 'dark_or_bright': 13}
```

This is deliberate: tuning `configs/processing.yaml` from a guess (even a
reasonable-sounding one) is how this engine ended up with an overly strict
default twice. Comparing the measured percentiles against the threshold
tells you immediately whether a check is well-calibrated for a given
source or needs adjusting — look at *this source's actual numbers*, not a
number that sounded right in the abstract. The engine also warns
explicitly if more than half of a batch was rejected, so a miscalibrated
threshold is obvious immediately instead of only showing up as a
suspiciously small final dataset.

Resolution is checked against `min(width, height)` — the shorter side —
not width and height independently: a `min_width=640, min_height=640`-style
check demands both dimensions individually clear 640, which only
near-square images satisfy. An ordinary 640×480 or 480×640 photo (the
single most common shape in datasets like COCO) fails a check like that
even though it's a perfectly good image. `configs/processing.yaml`'s
`image.min_shorter_side` (default `416`) avoids this — see
DATASET_SPEC.md section 4 for the full reasoning.

### If a fix doesn't seem to have taken effect (Colab/Jupyter stale-code gotcha)

Every run prints `g3e-data-engine v{version}` as its very first line. If
you've pulled updated code but the *behavior* still looks like the old
version — e.g. no rejection breakdown printed at all despite images being
rejected — check that version line first. **Replacing files on disk in a
notebook is not enough**: Python/Jupyter/Colab caches already-imported
modules in memory, so `import g3e_data_engine` after an update can silently
keep running the OLD in-memory code even though the files changed. Restart
the runtime (**Runtime → Restart session** in Colab, or restart the
kernel) after updating the repo, then re-run — don't just re-execute the
same cells.

### Category/label decoding (why a source might download 0 images and look "stuck")

Most HF object-detection datasets store `objects.category` (and sometimes a
bare `label`) as an **integer** `ClassLabel` id — e.g. `category: 4`
meaning `"car"`, not `category: "car"`. If that integer is compared
directly against G3E's string class names, nothing ever matches, on any
row — the downloader just streams the entire dataset finding zero images,
which looks exactly like a hang rather than an error.

The HF downloader resolves these ids to names automatically, once per
source, from the dataset's `Features` metadata (no extra network call —
this metadata is already fetched as part of `load_dataset()`). If it can't
resolve a names table for a source, it prints a warning up front rather
than silently scanning:

```
[coco] could not resolve category/label names from this dataset's
features — if its categories are stored as integers rather than
strings, no rows may match. If downloads stay at 0 accepted for a
while, this is almost certainly why; see DATASET_SPEC.md
'Category/label decoding'.
```

The download progress bar also counts **rows scanned** (not just accepted
images), with the accepted count as a live postfix — so a 0-accepted
scan is visible immediately as `coco: 4213row [00:12, accepted=0]`,
not just silence. And as a hard backstop, each source stops scanning after
`max_rows_scanned` (default 50,000; override per-source in
`configs/datasets.yaml`) rather than running indefinitely if something's
still wrong — the exact failure mode that used to cost 30+ minutes of
streaming for zero images.

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

v1 ships with a **6,000 image default request budget** against a **50,000
image global ceiling** — the default stays modest for a first-ever run, but
the ceiling is high enough to request a genuinely large training set
(`total_images=50000`) once you're past initial testing, accounting for the
fact that quality filtering + dedup will reject a meaningful fraction of
whatever gets downloaded (see "Why images get rejected" above) — requesting
more than you need is expected, not a mistake. Per-source caps are set
relative to each source's actual known size (e.g. `coco: 20000` against its
~117k rows, `weapons: 7000` against its ~7,615 total rows, `fire_smoke:
3500` against its ~3,884 images) — raise `global_max_images` further if you
add bigger sources later.

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

**A note on `min_per_class`/`max_per_class` clamping.** `configs/priority.yaml`'s
`min_per_class`/`max_per_class` are enforced *after* the proportional split —
so a small `total_images` against a large `min_per_class × number of classes`
floor can allocate noticeably *more* than you requested (every class gets
clamped up to the floor), and the mirror case with `max_per_class` can
allocate noticeably *less*. This is intentional, but it's exactly the kind
of thing that's confusing to discover only after staring at the numbers — so
`AllocationResult.notes` (and the console output of `Pipeline.run()` /
`scripts/run_pipeline.py`) says so explicitly whenever the deviation is
large:

```python
>>> allocator.allocate(total_images=1200).notes
"Requested 1200 total images but allocated 1432 (more than requested).
configs/priority.yaml's min_per_class=150 / max_per_class=2500 clamped the
proportional split — ..."
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
