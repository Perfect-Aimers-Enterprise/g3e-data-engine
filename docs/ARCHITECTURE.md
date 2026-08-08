# Architecture

```
                     configs/*.yaml
                          │
                          ▼
              core/config.py (EngineConfig)
                          │
                          ▼
              core/priority.py (PriorityAllocator)
                          │
                          ▼
              core/pipeline.py (Pipeline)
     ┌─────────┬──────────┬───────────┬──────────┬───────────┐
     ▼         ▼          ▼           ▼          ▼           ▼
downloader/ validators/ dedup/    filters/   converters/  exporters/
     │
     └─→ registry (base.py) dispatches by `kind` from datasets.yaml
```

`Pipeline.run()` is the single place that sequences:

```
Download -> Validate -> (Convert, when boxes are present) -> Quality Filter
-> Deduplicate -> Generate Metadata -> Split -> Statistics -> Export
```

Both `scripts/run_pipeline.py` (CLI) and `POST /pipeline/run` (API) call the
exact same `Pipeline` object, so behavior never diverges between the two
entry points.

## Files to touch vs. NOT touch

### Adding a new download source (e.g. a new HF dataset for `knife`)
**Touch:**
- `configs/datasets.yaml` — add the source block, set `hf_repo`, `classes`, `max_images`, `license`.

**Don't touch:**
- `g3e_data_engine/downloader/*.py` — `HFDownloader` already handles any
  `kind: huggingface` source generically.
- `core/pipeline.py` — the download stage already fans out to every enabled source.

### Adding a brand-new *kind* of source (e.g. plain HTTP zip, Roboflow, S3)
**Touch:**
- New file `g3e_data_engine/downloader/<kind>_downloader.py` implementing `Downloader`, decorated `@register("<kind>")`.
- One import line added to `g3e_data_engine/downloader/__init__.py`.
- `configs/datasets.yaml` — set `kind: <kind>` on the relevant source(s).

**Don't touch:**
- `g3e_data_engine/downloader/base.py` — the interface is intentionally source-agnostic.
- `core/pipeline.py`.

### Adding a new class (e.g. `helmet`)
**Touch:**
- `configs/classes.yaml` — append with the **next free id** (never reuse/reorder existing ids — see `DATASET_SPEC.md` §3).
- `configs/datasets.yaml` — add the class to whichever source(s) can supply it.
- `configs/priority.yaml` — decide its tier weight (or rely on the class's `priority_tier`).
- `metadata/classes.json` — regenerate from `classes.yaml` (there's no separate script for this yet — see `docs/FUTURE_UPDATES.md`; for now, mirror the `{name, id}` pairs by hand or via a one-off Python snippet like the one used to seed it).

**Don't touch:**
- Any code in `core/`, `validators/`, `dedup/`, `filters/` — these are all class-agnostic.
- `converters/coco_to_yolo.py` — works for any class already in `classes.yaml`.

### Changing quality thresholds (blur/brightness/resolution)
**Touch:** `configs/processing.yaml` only.
**Don't touch:** `validators/image_quality.py` — thresholds are parameters, not constants.

### Changing the priority/budget behavior for a specific run
**Touch:** nothing, if it's one-off — pass `overrides=` / `total_images=` at
the call site (CLI flag, API request body, or Python kwarg).
**Touch `configs/priority.yaml`** only if you want the new behavior to be
the *default* for every future run.
**Don't touch:** `core/priority.py` — unless you're changing the allocation
*algorithm* itself (e.g. moving from proportional-by-tier to something else).

### Changing the pipeline's stage order or adding a wholly new stage
**Touch:** `core/pipeline.py` (and the new stage's own module).
This is the one structural change that *does* require touching the
orchestrator — everything above this line in the doc is designed to avoid it.

## Why downloads happen through a `Downloader` registry instead of `if/elif`

Adding a new source type currently means: one new file, one import line.
No existing file's logic branches grow with every new source, which keeps
`core/pipeline.py` readable as the number of sources increases (COCO today,
Roboflow/S3/local-folder tomorrow).

## Why `priority.py` is separate from `pipeline.py`

The allocator answers "how many of each class, given a budget and
priorities" and is fully pure/deterministic — no I/O, no network. That's
what makes it cheap to expose directly over the API (`POST
/pipeline/allocate`) as a "plan before you commit" step, and why it has its
own, thorough, offline test file (`tests/test_priority.py`).
