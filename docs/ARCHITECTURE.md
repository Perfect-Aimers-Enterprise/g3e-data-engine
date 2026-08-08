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

### Adding a new download source on an EXISTING kind (e.g. another HF dataset)
**Touch:**
- `configs/datasets.yaml` — add the source block: `hf_repo`, `classes`,
  `class_map` (if the source's raw labels don't already match g3e's class
  names — e.g. `"GUN": gun`), `max_images`, `license: {name, verified, url}`,
  and `auth` if it needs a non-default token env var.

**Don't touch:**
- `g3e_data_engine/downloader/*.py` — `HFDownloader`/`RoboflowDownloader`
  already handle any source of their kind generically.
- `core/pipeline.py` — the download stage already fans out to every enabled source.
- `core/config.py` — `validate_sources_ready()` already checks any new
  source the same way it checks existing ones.

Remember: a new source ships with `license.verified: false` until a human
actually reviews its terms — see DATASET_SPEC.md §11. The engine will refuse
to download from it until that's flipped to `true`, which is intentional,
not a bug to work around.

### Adding a brand-new *kind* of source (e.g. Kaggle, GitHub releases, a direct URL/zip)
**Touch:**
- New file `g3e_data_engine/downloader/<kind>_downloader.py` implementing
  `Downloader`, decorated `@register("<kind>")`. Follow
  `roboflow_downloader.py` as the template for a non-HF source (it shows the
  pattern for: lazy-importing the third-party client, pulling a token via
  `credentials.py`, respecting per-class quotas, and applying `class_map`).
- One import line added to `g3e_data_engine/downloader/__init__.py`.
- If the new kind needs its own credential type, add its default env var
  name to `DEFAULT_ENV_VARS` in `g3e_data_engine/core/credentials.py`
  (e.g. `"kaggle": "KAGGLE_KEY"`) — one line.
- `configs/datasets.yaml` — set `kind: <kind>` on the relevant source(s),
  plus whatever fields that kind needs (mirror `project`/`version` on
  `SourceDef` in `core/config.py` if your kind needs its own reference
  fields beyond `hf_repo`).

**Don't touch:**
- `g3e_data_engine/downloader/base.py` — the interface is intentionally source-agnostic.
- `core/pipeline.py`.
- `core/config.py -> validate_sources_ready()` — unless your new kind has a
  genuinely different "is this source ready" condition than "has a
  repo/project reference and a verified license" (rare — most new kinds fit
  that same shape).

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

### Adding a new credential type or changing how a token is looked up
**Touch:** `g3e_data_engine/core/credentials.py` only — it's the single
place every downloader and the HF uploader call into (`get_token` /
`require_token`). Adding a new kind's default env var name is one line in
`DEFAULT_ENV_VARS`.

**Don't touch:** any downloader module — they already call `get_token`/
`require_token` rather than reading `os.environ` directly, so a change here
propagates everywhere automatically.

### Publishing a release to Hugging Face
**Touch:** nothing, if it's one-off — pass `upload_to_hf=<repo_id>` to
`Pipeline.run()` / `--upload-to-hf` on the CLI / `upload_to_hf` in the API
request body, or run `scripts/upload_hf.py` directly against an already-exported
release folder.
**Don't touch:** `g3e_data_engine/exporters/hf_uploader.py` unless you're
changing *how* uploads work (e.g. adding a dry-run/preview mode, or
supporting a second hub). It's intentionally the only file in this repo that
talks to a remote destination outside your own infra.

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
