# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

All tests run fully offline. If a test you're adding needs real network
access (e.g. exercising `HFDownloader` against a live HF repo), mark it
clearly and keep it out of the default `pytest -q` run — that suite should
stay something a contributor can run with zero setup and zero API keys.

## Before opening a PR

1. `pytest -q` passes.
2. If you touched `configs/classes.yaml`, `configs/datasets.yaml`, or
   `configs/priority.yaml`, re-read `DATASET_SPEC.md` — you may need to
   update it too (they're meant to stay in sync).
3. If you added a new source kind or a new pipeline stage, add the
   corresponding entry to `docs/ARCHITECTURE.md` → "Files to touch vs NOT touch".
4. Never commit anything under `datasets/raw/`, `datasets/processed/`, or
   `datasets/releases/` — `.gitignore` already blocks this; don't work around it.

## Code style

- Type hints on every public function/method.
- Dataclasses for internal data-carrying objects; Pydantic models for
  anything that's loaded from YAML/JSON or crosses the API boundary.
- Prefer adding a new small module over adding branches to an existing one
  (see the downloader registry pattern in `g3e_data_engine/downloader/`).
