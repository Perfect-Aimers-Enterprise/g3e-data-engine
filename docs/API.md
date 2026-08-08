# API Usage

Start the server:

```bash
uvicorn g3e_data_engine.api.main:app --reload --port 8000
```

Interactive docs: `http://localhost:8000/docs`

## Examples

**Get the v1 class list:**
```bash
curl http://localhost:8000/classes
```

**Plan a download (no downloading, no disk writes):**
```bash
curl -X POST http://localhost:8000/pipeline/allocate \
  -H "Content-Type: application/json" \
  -d '{"total_images": 3000, "overrides": {"fire": 2.0, "gun": 2.0}}'
```

**Run the pipeline for real** (requires network access to whatever
`hf_repo`s are configured in `configs/datasets.yaml`):
```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false, "total_images": 3000, "export": true}'
```

**Run the pipeline AND publish the result to Hugging Face:**
```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false, "total_images": 3000, "export": true, "upload_to_hf": "your-org/g3e-vision-dataset"}'
```
Requires `export: true`, network access to the enabled sources, and a
Hugging Face token with write access set as `HF_TOKEN` (see README
"Credentials"). Without `upload_to_hf`, `export: true` alone only writes a
local zip — nothing is ever published unless you ask for it explicitly.

**Check stats after a run:**
```bash
curl http://localhost:8000/stats
```

## Error responses

| Status | `error`               | Meaning |
|--------|------------------------|---------|
| `422`  | `source_not_ready`    | An enabled source is missing its repo/project reference or has `license.verified: false`. Fix `configs/datasets.yaml`, or run `python scripts/check_sources.py` for the same check outside the API. |
| `401`  | `missing_credential`  | A required token (e.g. `ROBOFLOW_API_KEY`, or `HF_TOKEN` for uploading) isn't set. See README "Credentials". |
