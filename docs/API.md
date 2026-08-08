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

**Check stats after a run:**
```bash
curl http://localhost:8000/stats
```
