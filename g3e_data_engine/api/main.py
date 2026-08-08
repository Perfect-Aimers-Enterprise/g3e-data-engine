"""
FastAPI app for g3e-data-engine.

Run locally with:

    uvicorn g3e_data_engine.api.main:app --reload --port 8000

Then see interactive docs at http://localhost:8000/docs

This app is a thin HTTP wrapper around the same `Pipeline` /
`PriorityAllocator` classes you can import directly in Python — it exists so
other services (e.g. g3e-app's backend) can trigger/inspect the pipeline
over HTTP instead of shelling out to scripts/run_pipeline.py.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from g3e_data_engine.api.routes import health, info, pipeline
from g3e_data_engine.core.exceptions import SourceConfigError, MissingCredentialError

app = FastAPI(
    title="g3e-data-engine",
    description="Configuration-driven dataset engine for the G3E vision pipeline.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(info.router)
app.include_router(pipeline.router)


@app.exception_handler(SourceConfigError)
async def source_config_error_handler(request: Request, exc: SourceConfigError) -> JSONResponse:
    # 422 (not 500): this is a config problem the caller can fix, not a bug —
    # e.g. an enabled source with an unverified license or missing repo/project.
    return JSONResponse(status_code=422, content={"error": "source_not_ready", "detail": str(exc)})


@app.exception_handler(MissingCredentialError)
async def missing_credential_error_handler(request: Request, exc: MissingCredentialError) -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "missing_credential", "detail": str(exc)})
