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

from fastapi import FastAPI

from g3e_data_engine.api.routes import health, info, pipeline

app = FastAPI(
    title="g3e-data-engine",
    description="Configuration-driven dataset engine for the G3E vision pipeline.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(info.router)
app.include_router(pipeline.router)
