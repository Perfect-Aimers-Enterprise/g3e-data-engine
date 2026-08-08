from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from g3e_data_engine.core.config import load_engine_config

router = APIRouter(tags=["info"])


@router.get("/classes")
def get_classes() -> dict:
    cfg = load_engine_config()
    return {
        "version": cfg.classes.version,
        "classes": [c.model_dump() for c in cfg.classes.classes],
    }


@router.get("/sources")
def get_sources() -> dict:
    cfg = load_engine_config()
    return {
        "global_max_images": cfg.datasets.global_max_images,
        "sources": {k: v.model_dump() for k, v in cfg.datasets.sources.items()},
    }


@router.get("/stats")
def get_stats() -> dict:
    stats_path = Path("metadata/stats.json")
    if not stats_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No stats.json yet — run the pipeline at least once (dry_run=false).",
        )
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)
