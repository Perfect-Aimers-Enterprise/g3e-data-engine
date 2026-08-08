from __future__ import annotations

from fastapi import APIRouter

from g3e_data_engine.core.config import load_engine_config
from g3e_data_engine.core.pipeline import Pipeline
from g3e_data_engine.core.priority import PriorityAllocator
from g3e_data_engine.api.schemas import (
    AllocateRequest,
    AllocateResponse,
    ClassBudgetOut,
    RunPipelineRequest,
    RunPipelineResponse,
)

router = APIRouter(tags=["pipeline"])


def _to_allocate_response(result) -> AllocateResponse:
    return AllocateResponse(
        total_requested=result.total_requested,
        total_allocated=result.total_allocated,
        budgets=[
            ClassBudgetOut(
                class_name=b.class_name,
                priority_tier=b.priority_tier,
                target_images=b.target_images,
                reason=b.reason,
            )
            for b in result.budgets
        ],
    )


@router.post("/pipeline/allocate", response_model=AllocateResponse)
def allocate(req: AllocateRequest) -> AllocateResponse:
    """
    Compute (without downloading anything) how many images of each class
    would be fetched given a budget + priority overrides. Use this to sanity
    check a plan before committing to a real run.
    """
    cfg = load_engine_config()
    allocator = PriorityAllocator(cfg)
    result = allocator.allocate(
        total_images=req.total_images,
        overrides=req.overrides or None,
        available_by_class=req.available_by_class or None,
    )
    return _to_allocate_response(result)


@router.post("/pipeline/run", response_model=RunPipelineResponse)
def run_pipeline(req: RunPipelineRequest) -> RunPipelineResponse:
    """
    Run the full pipeline. `dry_run=true` (the default) only computes and
    returns the download plan — it never touches the network or disk beyond
    reading configs. Set `dry_run=false` to actually download/process
    (requires network access to whatever HF repos are configured in
    configs/datasets.yaml).
    """
    cfg = load_engine_config()
    pipeline = Pipeline(cfg)
    result = pipeline.run(
        dry_run=req.dry_run,
        total_images=req.total_images,
        priority_overrides=req.priority_overrides or None,
        available_by_class=req.available_by_class or None,
        export=req.export,
    )

    return RunPipelineResponse(
        dry_run=result.dry_run,
        notes=result.notes,
        allocation=_to_allocate_response(result.allocation),
        accepted_images=(len(result.metadata_records) if not result.dry_run else None),
        duplicates_removed=(result.duplicates_removed if not result.dry_run else None),
        split_counts=({k: len(v) for k, v in result.split.items()} if result.split else None),
        stats=(result.stats or None),
    )
