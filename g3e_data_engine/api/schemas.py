from __future__ import annotations

from pydantic import BaseModel, Field


class AllocateRequest(BaseModel):
    total_images: int | None = Field(
        default=None, description="Override configs/priority.yaml budget.total_images for this call."
    )
    overrides: dict[str, float] = Field(
        default_factory=dict,
        description="Per-class weight multipliers for this call, e.g. {'fire': 2.0}. "
        "Merged on top of configs/priority.yaml overrides.",
    )
    available_by_class: dict[str, int] = Field(
        default_factory=dict,
        description="Optional cap on what's actually available per class, e.g. {'cat': 40}.",
    )


class ClassBudgetOut(BaseModel):
    class_name: str
    priority_tier: int
    target_images: int
    reason: str


class AllocateResponse(BaseModel):
    total_requested: int
    total_allocated: int
    budgets: list[ClassBudgetOut]


class RunPipelineRequest(BaseModel):
    dry_run: bool = True
    total_images: int | None = None
    priority_overrides: dict[str, float] = Field(default_factory=dict)
    available_by_class: dict[str, int] = Field(default_factory=dict)
    export: bool = False
    upload_to_hf: str | None = Field(
        default=None,
        description="Hugging Face dataset repo id, e.g. 'your-org/g3e-vision-dataset'. "
        "Requires export=true and a HF token with write access (see README 'Credentials'). "
        "Uploading never happens unless this is set explicitly.",
    )
    hf_private: bool = Field(
        default=True, description="Whether the uploaded HF dataset repo should be private."
    )


class RunPipelineResponse(BaseModel):
    dry_run: bool
    notes: str
    allocation: AllocateResponse
    accepted_images: int | None = None
    duplicates_removed: int | None = None
    split_counts: dict[str, int] | None = None
    stats: dict | None = None
    upload_url: str | None = None
