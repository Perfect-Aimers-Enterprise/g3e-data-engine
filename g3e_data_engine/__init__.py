"""
g3e-data-engine
================
Reusable, configuration-driven dataset engine for the G3E vision pipeline.

Public surface (what other repos, e.g. g3e-vision-dataset, should import):

    from g3e_data_engine import (
        EngineConfig, load_engine_config,
        PriorityAllocator,
        Pipeline,
    )

Everything else under g3e_data_engine.* is an implementation detail and may
change between minor versions without notice.
"""
from g3e_data_engine.core.config import EngineConfig, load_engine_config
from g3e_data_engine.core.priority import PriorityAllocator, ClassBudget
from g3e_data_engine.core.pipeline import Pipeline, PipelineRunResult

__all__ = [
    "EngineConfig",
    "load_engine_config",
    "PriorityAllocator",
    "ClassBudget",
    "Pipeline",
    "PipelineRunResult",
]

__version__ = "0.1.0"
