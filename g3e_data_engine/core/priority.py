"""
Priority allocation — the "what gets downloaded first" logic.

This is the piece the user-facing library call is built around: you give it
a total image budget, the class priority tiers, and (optionally) per-run
overrides, and it tells you how many images of each class to try to fetch.

Design goals:
- Deterministic and easy to reason about (no randomness).
- Safety-critical classes (tier 1: person, fire, gun) are never starved out
  by abundant, low-priority classes (tier 4: dog, cat) even if a source
  happens to have far more of the latter available.
- Respects hard caps: max_per_class, min_per_class, and whatever a source
  can actually supply (available_by_class, if known).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from g3e_data_engine.core.config import EngineConfig


@dataclass
class ClassBudget:
    class_name: str
    priority_tier: int
    target_images: int
    reason: str = ""


@dataclass
class AllocationResult:
    budgets: list[ClassBudget]
    total_requested: int
    total_allocated: int

    def as_dict(self) -> dict[str, int]:
        return {b.class_name: b.target_images for b in self.budgets}


class PriorityAllocator:
    """
    Turns (class -> priority tier) + a total image budget into a per-class
    download target.

    Usage (library):

        from g3e_data_engine import load_engine_config, PriorityAllocator

        cfg = load_engine_config()
        allocator = PriorityAllocator(cfg)

        # default allocation, using configs/priority.yaml as-is
        result = allocator.allocate()

        # override the budget and/or bump specific classes for this run only
        result = allocator.allocate(
            total_images=3000,
            overrides={"fire": 2.0, "gun": 2.0},   # extra weight multiplier
            available_by_class={"cat": 40},         # cap to what's actually available
        )

        print(result.as_dict())
        # {'person': 900, 'fire': 900, 'gun': 900, 'smoke': 675, 'knife': 675,
        #  'car': 450, 'dog': 225, 'cat': 40}
    """

    def __init__(self, config: EngineConfig):
        self.config = config

    def allocate(
        self,
        total_images: int | None = None,
        overrides: dict[str, float] | None = None,
        available_by_class: dict[str, int] | None = None,
    ) -> AllocationResult:
        pconf = self.config.priority
        budget_total = total_images if total_images is not None else pconf.budget.total_images
        max_per_class = pconf.budget.max_per_class
        min_per_class = pconf.budget.min_per_class
        tier_weights = pconf.tier_weights

        merged_overrides = dict(pconf.overrides)
        if overrides:
            merged_overrides.update(overrides)

        classes = self.config.classes.classes

        # Step 1: raw weight per class = tier_weight * override_multiplier
        weights: dict[str, float] = {}
        for c in classes:
            base_w = tier_weights.get(c.priority_tier, 1.0)
            mult = merged_overrides.get(c.name, 1.0)
            weights[c.name] = base_w * mult

        total_weight = sum(weights.values()) or 1.0

        # Step 2: proportional share of the budget
        raw_targets: dict[str, int] = {
            name: int(round(budget_total * (w / total_weight)))
            for name, w in weights.items()
        }

        # Step 3: enforce min/max per class
        for name in raw_targets:
            raw_targets[name] = max(min_per_class, raw_targets[name])
            raw_targets[name] = min(max_per_class, raw_targets[name])

        # Step 4: cap by what's actually available, if the caller told us
        if available_by_class:
            for name, avail in available_by_class.items():
                if name in raw_targets:
                    raw_targets[name] = min(raw_targets[name], avail)

        budgets = [
            ClassBudget(
                class_name=c.name,
                priority_tier=c.priority_tier,
                target_images=raw_targets[c.name],
                reason=(
                    f"tier={c.priority_tier}, weight={weights[c.name]:.2f}, "
                    f"share_of_total_weight={weights[c.name] / total_weight:.2%}"
                ),
            )
            for c in classes
        ]

        return AllocationResult(
            budgets=budgets,
            total_requested=budget_total,
            total_allocated=sum(b.target_images for b in budgets),
        )
