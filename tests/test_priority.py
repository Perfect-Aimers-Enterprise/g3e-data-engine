from g3e_data_engine.core.config import load_engine_config
from g3e_data_engine.core.priority import PriorityAllocator


def test_higher_tier_gets_more_budget_than_lower_tier():
    cfg = load_engine_config()
    allocator = PriorityAllocator(cfg)
    result = allocator.allocate()
    by_class = result.as_dict()

    # tier 1 (person, fire, gun) must never be allocated less than
    # tier 4 (dog, cat) — that's the entire point of the tier system.
    assert by_class["person"] >= by_class["dog"]
    assert by_class["fire"] >= by_class["cat"]
    assert by_class["gun"] >= by_class["cat"]


def test_total_allocated_respects_max_per_class():
    cfg = load_engine_config()
    allocator = PriorityAllocator(cfg)
    result = allocator.allocate(total_images=100000)  # deliberately huge
    for b in result.budgets:
        assert b.target_images <= cfg.priority.budget.max_per_class


def test_overrides_boost_a_specific_class():
    cfg = load_engine_config()
    allocator = PriorityAllocator(cfg)

    baseline = allocator.allocate().as_dict()
    boosted = allocator.allocate(overrides={"smoke": 10.0}).as_dict()

    assert boosted["smoke"] >= baseline["smoke"]


def test_available_by_class_caps_allocation():
    cfg = load_engine_config()
    allocator = PriorityAllocator(cfg)
    result = allocator.allocate(available_by_class={"cat": 5})
    assert result.as_dict()["cat"] == 5


def test_min_per_class_is_respected_when_room_allows():
    cfg = load_engine_config()
    allocator = PriorityAllocator(cfg)
    result = allocator.allocate()
    for b in result.budgets:
        assert b.target_images >= cfg.priority.budget.min_per_class
