from g3e_data_engine.core.config import load_engine_config


def test_loads_all_v1_classes():
    cfg = load_engine_config()
    names = cfg.classes.names()
    assert names == ["person", "fire", "gun", "smoke", "knife", "car", "dog", "cat"]


def test_class_ids_are_stable_and_unique():
    cfg = load_engine_config()
    ids = [c.id for c in cfg.classes.classes]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_priority_budget_within_global_cap():
    cfg = load_engine_config()
    assert cfg.priority.budget.total_images <= cfg.datasets.global_max_images


def test_split_ratios_sum_to_one():
    cfg = load_engine_config()
    s = cfg.processing.split
    assert abs((s.train + s.val + s.test) - 1.0) < 1e-6
