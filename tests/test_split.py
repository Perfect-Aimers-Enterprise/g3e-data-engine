from g3e_data_engine.core.config import SplitConfig
from g3e_data_engine.filters.split import split_ids


def test_split_covers_every_id_exactly_once():
    ids = [f"img_{i}" for i in range(1000)]
    cfg = SplitConfig(train=0.8, val=0.1, test=0.1, seed=42)
    result = split_ids(ids, cfg)

    all_out = result.train + result.val + result.test
    assert sorted(all_out) == sorted(ids)
    assert len(set(all_out)) == len(ids)


def test_split_is_deterministic_given_seed():
    ids = [f"img_{i}" for i in range(500)]
    cfg = SplitConfig(train=0.8, val=0.1, test=0.1, seed=7)
    r1 = split_ids(ids, cfg)
    r2 = split_ids(ids, cfg)
    assert r1.as_dict() == r2.as_dict()


def test_split_proportions_are_approximately_correct():
    ids = [f"img_{i}" for i in range(1000)]
    cfg = SplitConfig(train=0.8, val=0.1, test=0.1, seed=1)
    result = split_ids(ids, cfg)
    assert 790 <= len(result.train) <= 810
    assert 90 <= len(result.val) <= 110
    assert 90 <= len(result.test) <= 110
