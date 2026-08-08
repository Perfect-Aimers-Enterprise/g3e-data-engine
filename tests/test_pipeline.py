import pytest

from g3e_data_engine import Pipeline, load_engine_config
from g3e_data_engine.core.exceptions import SourceConfigError


def test_dry_run_never_raises_even_with_unverified_sources():
    cfg = load_engine_config()
    pipeline = Pipeline(cfg)
    result = pipeline.run(dry_run=True, total_images=1000)
    assert result.dry_run is True
    assert result.allocation.total_requested == 1000


def test_real_run_refuses_before_any_download_when_sources_not_ready(tmp_path, monkeypatch):
    cfg = load_engine_config()
    pipeline = Pipeline(cfg)
    # weapons/fire_smoke ship with license.verified=False by default — a
    # real run must refuse cleanly, before touching the network/disk.
    with pytest.raises(SourceConfigError):
        pipeline.run(dry_run=False, total_images=500, work_dir=tmp_path)

    # Nothing should have been created under work_dir since it aborted
    # before the download stage.
    assert not (tmp_path / "raw").exists() or not any((tmp_path / "raw").iterdir())
