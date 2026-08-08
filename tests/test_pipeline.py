import pytest

from g3e_data_engine import Pipeline, load_engine_config
from g3e_data_engine.core.config import SourceDef, LicenseInfo
from g3e_data_engine.core.exceptions import SourceConfigError
from g3e_data_engine.downloader.base import Downloader, DownloadRequest, DownloadedImage, register


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


# ---------------------------------------------------------------------------
# Per-source failure isolation: one source crashing must not take down the
# whole run or lose whatever other sources already produced.
# ---------------------------------------------------------------------------
@register("test_ok_source")
class _AlwaysSucceedsDownloader(Downloader):
    def download(self, request: DownloadRequest) -> list[DownloadedImage]:
        return [
            DownloadedImage(local_path=f"/nonexistent/{request.source_name}_{i}.jpg",
                             source_name=request.source_name, classes_present=["person"])
            for i in range(3)
        ]


@register("test_always_fails_source")
class _AlwaysFailsDownloader(Downloader):
    def download(self, request: DownloadRequest) -> list[DownloadedImage]:
        raise RuntimeError("simulated network failure")


def _build_config_with_fake_sources():
    # load_engine_config() is lru_cache'd — mutating the object it returns
    # would corrupt every other test's view of the "real" shipped config.
    # deep-copy before mutating so this stays fully isolated.
    cfg = load_engine_config().model_copy(deep=True)
    cfg.datasets.sources = {
        "ok_source": SourceDef(
            enabled=True, kind="test_ok_source", classes=["person"], max_images=10,
            license=LicenseInfo(name="test", verified=True),
        ),
        "fail_source": SourceDef(
            enabled=True, kind="test_always_fails_source", classes=["car"], max_images=10,
            license=LicenseInfo(name="test", verified=True),
        ),
    }
    return cfg


def test_one_source_failing_does_not_crash_the_whole_run(tmp_path):
    cfg = _build_config_with_fake_sources()
    pipeline = Pipeline(cfg)

    result = pipeline.run(dry_run=False, total_images=100, work_dir=tmp_path)

    assert "fail_source" in result.failed_sources
    assert "simulated network failure" in result.failed_sources["fail_source"]
    assert "ok_source" not in result.failed_sources


def test_download_stage_directly_isolates_failures(tmp_path):
    cfg = _build_config_with_fake_sources()
    pipeline = Pipeline(cfg)
    allocation = pipeline.allocator.allocate(total_images=100)

    paths, class_by_path, source_by_path, failed = pipeline._download_stage(allocation, tmp_path)

    assert len(paths) == 3  # only from ok_source
    assert all(source_by_path[p] == "ok_source" for p in paths)
    assert failed == {"fail_source": "simulated network failure"}
