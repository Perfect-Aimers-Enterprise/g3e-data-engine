from g3e_data_engine.core.config import load_engine_config
from g3e_data_engine.core.preflight import run_preflight, KIND_DEPENDENCIES


def test_preflight_catches_missing_roboflow_dependency():
    """
    The 'roboflow' package is NOT installed in this test environment on
    purpose — this is the real-world scenario the preflight check exists
    for (a downloader's dependency missing from the environment), not a
    mocked one.
    """
    cfg = load_engine_config()
    report = run_preflight(cfg, source_names=["fire_smoke"])
    assert report.all_ok is False

    check = next(c for c in report.checks if c.name == "fire_smoke")
    assert check.dependency_ok is False
    assert "roboflow" in check.errors[0].lower()
    assert 'pip install "g3e-data-engine[roboflow]"' in check.errors[0]


def test_preflight_skips_repo_and_license_checks_when_dependency_missing():
    """When the dependency itself is missing, don't also report repository/
    license problems — one clear blocking error, not three confusing ones."""
    cfg = load_engine_config()
    report = run_preflight(cfg, source_names=["fire_smoke"])
    check = next(c for c in report.checks if c.name == "fire_smoke")
    assert len(check.errors) == 1


def test_preflight_passes_coco_alone():
    """coco's dependency (datasets) IS installed, repo is set, license is verified."""
    cfg = load_engine_config()
    report = run_preflight(cfg, source_names=["coco"])
    assert report.all_ok is True
    check = report.checks[0]
    assert check.dependency_ok and check.repository_ok and check.license_ok


def test_preflight_report_render_includes_install_hint():
    cfg = load_engine_config()
    report = run_preflight(cfg, source_names=["fire_smoke"])
    rendered = report.render()
    assert "PREFLIGHT" in rendered
    assert "Processing aborted" in rendered
    assert "pip install" in rendered


def test_preflight_render_on_full_success_says_ready():
    cfg = load_engine_config()
    report = run_preflight(cfg, source_names=["coco"])
    rendered = report.render()
    assert "All enabled sources are ready." in rendered
    assert "ERROR" not in rendered


def test_kind_dependencies_registry_has_both_shipped_kinds():
    assert "huggingface" in KIND_DEPENDENCIES
    assert "roboflow" in KIND_DEPENDENCIES
