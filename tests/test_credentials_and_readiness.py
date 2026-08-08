import pytest

from g3e_data_engine.core.config import load_engine_config, AuthConfig
from g3e_data_engine.core.credentials import get_token, require_token
from g3e_data_engine.core.exceptions import MissingCredentialError, SourceConfigError


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
def test_get_token_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert get_token("huggingface") is None


def test_get_token_reads_default_env_var(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret123")
    assert get_token("huggingface") == "secret123"


def test_get_token_honors_custom_token_env_on_auth_config(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("MY_CUSTOM_HF_TOKEN", "custom-secret")
    auth = AuthConfig(token_env="MY_CUSTOM_HF_TOKEN")
    assert get_token("huggingface", auth) == "custom-secret"


def test_require_token_raises_when_missing(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError):
        require_token("roboflow")


def test_require_token_returns_value_when_present(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "rf-key")
    assert require_token("roboflow") == "rf-key"


# ---------------------------------------------------------------------------
# Source readiness gate
# ---------------------------------------------------------------------------
def test_default_shipped_config_refuses_unverified_sources():
    """
    Ships v1 with weapons/fire_smoke license.verified=False on purpose (real
    review pending) — validate_sources_ready must refuse both by name.
    """
    cfg = load_engine_config()
    with pytest.raises(SourceConfigError) as exc_info:
        cfg.validate_sources_ready()
    message = str(exc_info.value)
    assert "weapons" in message
    assert "fire_smoke" in message
    assert "Processing aborted" in message


def test_coco_alone_passes_readiness_check():
    """coco ships with license.verified=True and hf_repo set — must pass alone."""
    cfg = load_engine_config()
    cfg.validate_sources_ready(source_names=["coco"])  # should not raise


def test_readiness_check_does_not_block_config_loading():
    """Loading/allocating must keep working even with not-yet-verified sources."""
    cfg = load_engine_config()  # must not raise
    assert cfg.priority.budget.total_images > 0
