"""
Regression tests for the reported "I set the token but the library says
it's missing" issue — root cause was almost certainly Google Colab's
Secrets panel, which stores values in a per-notebook store that is NOT
exported to os.environ automatically. get_token() now falls back to
checking that store (via a fake google.colab.userdata module here, since
the real package only exists inside Colab itself).
"""
import sys
import types

import pytest

from g3e_data_engine.core.credentials import get_token, require_token
from g3e_data_engine.core.exceptions import MissingCredentialError


def _install_fake_colab(monkeypatch, secrets: dict, access_denied_for: set[str] = frozenset()):
    fake_google = types.ModuleType("google")
    fake_colab = types.ModuleType("google.colab")
    fake_userdata = types.ModuleType("google.colab.userdata")

    class NotebookAccessError(Exception):
        pass

    class SecretNotFoundError(Exception):
        pass

    def _get(key):
        if key in access_denied_for:
            raise NotebookAccessError(key)
        if key not in secrets:
            raise SecretNotFoundError(key)
        return secrets[key]

    fake_userdata.get = _get
    fake_userdata.NotebookAccessError = NotebookAccessError
    fake_userdata.SecretNotFoundError = SecretNotFoundError
    fake_colab.userdata = fake_userdata

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", fake_userdata)


def test_falls_back_to_colab_secret_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _install_fake_colab(monkeypatch, secrets={"HF_TOKEN": "colab-secret-value"})

    assert get_token("huggingface") == "colab-secret-value"


def test_os_environ_takes_precedence_over_colab_secret(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-value")
    _install_fake_colab(monkeypatch, secrets={"HF_TOKEN": "colab-secret-value"})

    assert get_token("huggingface") == "env-value"


def test_require_token_succeeds_via_colab_fallback(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    _install_fake_colab(monkeypatch, secrets={"ROBOFLOW_API_KEY": "rf-colab-key"})

    assert require_token("roboflow") == "rf-colab-key"


def test_colab_notebook_access_denied_treated_as_missing_not_a_crash(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _install_fake_colab(monkeypatch, secrets={"HF_TOKEN": "value"}, access_denied_for={"HF_TOKEN"})

    assert get_token("huggingface") is None
    captured = capsys.readouterr()
    assert "Notebook access" in captured.out


def test_no_colab_module_available_is_a_silent_noop(monkeypatch):
    """Outside Colab (the normal/CI case), google.colab simply isn't importable."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google", raising=False)

    assert get_token("huggingface") is None


def test_require_token_error_message_mentions_colab(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)

    with pytest.raises(MissingCredentialError) as exc_info:
        require_token("huggingface")
    assert "Colab" in str(exc_info.value)
