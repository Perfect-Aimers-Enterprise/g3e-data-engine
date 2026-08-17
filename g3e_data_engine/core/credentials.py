"""
Credential lookup for every downloader/uploader `kind`.

Design:
- Tokens are NEVER read from or written to any yaml/json config file — only
  *which environment variable to read* is configurable
  (`SourceDef.auth.token_env` in configs/datasets.yaml). The actual secret
  always comes from the process environment, optionally loaded from a local
  `.env` file (via python-dotenv, if installed) that stays out of git via
  .gitignore.
- Each `kind` has a sensible default env var name (HF_TOKEN for
  huggingface, ROBOFLOW_API_KEY for roboflow) so most users never need to
  set `token_env` explicitly — it's there for the cases where you're juggling
  multiple accounts/tokens and want a source to read from a non-default
  variable.
- A token is optional unless the source config says `auth.required: true`,
  or the caller explicitly calls `require_token(...)` (as the HF uploader
  does — you cannot push to Hugging Face anonymously).
- In Google Colab specifically: the Secrets panel (key icon in the left
  sidebar) does NOT automatically export to `os.environ` — it's a separate
  store accessed via `google.colab.userdata.get(...)`, and per-notebook
  access has to be toggled on for each secret. This is a very common cause
  of "I definitely set the token but the library says it's missing." When
  `os.environ` doesn't have the variable, `get_token` automatically falls
  back to checking Colab's secret store under the same variable name (see
  `_try_colab_secret` below) — no extra code needed on your end beyond
  naming the Colab secret the same as the env var (e.g. `HF_TOKEN`) and
  making sure its notebook-access toggle is on.

Usage:

    from g3e_data_engine.core.credentials import get_token, require_token

    token = get_token("huggingface")               # None if not set
    token = get_token("huggingface", source_def)    # honors source.auth.token_env
    token = require_token("huggingface")            # raises MissingCredentialError if unset
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from g3e_data_engine.core.exceptions import MissingCredentialError

if TYPE_CHECKING:
    from g3e_data_engine.core.config import SourceDef, AuthConfig

CredentialSource = "SourceDef | AuthConfig | None"

# Default environment variable name per downloader/uploader kind. Extend this
# when you add a new downloader kind (see docs/ARCHITECTURE.md).
DEFAULT_ENV_VARS: dict[str, str] = {
    "huggingface": "HF_TOKEN",
    "roboflow": "ROBOFLOW_API_KEY",
    "kaggle": "KAGGLE_KEY",
}

_dotenv_loaded = False


def _load_dotenv_once() -> None:
    """
    Best-effort, one-time load of a `.env` file from the repo root or the
    current working directory. Silently does nothing if python-dotenv isn't
    installed or no `.env` file exists — this must never be required for the
    engine to run, only a convenience when it's present.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    from g3e_data_engine.core.config import REPO_ROOT

    for candidate in (REPO_ROOT / ".env", Path.cwd() / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _try_colab_secret(env_var: str) -> str | None:
    """
    Fallback for Google Colab's Secrets panel, which stores values in a
    per-notebook secure store that is NOT exported to `os.environ`
    automatically. If `google.colab` isn't importable (i.e. this isn't
    Colab), this is a silent no-op — never a hard dependency.

    If the secret exists but this notebook hasn't been granted access to it
    yet (the "Notebook access" toggle in the Secrets panel), Colab raises
    its own access-error rather than just returning None — that case is
    caught separately so the person gets a specific, actionable hint
    instead of a generic "credential missing" message.
    """
    try:
        from google.colab import userdata
    except ImportError:
        return None

    try:
        value = userdata.get(env_var)
        return value or None
    except Exception as exc:
        if type(exc).__name__ == "NotebookAccessError":
            print(
                f"  [credentials] Found a Colab secret named '{env_var}' but this notebook "
                "doesn't have permission to use it yet — click the key icon in the left "
                f"sidebar and toggle 'Notebook access' on for '{env_var}'."
            )
        # SecretNotFoundError and anything else: treat the same as "not set
        # anywhere" and let the normal missing-credential path handle it.
        return None


def _env_var_for(kind: str, source) -> str:
    """
    `source` may be a SourceDef (has `.auth.token_env`), an AuthConfig
    itself (has `.token_env` directly), or None. Accepting either avoids
    forcing callers that only have an AuthConfig (e.g. the HF uploader,
    which isn't tied to any one dataset source) to fabricate a fake SourceDef.
    """
    if source is not None:
        token_env = getattr(getattr(source, "auth", source), "token_env", None)
        if token_env:
            return token_env
    return DEFAULT_ENV_VARS.get(kind, f"{kind.upper()}_TOKEN")


def get_token(kind: str, source=None) -> str | None:
    """
    Return the token for `kind`, or None if it isn't set anywhere.
    Checks, in order: the environment variable (including anything loaded
    from a local `.env` file), then — only if that's empty — Colab's
    Secrets panel under the same variable name.
    """
    _load_dotenv_once()
    env_var = _env_var_for(kind, source)
    token = os.environ.get(env_var)
    if not token:
        token = _try_colab_secret(env_var)
    return token or None


def require_token(kind: str, source=None) -> str:
    """Same as get_token, but raises MissingCredentialError instead of returning None."""
    token = get_token(kind, source)
    if not token:
        env_var = _env_var_for(kind, source)
        raise MissingCredentialError(
            f"No credential found for kind={kind!r}. Set the {env_var} "
            "environment variable (directly, or via a .env file in the repo "
            "root — see .env.example); in Google Colab, add a secret named "
            f"'{env_var}' in the Secrets panel (key icon, left sidebar) and "
            "make sure its 'Notebook access' toggle is on; or set "
            "`auth.token_env` for this source in configs/datasets.yaml if "
            "it should read a different variable."
        )
    return token


def token_required_for(source: "SourceDef") -> bool:
    return bool(source.auth.required)

