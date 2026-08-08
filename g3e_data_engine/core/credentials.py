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
    """Return the token for `kind`, or None if it isn't set anywhere."""
    _load_dotenv_once()
    env_var = _env_var_for(kind, source)
    return os.environ.get(env_var) or None


def require_token(kind: str, source=None) -> str:
    """Same as get_token, but raises MissingCredentialError instead of returning None."""
    token = get_token(kind, source)
    if not token:
        env_var = _env_var_for(kind, source)
        raise MissingCredentialError(
            f"No credential found for kind={kind!r}. Set the {env_var} "
            "environment variable (directly, or via a .env file in the repo "
            "root — see .env.example), or set `auth.token_env` for this "
            "source in configs/datasets.yaml if it should read a different variable."
        )
    return token


def token_required_for(source: "SourceDef") -> bool:
    return bool(source.auth.required)

