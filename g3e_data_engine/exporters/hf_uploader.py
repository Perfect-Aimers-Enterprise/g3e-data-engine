"""
Push a released G3E dataset to the Hugging Face Hub.

This is a plain library function — importable and callable on its own — as
well as something `Pipeline.run(..., upload_to_hf="org/repo")` can call
automatically right after `export_release()`. Either way, it is NEVER
triggered implicitly: a pipeline run only uploads when the caller explicitly
asks for it, since publishing a dataset is a one-way, public-by-default-risk
action that deserves an explicit opt-in.

Requires a Hugging Face token with WRITE access to the target repo — unlike
downloading (where a token is optional for public datasets), uploading has
no anonymous path, so this always calls `require_token`, not `get_token`.

Usage:

    from g3e_data_engine.exporters.hf_uploader import upload_release_to_hf

    url = upload_release_to_hf(
        folder="datasets/releases/g3e-vision-dataset-v1.1.0",
        repo_id="your-org/g3e-vision-dataset",
        private=True,
    )
    print(url)

Or let the token come from a source-style `auth.token_env` override instead
of the default `HF_TOKEN`:

    from g3e_data_engine.core.config import AuthConfig
    url = upload_release_to_hf(folder=..., repo_id=..., auth=AuthConfig(token_env="HF_WRITE_TOKEN"))
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from g3e_data_engine.core.credentials import require_token

if TYPE_CHECKING:
    from g3e_data_engine.core.config import AuthConfig


def upload_release_to_hf(
    folder: str | Path,
    repo_id: str,
    private: bool = True,
    auth: "AuthConfig | None" = None,
    commit_message: str = "g3e-data-engine: dataset release upload",
) -> str:
    """
    Uploads `folder` (typically a datasets/releases/<name> directory
    produced by export_release()) to a Hugging Face dataset repo.

    Returns the repo's URL on success. Raises MissingCredentialError if no
    token is configured, and re-raises anything huggingface_hub raises on
    network/auth failure — this function doesn't swallow upload errors,
    since a silently-failed upload is worse than a loud one.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "The 'huggingface_hub' package is required to upload to HF Hub. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Release folder not found: {folder}")

    # `auth` (an AuthConfig) lets a caller point at a non-default env var
    # the same way a dataset source's `auth.token_env` does — credentials.py
    # accepts either a SourceDef or a bare AuthConfig here.
    token = require_token("huggingface", auth)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        token=token,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/datasets/{repo_id}"
