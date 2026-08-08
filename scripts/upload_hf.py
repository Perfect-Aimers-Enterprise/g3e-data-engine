#!/usr/bin/env python3
"""
Upload a release folder to Hugging Face Hub.

This is the ONE script in this repo that talks to a remote destination
outside your own infra. It is never called automatically by the pipeline
unless you explicitly pass --upload-to-hf to run_pipeline.py (or
upload_to_hf=... to Pipeline.run() in Python) — running this script by hand
is the deliberate, review-it-first path.

Requires a Hugging Face token with WRITE access — set HF_TOKEN (env var or
.env file), or pass --token-env to read a different variable. See
README.md "Credentials".
"""
from __future__ import annotations

import argparse

from g3e_data_engine.core.config import AuthConfig
from g3e_data_engine.exporters.hf_uploader import upload_release_to_hf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. your-org/g3e-vision-dataset")
    parser.add_argument("--folder", required=True, help="Local release folder to upload")
    parser.add_argument("--public", action="store_true", help="Upload as a public repo (default: private)")
    parser.add_argument("--token-env", default=None, help="Read the HF token from this env var instead of HF_TOKEN")
    args = parser.parse_args()

    auth = AuthConfig(token_env=args.token_env) if args.token_env else None
    url = upload_release_to_hf(folder=args.folder, repo_id=args.repo_id, private=not args.public, auth=auth)
    print(f"Uploaded {args.folder} -> {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
