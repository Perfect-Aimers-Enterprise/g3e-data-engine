#!/usr/bin/env python3
"""
Upload a release zip/folder to Hugging Face Hub.

This is the ONE script in this repo that talks to a remote destination
outside your own infra. It is never called automatically by the pipeline —
you run it deliberately, after reviewing the release with
scripts/statistics.py, so nothing gets published by accident.

Requires: `huggingface-cli login` (or HF_TOKEN env var) beforehand.
"""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. your-org/g3e-vision-dataset")
    parser.add_argument("--folder", required=True, help="Local release folder to upload")
    parser.add_argument("--private", action="store_true", default=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=args.folder)
    print(f"Uploaded {args.folder} -> https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
