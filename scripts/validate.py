#!/usr/bin/env python3
"""Validate a directory of images against configs/processing.yaml thresholds."""
from __future__ import annotations

import argparse
import glob
import json

from g3e_data_engine import load_engine_config
from g3e_data_engine.validators.image_quality import validate_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory of images to validate (recursive)")
    args = parser.parse_args()

    cfg = load_engine_config()
    paths = glob.glob(f"{args.dir}/**/*.jpg", recursive=True) + glob.glob(
        f"{args.dir}/**/*.png", recursive=True
    )
    results = validate_batch(paths, cfg.processing.image)

    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]
    print(json.dumps({"total": len(results), "accepted": len(accepted), "rejected": len(rejected)}, indent=2))
    for r in rejected[:20]:
        print(f"REJECTED {r.path}: {r.reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
