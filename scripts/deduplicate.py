#!/usr/bin/env python3
"""Find near-duplicate images (pHash) in a directory."""
from __future__ import annotations

import argparse
import glob
import json

from g3e_data_engine import load_engine_config
from g3e_data_engine.dedup.phash_dedup import find_duplicates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    cfg = load_engine_config()
    paths = glob.glob(f"{args.dir}/**/*.jpg", recursive=True) + glob.glob(
        f"{args.dir}/**/*.png", recursive=True
    )
    result = find_duplicates(paths, cfg.processing.duplicates)
    print(json.dumps({"kept": len(result.kept), "duplicates": len(result.duplicates)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
