#!/usr/bin/env python3
"""Split a list of image ids into train/val/test using configs/processing.yaml ratios."""
from __future__ import annotations

import argparse
import json

from g3e_data_engine import load_engine_config
from g3e_data_engine.filters.split import split_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-json", required=True, help="JSON list of image ids")
    args = parser.parse_args()

    cfg = load_engine_config()
    with open(args.ids_json, "r", encoding="utf-8") as f:
        ids = json.load(f)

    result = split_ids(ids, cfg.processing.split)
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
