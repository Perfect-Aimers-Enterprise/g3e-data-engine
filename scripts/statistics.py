#!/usr/bin/env python3
"""Print metadata/stats.json (generate it first with run_pipeline.py --no-dry-run)."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    path = Path("metadata/stats.json")
    if not path.exists():
        print("No metadata/stats.json yet — run scripts/run_pipeline.py --no-dry-run first.")
        return 1
    with open(path, "r", encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
