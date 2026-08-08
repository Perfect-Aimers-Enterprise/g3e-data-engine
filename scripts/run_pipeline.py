#!/usr/bin/env python3
"""
CLI entry point for the full pipeline: download -> validate -> convert ->
filter -> dedup -> metadata -> split -> stats -> export.

Examples:

    # See the download plan without downloading anything
    python scripts/run_pipeline.py --dry-run

    # Real run with a smaller budget and fire/gun boosted further
    python scripts/run_pipeline.py --no-dry-run --total-images 3000 \\
        --override fire=2.0 --override gun=2.0 --export
"""
from __future__ import annotations

import argparse
import json
import sys

from g3e_data_engine import Pipeline, load_engine_config
from g3e_data_engine.core.exceptions import SourceConfigError, MissingCredentialError


def _parse_overrides(pairs: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"--override expects class=weight, got {pair!r}")
        name, value = pair.split("=", 1)
        out[name.strip()] = float(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the g3e-data-engine pipeline.")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--total-images", type=int, default=None)
    parser.add_argument(
        "--override", action="append", default=[], help="class=weight, repeatable"
    )
    parser.add_argument("--export", action="store_true", default=False)
    parser.add_argument(
        "--upload-to-hf", default=None,
        help="HF dataset repo id, e.g. your-org/g3e-vision-dataset. Requires --export. "
        "Never runs unless you pass this explicitly — see README 'Credentials'.",
    )
    parser.add_argument("--public", action="store_true", help="Upload as public (default: private)")
    args = parser.parse_args()

    if args.upload_to_hf and not args.export:
        parser.error("--upload-to-hf requires --export")

    cfg = load_engine_config()
    pipeline = Pipeline(cfg)

    try:
        result = pipeline.run(
            dry_run=args.dry_run,
            total_images=args.total_images,
            priority_overrides=_parse_overrides(args.override),
            export=args.export,
            upload_to_hf=args.upload_to_hf,
            hf_private=not args.public,
        )
    except SourceConfigError as exc:
        print(exc)
        return 1
    except MissingCredentialError as exc:
        print(f"Missing credential: {exc}")
        return 1

    print(json.dumps(result.allocation.as_dict(), indent=2))
    if result.dry_run:
        print("\n(dry run — nothing was downloaded; re-run with --no-dry-run to execute)")
    else:
        print(f"\naccepted images: {len(result.metadata_records)}")
        print(f"duplicates removed: {result.duplicates_removed}")
        print(f"split: {result.split and {k: len(v) for k, v in result.split.items()}}")
        if result.upload_url:
            print(f"uploaded to: {result.upload_url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
