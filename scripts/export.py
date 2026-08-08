#!/usr/bin/env python3
"""Package datasets/processed + metadata into a versioned release zip."""
from __future__ import annotations

import argparse

from g3e_data_engine.exporters.release_exporter import export_release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="e.g. 1.0.0")
    parser.add_argument("--processed-dir", default="datasets/processed")
    parser.add_argument("--metadata-dir", default="metadata")
    parser.add_argument("--releases-dir", default="datasets/releases")
    args = parser.parse_args()

    path = export_release(args.processed_dir, args.metadata_dir, args.releases_dir, args.version)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
