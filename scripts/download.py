#!/usr/bin/env python3
"""Run only the download stage for one or more sources, using the priority budget."""
from __future__ import annotations

import argparse
import json

from g3e_data_engine import load_engine_config, PriorityAllocator
from g3e_data_engine.downloader import get_downloader, DownloadRequest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source name from configs/datasets.yaml")
    parser.add_argument("--dest", default="datasets/raw")
    parser.add_argument("--total-images", type=int, default=None)
    args = parser.parse_args()

    cfg = load_engine_config()
    source = cfg.datasets.sources[args.source]
    allocation = PriorityAllocator(cfg).allocate(total_images=args.total_images)
    per_class = {c: allocation.as_dict().get(c, 0) for c in source.classes}

    downloader = get_downloader(source.kind)
    request = DownloadRequest(
        source_name=args.source, target_classes=per_class, dest_dir=f"{args.dest}/{args.source}"
    )
    images = downloader.download(request)
    print(json.dumps({"downloaded": len(images), "per_class_target": per_class}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
