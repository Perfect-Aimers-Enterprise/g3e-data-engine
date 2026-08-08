#!/usr/bin/env python3
"""Convert COCO-style boxes for one image into YOLO label lines (demo/utility script)."""
from __future__ import annotations

import argparse
import json

from g3e_data_engine import load_engine_config
from g3e_data_engine.converters.coco_to_yolo import CocoBox, convert_image_annotations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes-json", required=True, help="JSON list of {class_name,x_min,y_min,box_width,box_height}")
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    args = parser.parse_args()

    cfg = load_engine_config()
    class_name_to_id = {c.name: c.id for c in cfg.classes.classes}

    with open(args.boxes_json, "r", encoding="utf-8") as f:
        raw_boxes = json.load(f)
    boxes = [CocoBox(**b) for b in raw_boxes]

    lines = convert_image_annotations(boxes, class_name_to_id, args.image_width, args.image_height)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
