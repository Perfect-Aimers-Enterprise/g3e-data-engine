"""
Convert COCO-style bounding boxes ([x_min, y_min, width, height], absolute
pixels) into YOLO-style labels (class_id x_center y_center width height,
all normalized 0-1, one .txt file per image).

This is the target annotation format declared in DATASET_SPEC.md. If you
add a second target format later, add a sibling module here
(e.g. coco_to_coco.py for a passthrough / re-id case) rather than branching
inside this one.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CocoBox:
    class_name: str
    x_min: float
    y_min: float
    box_width: float
    box_height: float


def coco_box_to_yolo_line(box: CocoBox, class_id: int, image_width: int, image_height: int) -> str:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")

    x_center = box.x_min + box.box_width / 2.0
    y_center = box.y_min + box.box_height / 2.0

    x_center_n = x_center / image_width
    y_center_n = y_center / image_height
    width_n = box.box_width / image_width
    height_n = box.box_height / image_height

    # Clamp to [0, 1] — source annotations occasionally overshoot the image
    # bounds by a pixel or two; silently dropping the box would lose data,
    # silently keeping an out-of-range value would corrupt training, so clamp.
    x_center_n = min(max(x_center_n, 0.0), 1.0)
    y_center_n = min(max(y_center_n, 0.0), 1.0)
    width_n = min(max(width_n, 0.0), 1.0)
    height_n = min(max(height_n, 0.0), 1.0)

    return f"{class_id} {x_center_n:.6f} {y_center_n:.6f} {width_n:.6f} {height_n:.6f}"


def convert_image_annotations(
    boxes: list[CocoBox],
    class_name_to_id: dict[str, int],
    image_width: int,
    image_height: int,
) -> list[str]:
    lines = []
    for box in boxes:
        if box.class_name not in class_name_to_id:
            # Class not part of v1's supported set (classes.yaml) — skip it
            # rather than raising, since sources often carry extra classes
            # we deliberately don't want yet.
            continue
        class_id = class_name_to_id[box.class_name]
        lines.append(coco_box_to_yolo_line(box, class_id, image_width, image_height))
    return lines
