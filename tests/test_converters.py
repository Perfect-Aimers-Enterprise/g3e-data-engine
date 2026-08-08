from g3e_data_engine.converters.coco_to_yolo import CocoBox, coco_box_to_yolo_line, convert_image_annotations


def test_simple_box_conversion():
    box = CocoBox(class_name="person", x_min=100, y_min=100, box_width=200, box_height=400)
    line = coco_box_to_yolo_line(box, class_id=0, image_width=1000, image_height=1000)
    class_id, xc, yc, w, h = line.split()
    assert class_id == "0"
    assert abs(float(xc) - 0.2) < 1e-6
    assert abs(float(yc) - 0.3) < 1e-6
    assert abs(float(w) - 0.2) < 1e-6
    assert abs(float(h) - 0.4) < 1e-6


def test_out_of_bounds_box_is_clamped():
    box = CocoBox(class_name="car", x_min=-50, y_min=-50, box_width=100, box_height=100)
    line = coco_box_to_yolo_line(box, class_id=5, image_width=200, image_height=200)
    _, xc, yc, w, h = line.split()
    for v in (xc, yc, w, h):
        assert 0.0 <= float(v) <= 1.0


def test_unknown_class_is_skipped_not_raised():
    boxes = [
        CocoBox(class_name="person", x_min=0, y_min=0, box_width=10, box_height=10),
        CocoBox(class_name="unicorn", x_min=0, y_min=0, box_width=10, box_height=10),
    ]
    lines = convert_image_annotations(boxes, {"person": 0}, 100, 100)
    assert len(lines) == 1
