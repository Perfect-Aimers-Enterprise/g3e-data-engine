from pathlib import Path

import numpy as np
from PIL import Image

from g3e_data_engine.core.config import ImageThresholds
from g3e_data_engine.validators.image_quality import validate_image


def _make_image(path: Path, size=(800, 800), noisy=True, brightness=128):
    if noisy:
        arr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    else:
        arr = np.full((size[1], size[0], 3), brightness, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_accepts_good_image(tmp_path):
    p = tmp_path / "good.jpg"
    _make_image(p, size=(800, 800), noisy=True)
    thresholds = ImageThresholds(min_shorter_side=640, blur_threshold=1.0, max_brightness=250, min_brightness=5)
    result = validate_image(p, thresholds)
    assert result.accepted, result.reasons


def test_rejects_low_resolution(tmp_path):
    p = tmp_path / "small.jpg"
    _make_image(p, size=(100, 100), noisy=True)
    thresholds = ImageThresholds(min_shorter_side=640, blur_threshold=1.0, max_brightness=250, min_brightness=5)
    result = validate_image(p, thresholds)
    assert not result.accepted
    assert any("resolution" in r for r in result.reasons)


def test_accepts_ordinary_landscape_photo_shape(tmp_path):
    """
    Regression test: a plain 640x480 photo (COCO's most common shape) must
    be accepted at a reasonable threshold. The old width-AND-height check
    (min_width=640, min_height=640) rejected this exact shape, silently
    throwing away the large majority of a typical detection dataset.
    """
    p = tmp_path / "landscape.jpg"
    _make_image(p, size=(640, 480), noisy=True)  # shorter side = 480
    thresholds = ImageThresholds(min_shorter_side=416, blur_threshold=1.0, max_brightness=250, min_brightness=5)
    result = validate_image(p, thresholds)
    assert result.accepted, result.reasons


def test_accepts_ordinary_portrait_photo_shape(tmp_path):
    p = tmp_path / "portrait.jpg"
    _make_image(p, size=(480, 640), noisy=True)  # shorter side = 480
    thresholds = ImageThresholds(min_shorter_side=416, blur_threshold=1.0, max_brightness=250, min_brightness=5)
    result = validate_image(p, thresholds)
    assert result.accepted, result.reasons


def test_rejects_when_shorter_side_below_threshold_even_if_long_side_is_huge(tmp_path):
    """A 2000x300 banner-shaped image has a tiny shorter side and should still be rejected."""
    p = tmp_path / "banner.jpg"
    _make_image(p, size=(2000, 300), noisy=True)
    thresholds = ImageThresholds(min_shorter_side=416, blur_threshold=1.0, max_brightness=250, min_brightness=5)
    result = validate_image(p, thresholds)
    assert not result.accepted
    assert any("resolution" in r for r in result.reasons)


def test_shipped_default_min_shorter_side_is_416():
    assert ImageThresholds().min_shorter_side == 416


def test_rejects_blank_flat_image_as_blurry(tmp_path):
    p = tmp_path / "flat.jpg"
    _make_image(p, size=(800, 800), noisy=False, brightness=128)
    thresholds = ImageThresholds(min_shorter_side=640, blur_threshold=5.0, max_brightness=250, min_brightness=5)
    result = validate_image(p, thresholds)
    assert not result.accepted
    assert any("blurry" in r for r in result.reasons)


def test_rejects_corrupted_file(tmp_path):
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"not an image")
    thresholds = ImageThresholds()
    result = validate_image(p, thresholds)
    assert not result.accepted
    assert any("corrupted" in r for r in result.reasons)
