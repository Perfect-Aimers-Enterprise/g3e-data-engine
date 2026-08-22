"""
Image quality validation: resolution, blur, brightness.

Resolution is checked against the SHORTER side of the image, not width and
height independently. Rejecting on `width < X or height < X` effectively
demands both dimensions clear the same bar — which only near-square images
satisfy. A completely ordinary 640x480 (or 480x640) photo, the single most
common shape in datasets like COCO, would fail a `min_width=640,
min_height=640` check even though it's a perfectly good image; that
mismatch was a real bug (see DATASET_SPEC.md section 4) that silently threw
away the large majority of otherwise-valid downloaded images. Checking
`min(width, height)` against one threshold is aspect-ratio-independent and
matches how "is this image high enough resolution to be useful" actually
gets judged in practice.

Blur detection uses Laplacian variance (a lightweight, dependency-light
proxy for sharpness) computed with numpy convolution instead of pulling in
OpenCV, so the engine's dependency footprint stays small.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from g3e_data_engine.core.config import ImageThresholds

_LAPLACIAN_KERNEL = np.array(
    [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32
)


@dataclass
class ValidationResult:
    path: str
    accepted: bool
    width: int = 0
    height: int = 0
    blur_score: float = 0.0
    brightness: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Cheap 2D convolution with a Laplacian kernel; returns variance of the response."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    padded = np.pad(gray, 1, mode="reflect")
    response = np.zeros_like(gray, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            k = _LAPLACIAN_KERNEL[dy, dx]
            if k == 0:
                continue
            response += k * padded[dy : dy + h, dx : dx + w]
    return float(response.var())


def validate_image(path: str | Path, thresholds: ImageThresholds) -> ValidationResult:
    path = str(path)
    reasons: list[str] = []

    try:
        with Image.open(path) as img:
            img = img.convert("L")  # grayscale for blur/brightness
            width, height = img.size
            arr = np.asarray(img, dtype=np.float32)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return ValidationResult(path=path, accepted=False, reasons=[f"corrupted_or_unreadable: {exc}"])

    if min(width, height) < thresholds.min_shorter_side:
        reasons.append(
            f"resolution_too_low (shorter_side={min(width, height)}, image={width}x{height}, "
            f"min_shorter_side={thresholds.min_shorter_side})"
        )

    brightness = float(arr.mean())
    if brightness < thresholds.min_brightness:
        reasons.append(f"too_dark (brightness={brightness:.1f})")
    if brightness > thresholds.max_brightness:
        reasons.append(f"too_bright (brightness={brightness:.1f})")

    blur_score = _laplacian_variance(arr)
    if blur_score < thresholds.blur_threshold:
        reasons.append(f"too_blurry (score={blur_score:.1f})")

    return ValidationResult(
        path=path,
        accepted=len(reasons) == 0,
        width=width,
        height=height,
        blur_score=blur_score,
        brightness=brightness,
        reasons=reasons,
    )


def validate_batch(paths: list[str | Path], thresholds: ImageThresholds) -> list[ValidationResult]:
    return [validate_image(p, thresholds) for p in paths]
