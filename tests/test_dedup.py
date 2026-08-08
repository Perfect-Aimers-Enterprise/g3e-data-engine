import numpy as np
from PIL import Image

from g3e_data_engine.core.config import DuplicatesConfig
from g3e_data_engine.dedup.phash_dedup import find_duplicates


def test_identical_images_are_flagged_as_duplicates(tmp_path):
    arr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    Image.fromarray(arr).save(p1)
    Image.fromarray(arr).save(p2)  # exact duplicate content, different file

    config = DuplicatesConfig(enabled=True, method="phash", hamming_distance_threshold=5)
    result = find_duplicates([p1, p2], config)

    assert len(result.kept) == 1
    assert len(result.duplicates) == 1


def test_disabled_dedup_keeps_everything(tmp_path):
    arr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    Image.fromarray(arr).save(p1)
    Image.fromarray(arr).save(p2)

    config = DuplicatesConfig(enabled=False)
    result = find_duplicates([p1, p2], config)
    assert len(result.kept) == 2
    assert len(result.duplicates) == 0


def test_distinct_images_are_not_flagged(tmp_path):
    # Flat solid-color images are a degenerate case for perceptual hashing
    # (near-zero variance in every DCT block), so use structured/random
    # content — representative of real photos — for a meaningful check.
    rng = np.random.default_rng(0)
    arr1 = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    arr2 = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    p1 = tmp_path / "random1.jpg"
    p2 = tmp_path / "random2.jpg"
    Image.fromarray(arr1).save(p1)
    Image.fromarray(arr2).save(p2)

    config = DuplicatesConfig(enabled=True, method="phash", hamming_distance_threshold=5)
    result = find_duplicates([p1, p2], config)
    assert len(result.kept) == 2
    assert len(result.duplicates) == 0
