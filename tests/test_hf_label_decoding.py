"""
Regression tests for the real bug reported against this engine: HF
object-detection datasets store `objects.category` (and sometimes `label`)
as an INTEGER ClassLabel id, not a string. Comparing that int against G3E's
string class names silently matched nothing on every row — the downloader
would stream the entire dataset (30+ minutes, in the reported case) and
accept zero images, with no indication of why.

These tests use the REAL `datasets` library's `Features`/`ClassLabel`/
`Sequence` classes (not hand-rolled mocks) so a future `datasets` version
that changes how `Sequence({...})` normalizes internally would be caught
here rather than only in production.
"""
from datasets import Features, Sequence, ClassLabel, Value

from g3e_data_engine.downloader.hf_downloader import _resolve_label_names, _extract_class_names


class _FakeIterableDataset:
    """Minimal stand-in for datasets.IterableDataset — only `.features` is used."""

    def __init__(self, features):
        self.features = features


def test_resolves_int_categories_from_objects_sequence():
    """Reproduces detection-datasets/coco's actual schema shape."""
    features = Features({
        "image": Value("string"),
        "objects": Sequence({
            "category": ClassLabel(names=["person", "bicycle", "car", "motorcycle", "dog", "cat"]),
            "bbox": Sequence(Value("float32"), length=4),
        }),
    })
    obj_names, label_names = _resolve_label_names(_FakeIterableDataset(features))

    assert obj_names == ["person", "bicycle", "car", "motorcycle", "dog", "cat"]
    assert label_names is None


def test_decodes_int_category_ids_to_class_names():
    features = Features({
        "objects": Sequence({
            "category": ClassLabel(names=["person", "bicycle", "car", "motorcycle", "dog", "cat"]),
        }),
    })
    obj_names, _ = _resolve_label_names(_FakeIterableDataset(features))

    row = {"objects": {"category": [2, 4]}}  # 2="car", 4="dog"
    classes = _extract_class_names(row, allowed=["person", "car", "dog", "cat"], objects_category_names=obj_names)

    assert classes == ["car", "dog"]


def test_resolves_bare_label_classlabel():
    features = Features({"image": Value("string"), "label": ClassLabel(names=["gun", "knife", "person"])})
    obj_names, label_names = _resolve_label_names(_FakeIterableDataset(features))

    assert label_names == ["gun", "knife", "person"]

    row = {"label": 0}
    classes = _extract_class_names(row, allowed=["gun", "knife", "person"], label_names=label_names)
    assert classes == ["gun"]


def test_still_handles_datasets_that_already_use_plain_strings():
    """A source without ClassLabel-encoded categories must keep working exactly as before."""
    row = {"objects": {"category": ["car", "dog"]}}
    classes = _extract_class_names(row, allowed=["car", "dog"])  # no names table supplied
    assert classes == ["car", "dog"]


def test_resolve_label_names_returns_none_none_when_features_unavailable():
    class _NoFeatures:
        features = None

    obj_names, label_names = _resolve_label_names(_NoFeatures())
    assert obj_names is None
    assert label_names is None


def test_class_map_applies_after_int_decoding():
    """weapons-style sources need BOTH: int->name decoding AND class_map translation."""
    features = Features({"objects": Sequence({"category": ClassLabel(names=["GUN", "KNIFE", "PERSON"])})})
    obj_names, _ = _resolve_label_names(_FakeIterableDataset(features))

    row = {"objects": {"category": [0, 2]}}  # GUN, PERSON
    classes = _extract_class_names(
        row,
        allowed=["gun", "knife", "person"],
        class_map={"GUN": "gun", "KNIFE": "knife", "PERSON": "person"},
        objects_category_names=obj_names,
    )
    assert classes == ["gun", "person"]


def test_out_of_range_int_category_is_ignored_not_crashed():
    row = {"objects": {"category": [999]}}  # index far beyond any real names table
    classes = _extract_class_names(row, allowed=["car"], objects_category_names=["person", "car"])
    assert classes == []
