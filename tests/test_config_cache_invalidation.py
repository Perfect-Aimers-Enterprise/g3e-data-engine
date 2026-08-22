"""
Regression tests for a real reported bug: load_engine_config() used a plain
functools.lru_cache keyed only on the configs directory path. In a
long-running process (a notebook kernel, an API server) someone would edit
configs/datasets.yaml — e.g. flipping license.verified: true after actually
reviewing a source's terms — and every subsequent call to
load_engine_config() kept silently returning the PRE-EDIT config, because
the cache had no way to detect the files on disk had changed. The person
would see a preflight failure quoting license text they'd already replaced,
with no indication their edit hadn't taken effect.

Fixed by fingerprinting each config file's (mtime_ns, size) and
invalidating the cache automatically when any of them change, rather than
requiring a manual load_engine_config.cache_clear() call nobody knew to make.
"""
import textwrap

from g3e_data_engine.core.config import load_engine_config, clear_config_cache


def _write_configs(base, datasets_yaml_body):
    (base / "classes.yaml").write_text(textwrap.dedent("""
        version: "1.0.0"
        classes:
          - name: person
            id: 0
            priority_tier: 1
    """))
    (base / "datasets.yaml").write_text(datasets_yaml_body)
    (base / "processing.yaml").write_text(textwrap.dedent("""
        image: {min_shorter_side: 416, blur_threshold: 90, max_brightness: 250, min_brightness: 35}
        duplicates: {enabled: true, method: phash, hamming_distance_threshold: 5}
        labels: {remove_empty: false}
        split: {train: 0.8, val: 0.1, test: 0.1, seed: 42}
    """))
    (base / "priority.yaml").write_text(textwrap.dedent("""
        budget: {total_images: 1000, max_per_class: 2500, min_per_class: 10}
        tier_weights: {1: 4, 2: 3, 3: 2, 4: 1}
        overrides: {}
    """))


def _unverified_yaml():
    return textwrap.dedent("""
        sources:
          weapons:
            enabled: true
            kind: huggingface
            hf_repo: "some/repo"
            classes: [person]
            max_images: 10
            license: {name: "UNKNOWN", verified: false}
        global_max_images: 8000
    """)


def _verified_yaml():
    return textwrap.dedent("""
        sources:
          weapons:
            enabled: true
            kind: huggingface
            hf_repo: "some/repo"
            classes: [person]
            max_images: 10
            license: {name: "Apache-2.0", verified: true}
        global_max_images: 8000
    """)


def test_editing_a_config_file_is_picked_up_without_manual_cache_clear(tmp_path):
    clear_config_cache()
    _write_configs(tmp_path, _unverified_yaml())

    cfg1 = load_engine_config(configs_dir=str(tmp_path))
    assert cfg1.datasets.sources["weapons"].license.verified is False

    # Simulate editing the file in the same running process — no restart,
    # no manual cache_clear() call. This is exactly the notebook workflow
    # that triggered the real bug.
    _write_configs(tmp_path, _verified_yaml())

    cfg2 = load_engine_config(configs_dir=str(tmp_path))
    assert cfg2.datasets.sources["weapons"].license.verified is True
    assert cfg2.datasets.sources["weapons"].license.name == "Apache-2.0"


def test_repeated_calls_without_file_changes_return_a_cached_object(tmp_path):
    """The fix shouldn't defeat caching entirely — unchanged files should
    still return the same cached instance, not re-parse every call."""
    clear_config_cache()
    _write_configs(tmp_path, _unverified_yaml())

    cfg1 = load_engine_config(configs_dir=str(tmp_path))
    cfg2 = load_engine_config(configs_dir=str(tmp_path))
    assert cfg1 is cfg2


def test_clear_config_cache_forces_a_fresh_read(tmp_path):
    clear_config_cache()
    _write_configs(tmp_path, _unverified_yaml())
    cfg1 = load_engine_config(configs_dir=str(tmp_path))

    clear_config_cache()
    cfg2 = load_engine_config(configs_dir=str(tmp_path))
    assert cfg1 is not cfg2  # fresh object, even though file content is unchanged
    assert cfg1.datasets.sources["weapons"].license.verified == cfg2.datasets.sources["weapons"].license.verified
