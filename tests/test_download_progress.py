from g3e_data_engine.downloader.progress import load_progress, save_progress, progress_path


def test_load_progress_returns_empty_dict_when_no_manifest(tmp_path):
    assert load_progress(tmp_path) == {}


def test_save_then_load_round_trips(tmp_path):
    save_progress(tmp_path, row_offset=42, per_class_counts={"person": 10, "car": 3})
    loaded = load_progress(tmp_path)
    assert loaded["row_offset"] == 42
    assert loaded["per_class_counts"] == {"person": 10, "car": 3}


def test_save_progress_merges_rather_than_overwrites(tmp_path):
    save_progress(tmp_path, row_offset=1, per_class_counts={"person": 1})
    save_progress(tmp_path, per_class_counts={"person": 2})  # row_offset omitted this time
    loaded = load_progress(tmp_path)
    assert loaded["row_offset"] == 1  # untouched by the second call
    assert loaded["per_class_counts"] == {"person": 2}  # overwritten by the second call


def test_save_progress_is_atomic_no_tmp_file_left_behind(tmp_path):
    save_progress(tmp_path, row_offset=1)
    assert progress_path(tmp_path).exists()
    assert not progress_path(tmp_path).with_suffix(".json.tmp").exists()


def test_load_progress_survives_a_corrupt_manifest(tmp_path):
    path = progress_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")  # simulates a crash mid-write

    assert load_progress(tmp_path) == {}  # must not raise


def test_completed_flag_round_trips(tmp_path):
    save_progress(tmp_path, completed=False)
    assert load_progress(tmp_path)["completed"] is False
    save_progress(tmp_path, completed=True)
    assert load_progress(tmp_path)["completed"] is True
