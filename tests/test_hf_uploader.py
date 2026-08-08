import sys
import types

import pytest

from g3e_data_engine.core.exceptions import MissingCredentialError


def _install_fake_huggingface_hub(monkeypatch, capture: dict):
    fake_module = types.ModuleType("huggingface_hub")

    class FakeHfApi:
        def __init__(self, token=None):
            capture["init_token"] = token

        def create_repo(self, repo_id, repo_type, private, exist_ok, token):
            capture["create_repo"] = dict(
                repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=exist_ok, token=token
            )

        def upload_folder(self, repo_id, repo_type, folder_path, token, commit_message):
            capture["upload_folder"] = dict(
                repo_id=repo_id, repo_type=repo_type, folder_path=folder_path,
                token=token, commit_message=commit_message,
            )

    fake_module.HfApi = FakeHfApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)


def test_upload_release_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "release").mkdir()

    from g3e_data_engine.exporters.hf_uploader import upload_release_to_hf

    with pytest.raises(MissingCredentialError):
        upload_release_to_hf(folder=tmp_path / "release", repo_id="org/dataset")


def test_upload_release_calls_hf_api_with_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    capture: dict = {}
    _install_fake_huggingface_hub(monkeypatch, capture)

    release_dir = tmp_path / "g3e-vision-dataset-v1.1.0"
    release_dir.mkdir()

    from g3e_data_engine.exporters.hf_uploader import upload_release_to_hf

    url = upload_release_to_hf(folder=release_dir, repo_id="your-org/g3e-vision-dataset", private=True)

    assert url == "https://huggingface.co/datasets/your-org/g3e-vision-dataset"
    assert capture["create_repo"]["repo_id"] == "your-org/g3e-vision-dataset"
    assert capture["create_repo"]["private"] is True
    assert capture["create_repo"]["token"] == "test-token"
    assert capture["upload_folder"]["folder_path"] == str(release_dir)


def test_upload_release_raises_if_folder_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    _install_fake_huggingface_hub(monkeypatch, {})

    from g3e_data_engine.exporters.hf_uploader import upload_release_to_hf

    with pytest.raises(FileNotFoundError):
        upload_release_to_hf(folder=tmp_path / "does-not-exist", repo_id="org/dataset")
