"""Private HF artifact store (D-041). Uses an in-memory fake of HfApi; no network."""

import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from rbbd.utils import store as store_mod
from tests.conftest import REPO_ROOT

REPO = "SarthakGupta414/rbbd-artifacts"


class RepositoryNotFoundError(Exception):
    """Same name as huggingface_hub's, which is how store.py recognises it."""


class FakeApi:
    """Minimal HfApi stand-in backed by a local directory as the 'remote'."""

    def __init__(self, remote: Path, repo_type="model", private=True):
        self.remote, self.repo_type, self.private = remote, repo_type, private
        self.uploads = []

    def repo_info(self, repo_id, repo_type="model"):
        if repo_id != REPO or repo_type != self.repo_type:
            raise RepositoryNotFoundError(repo_id)
        return SimpleNamespace(private=self.private)

    def upload_folder(self, *, repo_id, repo_type, folder_path, path_in_repo, commit_message):
        self.uploads.append(path_in_repo)
        shutil.copytree(folder_path, self.remote / path_in_repo, dirs_exist_ok=True)
        return SimpleNamespace(oid=f"commit{len(self.uploads)}")

    def snapshot_download(self, *, repo_id, repo_type, allow_patterns, local_dir):
        import fnmatch

        for f in self.remote.rglob("*"):
            rel = f.relative_to(self.remote).as_posix()
            if f.is_file() and any(fnmatch.fnmatch(rel, pat) for pat in allow_patterns):
                dest = Path(local_dir) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dest)
        return local_dir

    def hf_hub_download(self, *, repo_id, filename, repo_type, cache_dir, force_download):
        dest = Path(cache_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.remote / filename, dest)
        return str(dest)


def test_open_store_resolves_type_and_requires_private(tmp_path):
    """ "auto" finds the existing repo type; a public or missing repo is refused."""
    assert store_mod.open_store(REPO, api=FakeApi(tmp_path, "model")).repo_type == "model"
    assert store_mod.open_store(REPO, api=FakeApi(tmp_path, "dataset")).repo_type == "dataset"
    with pytest.raises(store_mod.StoreError, match="not private"):
        store_mod.open_store(REPO, api=FakeApi(tmp_path, private=False))
    with pytest.raises(store_mod.StoreError, match="must already exist"):
        store_mod.open_store("someone/else", api=FakeApi(tmp_path))


def test_upload_rechecks_privacy_and_paths(tmp_path):
    """Visibility is re-read before each upload; '..' paths are rejected; nothing uploads."""
    api = FakeApi(tmp_path / "remote")
    store = store_mod.open_store(REPO, api=api)
    local = tmp_path / "local"
    local.mkdir()
    (local / "a.txt").write_text("x")
    with pytest.raises(store_mod.StoreError, match="relative"):
        store.upload_dir(local, "../escape", "m")
    api.private = False
    with pytest.raises(store_mod.StoreError, match="not private"):
        store.upload_dir(local, "env/s1", "m")
    assert api.uploads == []


def test_roundtrip_with_fake_api(tmp_path):
    """M0-T7 logic: 1 MB up, forced fresh download, hashes equal."""
    api = FakeApi(tmp_path / "remote")
    result = store_mod.roundtrip(
        store_mod.open_store(REPO, api=api), tmp_path / "artifacts", "s1", size=4096
    )
    assert result["ok"] and result["private"] and result["repo_type"] == "model"
    assert api.uploads == ["env/s1"]


def test_no_repo_creation_or_publication_calls():
    """No code under src/ creates, deletes, moves or re-publishes a repo, or pushes to the Hub."""
    forbidden = re.compile(
        r"\b(create_repo|delete_repo|move_repo|update_repo_settings|update_repo_visibility|push_to_hub)\s*\("
    )
    hits = [
        f"{p.relative_to(REPO_ROOT)}:{i}"
        for p in (REPO_ROOT / "src").rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if forbidden.search(line)
    ]
    assert not hits, hits


def test_download_dir_restores_a_subtree(tmp_path):
    """`restore` brings back one job subtree (and nothing else) into the artifact root."""
    remote = tmp_path / "remote"
    (remote / "train/x/full/harmful/seed0/k1/final").mkdir(parents=True)
    (remote / "train/x/full/harmful/seed0/k1/done.json").write_text("{}")
    (remote / "train/x/full/harmful/seed0/k1/final/model.safetensors").write_bytes(b"w")
    (remote / "env/s1").mkdir(parents=True)
    (remote / "env/s1/env.json").write_text("{}")
    store = store_mod.open_store(REPO, api=FakeApi(remote))
    root = tmp_path / "artifacts"
    local = store.download_dir("train/x", root)
    assert (local / "full/harmful/seed0/k1/done.json").exists()
    assert (local / "full/harmful/seed0/k1/final/model.safetensors").read_bytes() == b"w"
    assert not (root / "env").exists()
    with pytest.raises(store_mod.StoreError, match="relative"):
        store.download_dir("../x", root)
