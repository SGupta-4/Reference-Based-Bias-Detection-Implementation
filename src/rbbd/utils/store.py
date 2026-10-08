"""Artifact store: the private HF repo `SarthakGupta414/rbbd-artifacts` (D-041).

Local artifacts are staged under `$RBBD_ARTIFACTS` (ARCHITECTURE §3) and mirrored
into the repo at the same relative paths, one subfolder per stage/checkpoint, with
`HfApi.upload_folder`. Rules enforced here, per the user's instructions:

- every upload first re-reads `repo_info` and refuses unless the repo is private;
- this package never creates a repo, changes its visibility, or pushes a model
  (`tests/test_store.py` scans `src/` for those calls);
- the token comes from `$HF_TOKEN` via huggingface_hub and is never printed.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from rbbd.utils.logging import get_logger, redact

log = get_logger(__name__)

REPO_TYPES = ("dataset", "model")


class StoreError(RuntimeError):
    """The store is unreachable, ambiguous, or not private."""


def _is_not_found(exc: Exception) -> bool:
    # Matched by name so this module (and its tests) do not import huggingface_hub
    # just to recognise the exception.
    return type(exc).__name__ in {"RepositoryNotFoundError", "RevisionNotFoundError"}


def _check_rel(path_in_repo: str) -> str:
    rel = PurePosixPath(path_in_repo)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise StoreError(f"path in repo must be relative without '..': {path_in_repo!r}")
    return rel.as_posix()


@dataclass
class HFStore:
    """Handle on the private repo. `repo_type` is resolved once by `open_store`."""

    repo_id: str
    repo_type: str
    api: Any

    def assert_private(self) -> None:
        """Re-read repo metadata and raise `StoreError` unless `private is True`."""
        info = self.api.repo_info(self.repo_id, repo_type=self.repo_type)
        if getattr(info, "private", None) is not True:
            raise StoreError(f"refusing to use {self.repo_id}: repo is not private")

    def upload_dir(self, local_dir: Path, path_in_repo: str, message: str) -> str:
        """Upload `local_dir` to `path_in_repo`; returns the commit id. Private check first."""
        rel = _check_rel(path_in_repo)
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise StoreError(f"not a directory: {local_dir}")
        self.assert_private()
        commit = self.api.upload_folder(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            folder_path=str(local_dir),
            path_in_repo=rel,
            commit_message=message,
        )
        oid = getattr(commit, "oid", None) or str(commit)
        log.info("uploaded %s -> %s:%s (commit %s)", local_dir, self.repo_id, rel, oid)
        return oid

    def download_file(self, path_in_repo: str, cache_dir: Path, force: bool = False) -> Path:
        """Download one file into `cache_dir` and return its local path."""
        rel = _check_rel(path_in_repo)
        local = self.api.hf_hub_download(
            repo_id=self.repo_id,
            filename=rel,
            repo_type=self.repo_type,
            cache_dir=str(cache_dir),
            force_download=force,
        )
        return Path(local)

    def download_dir(self, path_in_repo: str, local_root: Path) -> Path:
        """Mirror `path_in_repo/**` from the store into `local_root/path_in_repo` (restore).

        Used at the start of a session to bring back finished jobs (`done.json`) so the
        train stage skips them (D-072). Private check first. Returns the local directory.
        """
        rel = _check_rel(path_in_repo)
        self.assert_private()
        self.api.snapshot_download(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            allow_patterns=[f"{rel}/**"],
            local_dir=str(local_root),
        )
        return Path(local_root) / rel


def open_store(repo_id: str, repo_type: str = "auto", api: Any = None) -> HFStore:
    """Resolve the repo type (probing dataset, then model, when "auto") and check privacy."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    candidates = REPO_TYPES if repo_type == "auto" else (repo_type,)
    found = []
    for candidate in candidates:
        try:
            api.repo_info(repo_id, repo_type=candidate)
        except Exception as exc:
            if _is_not_found(exc):
                continue
            raise StoreError(
                redact(f"cannot read {repo_id} ({candidate}): {type(exc).__name__}: {exc}")
            ) from None
        found.append(candidate)
    if not found:
        raise StoreError(
            f"{repo_id} not found as {' or '.join(candidates)} (it must already exist)"
        )
    if len(found) > 1:
        raise StoreError(f"{repo_id} exists as both {found}; set store.repo_type explicitly")
    store = HFStore(repo_id=repo_id, repo_type=found[0], api=api)
    store.assert_private()
    return store


def roundtrip(
    store: HFStore, artifacts_root: Path, session: str, size: int = 1 << 20
) -> dict[str, Any]:
    """M0-T7: upload a random `size`-byte file under `env/<session>/` and read it back.

    The read uses a fresh empty cache with `force_download=True`, so a match proves
    the bytes persisted in the remote repo, not in a local cache (D-045).
    """
    rel_dir = f"env/{session}"
    local_dir = Path(artifacts_root) / rel_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    payload = os.urandom(size)
    (local_dir / "roundtrip.bin").write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    commit = store.upload_dir(local_dir, rel_dir, f"M0 store round trip {session}")
    with tempfile.TemporaryDirectory() as cache:
        path = store.download_file(f"{rel_dir}/roundtrip.bin", Path(cache), force=True)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "ok": actual == expected,
        "repo_id": store.repo_id,
        "repo_type": store.repo_type,
        "private": True,
        "commit": commit,
        "bytes": size,
        "sha256": expected,
    }
