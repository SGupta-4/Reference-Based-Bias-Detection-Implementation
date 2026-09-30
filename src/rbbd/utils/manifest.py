"""Per-stage `manifest.json`: write, read and validate (D-030).

The runner writes a manifest with `complete=False` when a stage starts, updates
its `progress` field while the stage checkpoints, and rewrites it with
`complete=True` plus output hashes when the stage returns. A killed session
therefore leaves `complete=False`, which the runner treats as "resume", never as
"skip".

Output paths are stored relative to the artifact root so a manifest stays valid
when the root moves between Kaggle sessions.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rbbd.utils.cache import sha256_file
from rbbd.utils.logging import redact

MANIFEST_SCHEMA = 1


@dataclass
class Manifest:
    """One stage run. `run_key` is the content key of (config hash, stage, upstream keys)."""

    stage: str
    run_key: str
    config_hash: str
    git_sha: str
    input_hashes: dict[str, str] = field(default_factory=dict)
    output_hashes: dict[str, str] = field(default_factory=dict)
    started: str | None = None
    finished: str | None = None
    seconds: float | None = None
    env_summary: dict[str, Any] = field(default_factory=dict)
    complete: bool = False
    progress: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    schema: int = MANIFEST_SCHEMA


def manifest_path(artifacts_root: Path, stage: str, run_key: str) -> Path:
    """`<root>/manifests/<stage>/<run_key>.json` (ARCHITECTURE §3)."""
    return Path(artifacts_root) / "manifests" / stage / f"{run_key}.json"


def hash_path(path: Path) -> str:
    """SHA-256 of a file, or of the sorted `relpath:sha256` listing of a directory."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        lines = [
            f"{p.relative_to(path).as_posix()}:{sha256_file(p)}"
            for p in sorted(path.rglob("*"))
            if p.is_file()
        ]
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    raise FileNotFoundError(path)


def write(path: Path, manifest: Manifest) -> None:
    """Atomically write `manifest` as JSON; any error text is redacted first."""
    if manifest.error:
        manifest.error = redact(manifest.error)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    os.replace(tmp, path)


def read(path: Path) -> Manifest | None:
    """Return the manifest at `path`, or None if it is missing or unreadable."""
    try:
        raw = json.loads(Path(path).read_text())
        return Manifest(**raw)
    except (OSError, ValueError, TypeError):
        return None


def validate(
    manifest: Manifest | None,
    *,
    config_hash: str,
    input_hashes: dict[str, str],
    artifacts_root: Path,
) -> tuple[bool, str]:
    """Decide whether a stage may be skipped. Returns (valid, reason).

    Valid means: complete, same config hash, same upstream input hashes, and every
    recorded output still exists with the recorded hash.
    """
    if manifest is None:
        return False, "no manifest"
    if not manifest.complete:
        return False, "incomplete"
    if manifest.config_hash != config_hash:
        return False, "config hash changed"
    if manifest.input_hashes != input_hashes:
        return False, "upstream outputs changed"
    for rel, digest in manifest.output_hashes.items():
        target = Path(artifacts_root) / rel
        if not target.exists():
            return False, f"output missing: {rel}"
        if hash_path(target) != digest:
            return False, f"output changed: {rel}"
    return True, "valid"


def git_sha(repo_dir: Path | None = None) -> str:
    """Current commit SHA, suffixed `-dirty` if the work tree has changes; `unknown` outside git."""
    cwd = str(repo_dir) if repo_dir else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha
