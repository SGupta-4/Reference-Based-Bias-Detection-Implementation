"""Shared fixtures. M0: isolated artifact root and a two-file config directory."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """Point `$RBBD_ARTIFACTS` at an empty per-test directory."""
    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.setenv("RBBD_ARTIFACTS", str(root))
    return root


@pytest.fixture
def config_dir(tmp_path):
    """Write `base.yaml` + a tier file and return a function tier_text -> tier path."""

    def make(base_text: str, tier_text: str, tier_name: str = "tier.yaml") -> Path:
        d = tmp_path / "configs"
        d.mkdir(exist_ok=True)
        (d / "base.yaml").write_text(base_text)
        (d / tier_name).write_text(tier_text)
        return d / tier_name

    return make
