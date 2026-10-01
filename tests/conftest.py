"""Shared fixtures: isolated artifact root, a two-file config directory, synthetic embeddings."""

from pathlib import Path

import numpy as np
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


def make_embedding_set(seed=0, d=16, n_per_group=6, groups=("A", "B", "C"), p=8, q=8, m=40):
    """Synthetic EmbeddingSet: Gaussian rows with a shared offset so cosines are non-trivial."""
    from rbbd.metrics.delta_b import EmbeddingSet

    rng = np.random.default_rng(seed)
    shift = rng.normal(size=d)

    def rows(k):
        return rng.normal(size=(k, d)) + shift

    labels = tuple(g for g in groups for _ in range(n_per_group))
    return EmbeddingSet(rows(len(labels)), labels, rows(p), rows(q), rows(m))


@pytest.fixture
def emb():
    return make_embedding_set()
