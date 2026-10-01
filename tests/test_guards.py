"""NaN/Inf guard (D-006). DC-09."""

import numpy as np
import pytest

from rbbd.utils.cache import TensorCache
from rbbd.utils.guards import NonFiniteError, assert_finite, assert_finite_scalar


def _guarded_write(cache, batches):
    """Mimics the embed loop (M2): guard every batch, write the cache only if all pass."""
    pooled = []
    for i, hidden in enumerate(batches):
        assert_finite(hidden, "embed.hidden_states", batch=i, ckpt="a050")
        pooled.append(hidden.mean(axis=1))
    cache.write("embeddings/test", "k", {"mean": np.concatenate(pooled)}, {})


def test_guard_triggers_on_injected_inf(tmp_path):
    """Injected Inf raises NonFiniteError naming the batch; no cache file is written."""
    batches = [np.ones((2, 3, 4), np.float16) for _ in range(3)]
    batches[1][0, 2, 1] = np.inf
    cache = TensorCache(tmp_path)
    with pytest.raises(NonFiniteError, match=r"inf=1 .*batch=1"):
        _guarded_write(cache, batches)
    assert not list(tmp_path.rglob("*.safetensors"))
    batches[1][0, 2, 1] = 1.0
    _guarded_write(cache, batches)
    assert cache.lookup("embeddings/test", "k") is not None


def test_guard_scalar_and_nan():
    """NaN in an array and a non-finite loss both raise; finite values pass silently."""
    with pytest.raises(NonFiniteError, match="nan=1"):
        assert_finite(np.array([1.0, np.nan]), "x")
    with pytest.raises(NonFiniteError, match="train.loss"):
        assert_finite_scalar(float("inf"), "train.loss", step=7)
    assert_finite(np.zeros(3), "ok")
    assert_finite_scalar(0.5, "ok")
