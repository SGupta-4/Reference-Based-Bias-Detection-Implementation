"""Content-addressed cache (D-019). DC-10."""

import logging

import numpy as np
import pytest

from rbbd.utils.cache import TensorCache, make_key

BASE_FIELDS = {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "revision": "abc123",
    "checkpoint": {"regime": "qlora", "adapters": ["u1", "h1"], "alpha": 0.7},
    "layer": 32,
    "layer_site": "post_norm",
    "precision": "fp16",
    "quant": None,
    "placement": "balanced_2gpu",
    "tokenizer": {"add_special_tokens": True, "padding_side": "right", "max_len": 64},
    "sentence_union_hash": "s" * 64,
    "schema_version": 1,
}


def _mutations():
    yield "model_id", "mistralai/Mistral-7B-Instruct-v0.3"
    yield "revision", "def456"
    yield "checkpoint", {"regime": "qlora", "adapters": ["u1", "h1"], "alpha": 0.5}
    yield "layer", 31
    yield "layer_site", "pre_norm"
    yield "precision", "fp32"
    yield "quant", "nf4"
    yield "placement", "cuda:0"
    yield "tokenizer", {"add_special_tokens": False, "padding_side": "right", "max_len": 64}
    yield "sentence_union_hash", "t" * 64
    yield "schema_version", 2


def test_key_changes_with_any_field():
    """Changing any key field (alpha, precision, layer, sentence hash...) changes the key."""
    base = make_key(BASE_FIELDS)
    assert len(base) == 16
    seen = {base}
    for field, value in _mutations():
        key = make_key({**BASE_FIELDS, field: value})
        assert key not in seen, field
        seen.add(key)
    # Order-independent, and floats compare after rounding to 6 decimals.
    assert make_key(dict(reversed(list(BASE_FIELDS.items())))) == base
    assert make_key({"a": 0.1 + 0.2}) == make_key({"a": 0.3})
    with pytest.raises(ValueError):
        make_key({"a": float("nan")})


def test_write_then_lookup_logs_cache_hit(tmp_path, caplog):
    """A written entry is found with a `cache hit` log line and reads back bit-exact in fp16."""
    cache = TensorCache(tmp_path)
    ns = "embeddings/qwen2.5-0.5b-it/lora/a050"
    key = make_key(BASE_FIELDS)
    emb = np.random.default_rng(0).standard_normal((5, 8)).astype(np.float16)
    with caplog.at_level(logging.INFO, logger="rbbd"):
        assert cache.lookup(ns, key) is None
        cache.write(ns, key, {"mean": emb}, {"sentence_hashes": [f"h{i}" for i in range(5)]})
        assert cache.lookup(ns, key) == cache.path_for(ns, key)
    assert f"cache miss: {ns}/{key}" in caplog.text
    assert f"cache hit: {ns}/{key}" in caplog.text
    tensors, meta = cache.read(ns, key)
    assert tensors["mean"].dtype == np.float16 and np.array_equal(tensors["mean"], emb)
    assert meta["key"] == key and meta["sentence_hashes"][4] == "h4"
    with pytest.raises(FileExistsError):
        cache.write(ns, key, {"mean": emb}, {})
    with pytest.raises(ValueError):
        cache.path_for("../outside", key)


def test_second_run_logs_cache_hit_and_no_forward():
    """PLACEHOLDER(M2) Second identical run logs `cache hit` and the forward counter stays 0."""
