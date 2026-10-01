"""Batched extraction, guard, cache and the embed stage on a tiny CPU model (M2)."""

import logging

import numpy as np
import pytest
import torch

from rbbd import config, runner
from rbbd.embed import extract
from rbbd.embed.extract import ExtractSettings, encode, extract_cached, make_batches
from rbbd.utils import env as env_mod
from rbbd.utils.cache import TensorCache
from rbbd.utils.guards import NonFiniteError
from tests.conftest import FakeTokenizer, make_tiny_model

TEXTS = [
    "Women attend community events.",
    "She danced with pure joy in her heart.",
    "He felt shame after being caught in a lie.",
    "Many Asians participate in online discussions.",
    "Homosexuals speak multiple languages and dialects.",
    "Rich people book trips for the summer.",
]


def test_make_batches_budget_and_order():
    """Longest first; every batch respects size x longest <= budget; all rows covered once."""
    lengths = [3, 9, 5, 9, 1, 7, 2]
    batches = make_batches(lengths, token_budget=18, max_batch=4)
    assert sorted(i for b in batches for i in b) == list(range(7))
    assert batches[0][0] in (1, 3)
    for b in batches:
        assert len(b) * max(lengths[i] for i in b) <= 18 or len(b) == 1


def test_encode_shapes_order_and_sites(tiny):
    """Rows come back in input order regardless of batching; pre-norm differs from post-norm."""
    model, tok = tiny
    one_by_one = ExtractSettings(token_budget=1, layer_sites=("post_norm", "pre_norm"))
    all_at_once = ExtractSettings(token_budget=1 << 20, layer_sites=("post_norm", "pre_norm"))
    a = encode(model, tok, TEXTS, one_by_one)
    b = encode(model, tok, TEXTS, all_at_once)
    assert sorted(a) == ["last", "max", "mean", "pre_norm/last", "pre_norm/max", "pre_norm/mean"]
    for name in a:
        assert a[name].shape == (6, 16) and a[name].dtype == np.float16
        np.testing.assert_allclose(
            a[name].astype(np.float32), b[name].astype(np.float32), atol=1e-3
        )
    assert not np.allclose(a["mean"], a["pre_norm/mean"])


def test_guard_fires_on_real_forward(tmp_path, tiny):
    """An Inf injected into the last layer's output raises NonFiniteError; nothing is cached."""
    model, tok = tiny

    def poison(_m, _i, output):
        out = output[0] if isinstance(output, tuple) else output
        out[0, 1, 3] = float("inf")
        return output

    handle = model.model.layers[-1].register_forward_hook(poison)
    cache = TensorCache(tmp_path)
    try:
        with pytest.raises(NonFiniteError, match="embed.post_norm.*batch=0"):
            extract_cached(cache, "embeddings/t", {"k": 1}, TEXTS, ExtractSettings(),
                           lambda: (model, tok), context={"ckpt": "ref"})  # fmt: skip
    finally:
        handle.remove()
    assert not list(tmp_path.rglob("*.safetensors"))


def test_fp16_overflow_caught(tiny):
    """Pooled values beyond the fp16 range (possible from fp32 models) are caught before caching."""
    model, tok = tiny
    with torch.no_grad():
        model.model.norm.weight.fill_(1e6)
    with pytest.raises(NonFiniteError, match="fp16"):
        encode(model, tok, TEXTS[:2], ExtractSettings())


def _smoke_cfg(config_dir):
    base = (
        "run_name: base\nseed: 0\nschema_version: {embed: 1}\n"
        "embed: {checkpoints: [ref], poolings: [mean, max, last], layer_sites: [post_norm],"
        " max_length: 64, token_budget: 4096, max_batch: 64}\n"
    )
    tier = (
        "run_name: t\nmodels:\n"
        "  - {id: tiny/llama, slug: tiny, layer: 2, precision: fp32, placement: cpu}\n"
        "sentences:\n  subset: {groups: [Women, Muslims, Immigrants], n_targets: 5, n_attr: 10,"
        " n_anchors: 50, variants: [base]}\n"
    )
    return config.load(config_dir(base, tier))


def test_embed_stage_and_cache_hit(config_dir, artifacts, monkeypatch, caplog):
    """Stage writes one cache entry for the smoke subset; a forced re-run hits the cache and
    never loads the model (DC-10)."""
    loads = []

    def fake_load(spec):
        loads.append(spec)
        return make_tiny_model(), FakeTokenizer()

    monkeypatch.setattr("rbbd.models.loading.load_for_inference", fake_load)
    monkeypatch.setattr(extract, "resolve_revision", lambda mid, rev: "sha-test")
    monkeypatch.setattr("rbbd.models.loading.download", lambda spec: "/nonexistent")
    cfg = _smoke_cfg(config_dir)
    env = env_mod.detect()
    assert runner.run(cfg, env, ["sentences", "embed"]) == {"sentences": "ran", "embed": "ran"}
    assert len(loads) == 1 and loads[0].revision == "sha-test" and loads[0].precision == "fp32"
    files = list(artifacts.rglob("embeddings/tiny/base/ref/*.safetensors"))
    assert len(files) == 1
    tensors, meta = TensorCache(artifacts).read("embeddings/tiny/base/ref", files[0].stem)
    # 15 targets + 10 P + 10 N + 50 anchors, deduplicated
    assert tensors["mean"].shape == (85, 16) and len(meta["text_hashes"]) == 85
    with caplog.at_level(logging.INFO, logger="rbbd"):
        assert runner.run(cfg, env, ["embed"], force=["embed"]) == {"embed": "ran"}
    assert len(loads) == 1, "cache hit must not load the model"
    assert "cache hit: embeddings/tiny/base/ref/" in caplog.text


def test_embed_stage_checks_layer_count(config_dir, artifacts, monkeypatch):
    """A config layer index that disagrees with the model's decoder depth fails loudly (D-017)."""
    monkeypatch.setattr(
        "rbbd.models.loading.load_for_inference",
        lambda spec: (make_tiny_model(layers=3), FakeTokenizer()),
    )
    monkeypatch.setattr(extract, "resolve_revision", lambda mid, rev: "sha-test")
    monkeypatch.setattr("rbbd.models.loading.download", lambda spec: "/nonexistent")
    cfg = _smoke_cfg(config_dir)
    with pytest.raises(ValueError, match="config layer 2 != 3 decoder layers"):
        runner.run(cfg, env_mod.detect(), ["sentences", "embed"])


def test_embed_depends_on_train_only_for_alpha_checkpoints(config_dir):
    """Reference-only extraction depends on `sentences`; α-checkpoints add `train` (D-057)."""
    cfg = _smoke_cfg(config_dir)
    assert runner.stage_deps("embed", cfg) == ("sentences",)
    spectrum = config.load(
        config_dir(
            "run_name: b\nseed: 0\n", "run_name: t\nembed: {checkpoints: [ref, a050]}\n", "s.yaml"
        )
    )
    assert runner.stage_deps("embed", spectrum) == ("sentences", "train")


def test_compare_entries_identity_and_perturbation(config_dir, artifacts, monkeypatch):
    """Identical extractions give cosine 1 and zero B differences; added noise does not."""
    monkeypatch.setattr(
        "rbbd.models.loading.load_for_inference", lambda spec: (make_tiny_model(), FakeTokenizer())
    )
    monkeypatch.setattr(extract, "resolve_revision", lambda mid, rev: "sha-test")
    monkeypatch.setattr("rbbd.models.loading.download", lambda spec: "/nonexistent")
    cfg = _smoke_cfg(config_dir)
    env = env_mod.detect()
    runner.run(cfg, env, ["sentences", "embed"])
    tensors, union = extract.cached_ref_entry(cfg, env, "tiny")
    same = extract.compare_entries(tensors, tensors, union)
    assert same["cosine"]["mean"]["min"] > 0.9999
    assert same["bias_rr"]["max_abs_diff"] == 0.0 and same["bias_seat"]["max_abs_diff"] == 0.0
    rng = np.random.default_rng(0)
    noisy = {
        k: (v + rng.normal(scale=0.5, size=v.shape)).astype(np.float16) for k, v in tensors.items()
    }
    diff = extract.compare_entries(noisy, tensors, union)
    assert diff["cosine"]["mean"]["mean"] < 0.999 and diff["bias_rr"]["max_abs_diff"] > 0


def test_encode_keep_fp32_matches_cached_fp16(tiny):
    """B-014: `keep_fp32` returns the pre-cast pooled vectors; casting them gives the
    fp16 rows `encode` returns by default."""
    model, tok = tiny
    texts = ["a b c", "a b c d e f", "a"]
    half = encode(model, tok, texts, ExtractSettings())
    full = encode(model, tok, texts, ExtractSettings(), keep_fp32=True)
    for name in half:
        assert full[name].dtype == np.float32 and half[name].dtype == np.float16
        np.testing.assert_array_equal(full[name].astype(np.float16), half[name])
