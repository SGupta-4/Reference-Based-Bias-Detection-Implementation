"""GPU extraction checks on Kaggle 2x T4 (M2). Marked gpu; results land in
`<artifacts>/env/<session>/m2_extract_gpu.json` for upload."""

import json
from pathlib import Path

import numpy as np
import pytest

from rbbd.embed.extract import ExtractSettings, encode, extract_cached
from rbbd.models.loading import LoadSpec, base_decoder, load_for_inference
from rbbd.utils import env as env_mod
from rbbd.utils.cache import TensorCache
from rbbd.utils.guards import NonFiniteError

pytestmark = pytest.mark.gpu

QWEN = LoadSpec("Qwen/Qwen2.5-0.5B-Instruct", precision="fp16", placement="cuda:0")
QWEN32 = LoadSpec("Qwen/Qwen2.5-0.5B-Instruct", precision="fp32", placement="cuda:1")
SHORT = "Women attend community events."
LONG = [
    "Many different countries are inhabited by physically disabled people in large numbers today.",
    "The local population is partly formed by able-bodied people who moved here many years ago.",
]


def _smoke_cfg():
    from rbbd import config

    return config.load(Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml")


def _smoke_texts():
    """The 85-text smoke union, built exactly as the `sentences` stage builds it."""
    from rbbd.data import sentences

    sets = sentences.subset_sets(sentences.load_sets(), _smoke_cfg().get("sentences.subset"))
    return sentences.build_union(sets).texts


def _diffs(a, b):
    """Elementwise gaps between two [d] vectors (b measured against a)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    return {
        "max_abs": float(np.abs(a - b).max()),
        "max_rel": float((np.abs(a - b) / np.maximum(np.abs(a), 1e-6)).max()),
        "max_value": float(np.abs(a).max()),
    }


def _row_stats(a, b):
    """Per-row agreement of two [N, d] matrices: cosine and relative L2 error of b vs a."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)
    return {
        "min_cos": float(cos.min()),
        "mean_cos": float(cos.mean()),
        "max_rel_l2": float(rel.max()),
        "mean_rel_l2": float(rel.mean()),
    }


def _record(key, value):
    env = env_mod.detect()
    path = env.artifacts_root / "env" / env_mod.session_id() / "m2_extract_gpu.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    env_mod.write_json(path, data)


@pytest.fixture(scope="module")
def qwen():
    return load_for_inference(QWEN)


def test_padding_invariance_fp32():
    """DC-06 part 1 (D-061): in fp32 a sentence pooled alone == inside a padded batch,
    `assert_close(atol=1e-3, rtol=1e-4)` for mean, max and last.

    fp32 rounding is ~1000x finer than fp16, so any gap left here is a logic error
    (padding leaking into real positions or into the pooling), not kernel rounding.
    Runs on cuda:1 so it does not compete with the fp16 fixture on cuda:0.
    """
    import torch

    model, tok = load_for_inference(QWEN32)
    try:
        settings = ExtractSettings(token_budget=1 << 20)
        alone = encode(model, tok, [SHORT], settings)
        batched = encode(model, tok, [LONG[0], SHORT, LONG[1]], settings)
        pairs = {k: (alone[k][0], batched[k][1]) for k in ("mean", "max", "last")}
        _record("padding_invariance_fp32", {k: _diffs(a, b) for k, (a, b) in pairs.items()})
        for a, b in pairs.values():
            torch.testing.assert_close(
                torch.from_numpy(b).float(), torch.from_numpy(a).float(), atol=1e-3, rtol=1e-4
            )
    finally:
        del model
        torch.cuda.empty_cache()


def test_padding_cosine_fp16_smoke(qwen):
    """DC-06 part 2 (D-061): every smoke-union sentence (85 texts) pooled alone vs inside
    its production batch (the smoke config's `embed` settings), in fp16 on T4.

    Pass: per-sentence cosine(alone, batched) >= 0.9999 for mean, max and last.
    Recorded, not asserted: the same sentences in fp32 (alone), so the padding error
    can be set against fp16's own error vs fp32.
    """
    import torch

    model, tok = qwen
    texts = _smoke_texts()
    settings = ExtractSettings.from_config(_smoke_cfg().get("embed", {}))
    batched = encode(model, tok, texts, settings)
    alone_rows = [encode(model, tok, [t], settings) for t in texts]
    alone = {k: np.concatenate([r[k] for r in alone_rows]) for k in ("mean", "max", "last")}

    model32, tok32 = load_for_inference(QWEN32)
    try:
        ref32 = encode(model32, tok32, texts, ExtractSettings(token_budget=1 << 20, max_batch=1))
    finally:
        del model32
        torch.cuda.empty_cache()

    report = {"n_texts": len(texts)}
    for k in ("mean", "max", "last"):
        report[k] = {
            "padding_fp16": _row_stats(alone[k], batched[k]),
            "fp16_alone_vs_fp32": _row_stats(ref32[k], alone[k]),
            "fp16_batched_vs_fp32": _row_stats(ref32[k], batched[k]),
        }
    _record("padding_cosine_fp16_smoke", report)
    for k in ("mean", "max", "last"):
        assert report[k]["padding_fp16"]["min_cos"] >= 0.9999, (k, report[k]["padding_fp16"])


def test_guard_active_on_real_model(qwen, tmp_path):
    """The guard runs on every batch of a real fp16 forward: an injected Inf raises and
    nothing is cached (DC-09 on GPU)."""
    model, tok = qwen

    def poison(_m, _i, output):
        out = output[0] if isinstance(output, tuple) else output
        out[0, 0, 0] = float("inf")
        return output

    handle = base_decoder(model).layers[-1].register_forward_hook(poison)
    try:
        with pytest.raises(NonFiniteError, match="batch=0"):
            extract_cached(
                TensorCache(tmp_path), "embeddings/q", {"k": 1}, [SHORT, *LONG],
                ExtractSettings(), lambda: (model, tok), context={"ckpt": "ref"},
            )  # fmt: skip
    finally:
        handle.remove()
    assert not list(tmp_path.rglob("*.safetensors"))
    _record("guard_on_gpu", "raised NonFiniteError, nothing cached")
