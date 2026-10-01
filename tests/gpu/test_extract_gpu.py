"""GPU extraction checks on Kaggle 2x T4 (M2). Marked gpu; results land in
`<artifacts>/env/<session>/m2_extract_gpu.json` for upload."""

import json

import pytest

from rbbd.embed.extract import ExtractSettings, encode, extract_cached
from rbbd.models.loading import LoadSpec, base_decoder, load_for_inference
from rbbd.utils import env as env_mod
from rbbd.utils.cache import TensorCache
from rbbd.utils.guards import NonFiniteError

pytestmark = pytest.mark.gpu

QWEN = LoadSpec("Qwen/Qwen2.5-0.5B-Instruct", precision="fp16", placement="cuda:0")
SHORT = "Women attend community events."
LONG = [
    "Many different countries are inhabited by physically disabled people in large numbers today.",
    "The local population is partly formed by able-bodied people who moved here many years ago.",
]


def _record(key, value):
    env = env_mod.detect()
    path = env.artifacts_root / "env" / env_mod.session_id() / "m2_extract_gpu.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    env_mod.write_json(path, data)


@pytest.fixture(scope="module")
def qwen():
    return load_for_inference(QWEN)


def test_padding_invariance_fp16(qwen):
    """DC-06 in fp16 on T4: a sentence pooled alone == inside a padded batch
    (atol 1e-3, rtol 1e-3; D-058) for mean, max and last."""
    import torch

    model, tok = qwen
    settings = ExtractSettings(token_budget=1 << 20)
    alone = encode(model, tok, [SHORT], settings)
    batched = encode(model, tok, [LONG[0], SHORT, LONG[1]], settings)
    diffs = {}
    for kind in ("mean", "max", "last"):
        a = torch.from_numpy(alone[kind][0]).float()
        b = torch.from_numpy(batched[kind][1]).float()
        diffs[kind] = {
            "max_abs": float((a - b).abs().max()),
            "max_rel": float(((a - b).abs() / a.abs().clamp_min(1e-6)).max()),
            "max_value": float(a.abs().max()),
        }
    _record("padding_invariance_fp16", diffs)
    for kind in ("mean", "max", "last"):
        torch.testing.assert_close(
            torch.from_numpy(batched[kind][1]).float(), torch.from_numpy(alone[kind][0]).float(),
            atol=1e-3, rtol=1e-3,
        )  # fmt: skip


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
