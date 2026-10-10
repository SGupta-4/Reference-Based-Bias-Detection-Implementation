"""Merge exactness (D-007, D-008, D-066). DC-07."""

import pytest
import torch

from rbbd.models import adapters as ad
from rbbd.models import interpolate as ip
from tests.conftest import make_tiny_model, random_adapter

ALPHAS = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@pytest.fixture
def endpoints(tmp_path):
    u = ad.read_adapter(random_adapter(tmp_path, "u", 1))
    h = ad.read_adapter(random_adapter(tmp_path, "h", 2))
    return u, h


def _ba(adapter, prefix):
    """B @ A in float64 for one module."""
    t = adapter.tensors
    return t[prefix + ".lora_B.weight"].double() @ t[prefix + ".lora_A.weight"].double()


def _same_tensors(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_lora_alpha1_equals_unharmful(endpoints, tmp_path):
    """α=1 serves the unharmful adapter itself, bit-identical, also after a disk round trip."""
    u, h = endpoints
    a100 = ad.adapter_for_alpha(u, h, 1.0)
    assert _same_tensors(a100.tensors, u.tensors) and a100.config == u.config
    back = ad.read_adapter(ad.write_adapter(a100, tmp_path / "a100"))
    assert _same_tensors(back.tensors, u.tensors)
    dw, dw_u = ad.delta_w(back), ad.delta_w(u)
    assert all(torch.equal(dw[p], dw_u[p]) for p in dw_u)


def test_lora_alpha0_equals_harmful(endpoints):
    """α=0 serves the harmful adapter itself, bit-identical."""
    u, h = endpoints
    a000 = ad.adapter_for_alpha(u, h, 0.0)
    assert _same_tensors(a000.tensors, h.tensors) and a000.config == h.config
    dw, dw_h = ad.delta_w(a000), ad.delta_w(h)
    assert all(torch.equal(dw[p], dw_h[p]) for p in dw_h)


def test_lora_combined_delta_w_linear(endpoints):
    """Rank-2r concatenation: ΔW(α) == α·ΔW_u + (1−α)·ΔW_h for every α, endpoints included.

    Both sides are evaluated in float64 from the stored fp32 tensors, so the comparison
    measures the construction, not matmul rounding: rtol 1e-6 with atol 1e-6·max|ΔW|
    (elements near zero have no meaningful relative error).
    """
    u, h = endpoints
    s_u, s_h = ad.scaling(u.config), ad.scaling(h.config)
    for alpha in ALPHAS:
        c = ad.combine(u, h, alpha)
        assert c.config["r"] == 8 and c.config["lora_alpha"] == 8 and ad.scaling(c.config) == 1.0
        assert c.meta["combined"]["alpha"] == alpha
        for p in u.prefixes():
            got = _ba(c, p)
            ref = alpha * s_u * _ba(u, p) + (1 - alpha) * s_h * _ba(h, p)
            torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6 * float(ref.abs().max()))


def test_combined_adapter_loads_in_peft(endpoints, tmp_path):
    """The written rank-2r adapter is a valid PEFT adapter: merged weights == W0 + ΔW(α)."""
    from peft import PeftModel

    u, h = endpoints
    alpha = 0.3
    path = ad.write_adapter(ad.combine(u, h, alpha), tmp_path / "a030")
    base = make_tiny_model()
    w0 = {k: v.clone() for k, v in base.state_dict().items()}
    merged = PeftModel.from_pretrained(base, str(path)).merge_and_unload()
    dw_u, dw_h = ad.delta_w(u), ad.delta_w(h)
    for p in u.prefixes():
        key = p.removeprefix("base_model.model.") + ".weight"
        expected = w0[key] + alpha * dw_u[p] + (1 - alpha) * dw_h[p]
        torch.testing.assert_close(merged.state_dict()[key], expected, rtol=1e-5, atol=1e-6)


def test_combine_rejects_mismatched_endpoints(endpoints):
    u, h = endpoints
    bad = ad.LoraAdapter(config={**h.config, "r": 8}, tensors=h.tensors)
    with pytest.raises(ad.AdapterError, match="'r'"):
        ad.combine(u, bad, 0.5)
    with pytest.raises(ad.AdapterError, match="DoRA"):
        ad._check_combinable({**u.config, "use_dora": True}, "x")
    with pytest.raises(ValueError):
        ad.adapter_for_alpha(u, h, 1.5)


def test_full_alpha_endpoints_exact():
    """(1−α)W_h + αW_u reproduces W_u at α=1 and W_h at α=0 (torch.equal), on a real model."""
    w_h = {k: v.float().clone() for k, v in make_tiny_model(seed=1).state_dict().items()}
    w_u = {k: v.float().clone() for k, v in make_tiny_model(seed=2).state_dict().items()}
    ip.check_endpoints(w_h, w_u)
    model = make_tiny_model(seed=3)
    for alpha, expected in ((1.0, w_u), (0.0, w_h)):
        ip.apply_alpha(model, w_h, w_u, alpha)
        assert _same_tensors(model.state_dict(), expected)
    ip.apply_alpha(model, w_h, w_u, 0.3)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, 0.7 * w_h[k] + 0.3 * w_u[k])


def test_full_fp16_endpoints_roundtrip(tmp_path):
    """Endpoints stored as fp16 safetensors load back as fp32 and stay exact at α ∈ {0, 1}."""
    from safetensors.torch import save_file

    w = {
        k: v.half().contiguous()
        for k, v in make_tiny_model(seed=1).state_dict().items()
        if k != "lm_head.weight"
    }
    save_file(w, str(tmp_path / "model.safetensors"))
    loaded = ip.load_endpoint(tmp_path)
    assert all(v.dtype == torch.float32 for v in loaded.values())
    norm = loaded["model.norm.weight"]
    assert torch.equal(ip.interpolate(norm, norm * 2, 0.0), norm)


def test_materialize_refuses_persistent_paths(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    with pytest.raises(ValueError, match="must not be written"):
        ip.assert_ephemeral(root / "merged" / "a050", root)
    with pytest.raises(ValueError, match="must not be written"):
        ip.assert_ephemeral("/kaggle/working/x", root)
    eph = tmp_path / "eph" / "a050"
    assert ip.assert_ephemeral(eph, root) == eph.resolve()


def test_combine_accepts_target_modules_in_any_order(endpoints):
    """B-016: PEFT writes target_modules from a set, so endpoints saved by two processes
    list them in different orders; that is the same adapter layout, not a mismatch."""
    u, h = endpoints
    shuffled = ad.LoraAdapter(
        config={**h.config, "target_modules": list(reversed(sorted(h.config["target_modules"])))},
        tensors=h.tensors,
    )
    c = ad.combine(u, shuffled, 0.5)
    assert c.config["target_modules"] == sorted(TARGETS)


def test_apply_alpha_with_tied_embeddings(tmp_path):
    """Llama-3.2-1B ties lm_head to embed_tokens; save_pretrained stores the tensor once.
    Endpoints without lm_head.weight still apply, and the alias follows (D-072)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    def tied(seed):
        torch.manual_seed(seed)
        cfg = LlamaConfig(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=True,
        )
        return LlamaForCausalLM(cfg).eval()

    for name, seed in (("h", 1), ("u", 2)):
        tied(seed).save_pretrained(str(tmp_path / name), safe_serialization=True)
    w_h, w_u = ip.load_endpoint(tmp_path / "h"), ip.load_endpoint(tmp_path / "u")
    assert "lm_head.weight" not in w_h
    model = tied(3)
    ip.apply_alpha(model, w_h, w_u, 1.0)
    sd = model.state_dict()
    assert torch.equal(sd["model.embed_tokens.weight"], w_u["model.embed_tokens.weight"])
    assert torch.equal(sd["lm_head.weight"], w_u["model.embed_tokens.weight"])
    w_h.pop("model.norm.weight")
    with pytest.raises(ip.EndpointMismatch, match="model.norm.weight"):
        ip.apply_alpha(model, w_h, w_u, 0.5)
