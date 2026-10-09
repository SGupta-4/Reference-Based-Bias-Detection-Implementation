"""Spectrum slugs, α activation on one loaded base, and the M4 `embed` sweep (D-004, D-007,
D-008, D-066, D-081) on the tiny CPU model. No downloads: endpoints are random adapters
and perturbed copies of the tiny model."""

import json
import logging

import numpy as np
import pytest
import torch

from rbbd import config, runner
from rbbd.embed import extract
from rbbd.models import adapters as ad
from rbbd.models import spectrum as spm
from rbbd.models.loading import base_decoder
from rbbd.utils import env as env_mod
from rbbd.utils import manifest as mf
from tests.conftest import STUBS, SWEEP_BASE, SWEEP_TIER, FakeTokenizer, make_tiny_model

IDS = torch.tensor([[1, 5, 9, 17, 33, 2]])


def _hidden(model):
    with torch.no_grad():
        return base_decoder(model)(input_ids=IDS).last_hidden_state


def test_slugs_roundtrip():
    assert spm.CHECKPOINTS == ("ref", "a100", "a090", "a070", "a050", "a030", "a010", "a000")
    assert all(spm.slug_for(spm.alpha_of(s)) == s for s in spm.CHECKPOINTS[1:])
    assert spm.alpha_of("ref") is None
    with pytest.raises(ValueError):
        spm.alpha_of("a5")


def test_lora_activation_matches_merged_weights(lora_spectrum):
    """α=0.5 through PEFT equals W0 + 0.5·ΔW_u + 0.5·ΔW_h written into the weights; only
    one adapter is resident after switching; `plain()` restores the exact base."""
    act = spm.Activator(lambda: (make_tiny_model(), FakeTokenizer()))
    ref = _hidden(act.plain()[0])
    act.get(lora_spectrum, "a100")
    model, _ = act.get(lora_spectrum, "a050")
    got = _hidden(model)
    assert list(model.peft_config) == ["lora-s0-a050"]

    merged = make_tiny_model()
    u = ad.read_adapter(lora_spectrum.finals["unharmful"])
    h = ad.read_adapter(lora_spectrum.finals["harmful"])
    params = dict(merged.named_parameters())
    with torch.no_grad():
        for prefix, dw in ad.delta_w(ad.combine(u, h, 0.5)).items():
            params[prefix.removeprefix("base_model.model.") + ".weight"].add_(dw)
    torch.testing.assert_close(got, _hidden(merged), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(got, ref)
    torch.testing.assert_close(_hidden(act.plain()[0]), ref, atol=0, rtol=0)


def test_lora_endpoints_are_the_trained_adapters(lora_spectrum):
    """a000 through the activator equals loading the harmful adapter directly (D-066)."""
    from peft import PeftModel

    act = spm.Activator(lambda: (make_tiny_model(), FakeTokenizer()))
    got = _hidden(act.get(lora_spectrum, "a000")[0])
    direct = PeftModel.from_pretrained(make_tiny_model(), str(lora_spectrum.finals["harmful"]))
    torch.testing.assert_close(got, _hidden(direct.eval()), atol=0, rtol=0)


def test_full_activation_interpolates_and_blocks_ref(full_spectrum):
    """Full FT: a100 is W_u exactly; a050 is the fp32 mix; the reference cannot be
    requested once weights were overwritten."""
    from rbbd.models import interpolate as ip

    act = spm.Activator(lambda: (make_tiny_model(), FakeTokenizer()))
    model, _ = act.get(full_spectrum, "a100")
    w_u = ip.load_endpoint(full_spectrum.finals["unharmful"])
    assert all(torch.equal(p, w_u[n]) for n, p in model.state_dict().items() if n in w_u)
    model, _ = act.get(full_spectrum, "a050")
    w_h = ip.load_endpoint(full_spectrum.finals["harmful"])
    name = "model.layers.0.self_attn.q_proj.weight"
    torch.testing.assert_close(model.state_dict()[name], 0.5 * w_h[name] + 0.5 * w_u[name])
    with pytest.raises(RuntimeError, match="reference requested after full-FT"):
        act.plain()


def test_assert_same_but_checkpoint():
    a = {"model": {"precision": "fp16"}, "layer": 2, "checkpoint": {"slug": "ref"}}
    extract.assert_same_but_checkpoint(a, {**a, "checkpoint": {"slug": "a050"}})
    with pytest.raises(ValueError, match=r"\['model'\]"):
        extract.assert_same_but_checkpoint(a, {**a, "model": {"precision": "fp32"}})


def test_embed_sweep_writes_every_checkpoint_and_hits_cache(sweep, artifacts, caplog):
    """15 entries (ref + 2 spectra × 7 α) from one base load; keys differ only in
    `checkpoint`; a000 differs from a100; a second run loads nothing (DC-10)."""
    cfg, env, loads = sweep
    out = runner.run(cfg, env, ["sentences", "ftdata", "train", "embed"], stage_fns=STUBS)
    assert out["embed"] == "ran" and len(loads) == 1
    keys = runner.stage_run_keys(cfg)
    m = mf.read(mf.manifest_path(artifacts, "embed", keys["embed"]))
    index_rel = next(p for p in m.output_hashes if p.endswith("index.json"))
    index = json.loads((artifacts / index_rel).read_text())
    assert [(r["regime"], r["ckpt"]) for r in index][:3] == [
        ("base", "ref"), ("lora", "a100"), ("lora", "a090")]  # fmt: skip
    assert len(index) == 15 and len({r["key"] for r in index}) == 15
    from rbbd.utils.cache import TensorCache

    cache = TensorCache(artifacts)

    def read(row):
        path = artifacts / row["path"]
        return cache.read(str(path.parent.relative_to(artifacts)), path.stem)

    sidecars = {(r["regime"], r["ckpt"]): read(r) for r in index}
    ref_fields = sidecars[("base", "ref")][1]["fields"]
    for _, meta in sidecars.values():
        extract.assert_same_but_checkpoint(ref_fields, meta["fields"])
    for regime in ("lora", "full"):
        a, b = sidecars[(regime, "a100")][0]["mean"], sidecars[(regime, "a000")][0]["mean"]
        assert not np.allclose(a, b)
    assert sidecars[("full", "a050")][1]["fields"]["checkpoint"]["merge"] == "interp-fp32"
    assert sidecars[("lora", "a050")][1]["fields"]["checkpoint"]["merge"] == "concat-rank2r"

    with caplog.at_level(logging.INFO, logger="rbbd"):
        again = runner.run(cfg, env, ["embed"], force=["embed"], stage_fns=STUBS)
    assert again == {"embed": "ran"} and len(loads) == 1, "a full cache hit must not load"
    assert caplog.text.count("cache hit: embeddings/tiny/") == 15


def test_only_ckpt_leaves_manifest_incomplete_then_resumes(sweep, artifacts):
    """`--only-ckpt a050` extracts that slug only and never completes the manifest; the
    following full run resumes and reuses the a050 entries (DC-12 path, D-081)."""
    cfg, env, loads = sweep
    runner.run(cfg, env, ["sentences", "ftdata", "train"], stage_fns=STUBS)
    out = runner.run(cfg, env, ["embed"], stage_fns=STUBS, only_ckpt="a050")
    assert out == {"embed": "partial"}
    keys = runner.stage_run_keys(cfg)
    m = mf.read(mf.manifest_path(artifacts, "embed", keys["embed"]))
    assert not m.complete and sorted(m.progress["done"]) == ["tiny/full-s0/a050",
                                                             "tiny/lora-s0/a050"]  # fmt: skip
    assert runner.run(cfg, env, ["embed"], stage_fns=STUBS) == {"embed": "resumed"}
    assert mf.read(mf.manifest_path(artifacts, "embed", keys["embed"])).complete
    timing = json.loads(next(artifacts.rglob("embed/*/timing.json")).read_text())
    ckpts = timing["models"]["tiny"]["checkpoints"]
    assert ckpts["tiny/lora-s0/a050"]["cache_hit"] and not ckpts["tiny/lora-s0/a030"]["cache_hit"]


def test_embed_rejects_unknown_checkpoint(config_dir, artifacts):
    cfg = config.load(config_dir(SWEEP_BASE.replace("a000]", "a000, a055]"), SWEEP_TIER))
    runner.run(cfg, env_mod.detect(), ["sentences", "ftdata", "train"], stage_fns=STUBS)
    with pytest.raises(ValueError, match="unknown checkpoints"):
        runner.run(cfg, env_mod.detect(), ["embed"], stage_fns=STUBS)
