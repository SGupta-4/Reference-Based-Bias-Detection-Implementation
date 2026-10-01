"""GPU training checks on Kaggle 2x T4 (M3a). Marked gpu.

`test_real_adapter_endpoints` reads the endpoint adapters that the smoke `train` stage wrote
(`run --config configs/smoke.yaml --stages ftdata,train` must have run first).
`test_resume_after_kill` trains Qwen2.5-0.5B LoRA on synthetic word rows (no WildGuardMix
text). Results land in `<artifacts>/env/<session>/m3a_train_gpu.json`.
"""

import json
from pathlib import Path

import pytest
import torch

from rbbd import config
from rbbd.finetune import sft
from rbbd.models import adapters as ad
from rbbd.utils import env as env_mod
from rbbd.utils.guards import assert_finite

pytestmark = pytest.mark.gpu

SMOKE = Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"


def _record(key, value):
    env = env_mod.detect()
    path = env.artifacts_root / "env" / env_mod.session_id() / "m3a_train_gpu.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    env_mod.write_json(path, data)


def _smoke_finals():
    """{split: final adapter dir} of the smoke config's seed-0 LoRA endpoints."""
    root = env_mod.detect().artifacts_root
    found = {}
    for split in sft.SPLITS:
        dirs = sorted(root.glob(f"train/qwen2.5-0.5b-it/lora/{split}/seed0/*/done.json"))
        if len(dirs) != 1:
            pytest.fail(
                f"expected one finished smoke {split} job, found {len(dirs)}: run the train stage"
            )
        found[split] = dirs[0].parent / "final"
    return found


def test_real_adapter_endpoints():
    """DC-07 on real Tier 0 adapters: α=1/α=0 are the trained adapters bit for bit; the
    rank-2r combination is linear in ΔW (float64 check, rtol 1e-6); both endpoints moved
    off zero; a combined adapter loads in PEFT on the fp16 base and gives a finite forward."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    finals = _smoke_finals()
    u, h = ad.read_adapter(finals["unharmful"]), ad.read_adapter(finals["harmful"])
    a100, a000 = ad.adapter_for_alpha(u, h, 1.0), ad.adapter_for_alpha(u, h, 0.0)
    assert a100 is u and a000 is h
    report = {"r": u.config["r"], "modules": len(u.prefixes()), "max_rel_err": {}}
    for name, adapter in (("u", u), ("h", h)):
        report[f"max_abs_dw_{name}"] = max(
            float(d.abs().max()) for d in ad.delta_w(adapter).values()
        )
        assert report[f"max_abs_dw_{name}"] > 0, f"{name} adapter never trained (B still zero)"
    s_u, s_h = ad.scaling(u.config), ad.scaling(h.config)
    for alpha in (0.9, 0.5, 0.1):
        c = ad.combine(u, h, alpha)
        worst = 0.0
        for p in u.prefixes():
            got = (
                c.tensors[p + ".lora_B.weight"].double() @ c.tensors[p + ".lora_A.weight"].double()
            )
            ref = alpha * s_u * (
                u.tensors[p + ".lora_B.weight"].double() @ u.tensors[p + ".lora_A.weight"].double()
            ) + (1 - alpha) * s_h * (
                h.tensors[p + ".lora_B.weight"].double() @ h.tensors[p + ".lora_A.weight"].double()
            )
            torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6 * float(ref.abs().max()))
            worst = max(worst, float((got - ref).abs().max() / ref.abs().max()))
        report["max_rel_err"][str(alpha)] = worst

    path = ad.write_adapter(ad.combine(u, h, 0.5), env_mod.ephemeral_dir() / "rbbd_m3a_a050")
    tok = AutoTokenizer.from_pretrained(QWEN)
    base = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, str(path)).eval()
    batch = tok(["Women attend community events."], return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out = model(**batch, output_hidden_states=True)
    assert_finite(out.hidden_states[-1], "m3a.a050.hidden", ckpt="a050")
    report["a050_forward"] = "finite"
    _record("real_adapter_endpoints", report)


def test_resume_after_kill(tmp_path):
    """DC-13 on the T4: a run killed right after its step-3 checkpoint resumes and finishes at
    the same global step as an uninterrupted run. Weight differences are recorded only
    (GPU kernels are not bit-deterministic)."""
    import numpy as np

    from rbbd.models.loading import load_tokenizer

    rng = np.random.default_rng(0)
    rows = [
        {
            "prompt": " ".join(f"w{int(i)}" for i in rng.integers(0, 50, 8)),
            "response": " ".join(f"w{int(i)}" for i in rng.integers(0, 50, 6)),
        }
        for _ in range(48)
    ]
    tok = load_tokenizer(QWEN, None)
    spec = sft.TrainSpec(
        model_id=QWEN,
        slug="qwen-dc13",
        regime="lora",
        split="unharmful",
        seed=0,
        data_key="synthetic",
        epochs=1,
        per_device_batch=4,
        grad_accum=2,
        max_length=64,
    )

    ref = sft.train_one(spec, rows, tok, sft.load_model(spec), tmp_path / "ref", save_minutes=1e6)
    with pytest.raises(sft.SimulatedKill):
        sft.train_one(
            spec,
            rows,
            tok,
            sft.load_model(spec),
            tmp_path / "job",
            save_steps=1,
            extra_callbacks=[sft.KillAfterSave(3)],
        )
    torch.cuda.empty_cache()
    done = sft.train_one(spec, rows, tok, sft.load_model(spec), tmp_path / "job", save_minutes=1e6)
    a = ad.read_adapter(tmp_path / "ref" / "final").tensors
    b = ad.read_adapter(tmp_path / "job" / "final").tensors
    max_diff = max(float((a[k] - b[k]).abs().max()) for k in a)
    _record(
        "resume_after_kill",
        {
            "ref_global_step": ref["global_step"],
            "resumed_global_step": done["global_step"],
            "resumed_from": done["resumed_from"],
            "max_abs_adapter_diff": max_diff,
            "scaler_skipped_ref": ref["scaler_skipped_steps"],
        },
    )
    assert done["resumed_from"] == "checkpoint-3"
    assert done["global_step"] == ref["global_step"] == 6


def test_smoke_config_has_train_section():
    """Guard against running the notebook with a config that never trains (CPU-cheap)."""
    cfg = config.load(SMOKE)
    assert cfg.get("train.regimes") == ["lora"] and cfg.get("ftdata.n_per_split") == 64
