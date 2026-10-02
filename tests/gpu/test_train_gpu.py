"""GPU training checks on Kaggle 2x T4 (M3a). Marked gpu.

`test_real_adapter_endpoints` reads the endpoint adapters that the smoke `train` stage wrote
(`run --config configs/smoke.yaml --stages ftdata,train` must have run first).
`test_resume_after_kill` trains Qwen2.5-0.5B LoRA on synthetic word rows (no WildGuardMix
text) in child processes that each see one GPU (B-017).
Results land in `<artifacts>/env/<session>/m3a_train_gpu.json`.
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
    """{split: final adapter dir} of the smoke config's seed-0 LoRA jobs, located by the
    config's current train_key (via the ftdata manifest), not by globbing old job dirs."""
    from rbbd import runner
    from rbbd.utils import manifest as mf

    cfg = config.load(SMOKE)
    root = env_mod.detect().artifacts_root
    ft = mf.read(mf.manifest_path(root, "ftdata", runner.stage_run_keys(cfg)["ftdata"]))
    if ft is None or not ft.complete:
        pytest.fail("no complete smoke ftdata manifest: run the ftdata and train stages first")
    found = {}
    for job in sft.plan_jobs(cfg, root, list(ft.output_hashes)):
        if not (job.out_dir / "done.json").exists():
            pytest.fail(f"smoke job not finished: {job.out_dir}")
        found[job.split] = job.out_dir / "final"
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


DC13_ROWS, DC13_STEPS = 48, 6  # 48 rows / (batch 4 x grad-acc 2) x 1 epoch on ONE GPU


def _dc13_spec():
    return sft.TrainSpec(model_id=QWEN, slug="qwen-dc13", regime="lora", split="unharmful",
                         seed=0, data_key="synthetic", epochs=1, per_device_batch=4,
                         grad_accum=2, max_length=64)  # fmt: skip


def _dc13_child(phase, work):
    """Child-process body for `test_resume_after_kill` (one visible GPU, as in the stage).

    phase "ref": uninterrupted run into work/ref. "kill": checkpoint every step and die
    right after checkpoint-3 (exit code 3). "resume": continue work/job to the end.
    """
    import sys

    import numpy as np

    from rbbd.models.loading import load_tokenizer

    rng = np.random.default_rng(0)
    rows = [{"prompt": " ".join(f"w{int(i)}" for i in rng.integers(0, 50, 8)),
             "response": " ".join(f"w{int(i)}" for i in rng.integers(0, 50, 6))}
            for _ in range(DC13_ROWS)]  # fmt: skip
    spec, tok = _dc13_spec(), load_tokenizer(QWEN, None)
    work = Path(work)
    if phase == "ref":
        sft.train_one(spec, rows, tok, sft.load_model(spec), work / "ref", save_minutes=1e6)
    elif phase == "kill":
        try:
            sft.train_one(spec, rows, tok, sft.load_model(spec), work / "job", save_steps=1,
                          extra_callbacks=[sft.KillAfterSave(3)])  # fmt: skip
        except sft.SimulatedKill:
            sys.exit(3)
        sys.exit(1)  # the kill hook never fired
    else:
        sft.train_one(spec, rows, tok, sft.load_model(spec), work / "job", save_minutes=1e6)


def _run_child(phase, work):
    import os
    import subprocess
    import sys

    code = (
        f"from tests.gpu.test_train_gpu import _dc13_child; _dc13_child({phase!r}, {str(work)!r})"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    root = Path(__file__).resolve().parents[2]
    return subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                          capture_output=True, text=True)  # fmt: skip


def test_resume_after_kill(tmp_path):
    """DC-13 on the T4: a process that dies right after its step-3 checkpoint is resumed by a
    new process and finishes at the same global step as an uninterrupted run (6). Each run
    is a child process with one visible GPU, as in the train stage. Adapter differences are
    recorded only, because GPU kernels are not bit-deterministic."""
    ref = _run_child("ref", tmp_path)
    assert ref.returncode == 0, ref.stderr[-3000:]
    kill = _run_child("kill", tmp_path)
    assert kill.returncode == 3, kill.stderr[-3000:]
    assert not (tmp_path / "job" / "done.json").exists()
    resume = _run_child("resume", tmp_path)
    assert resume.returncode == 0, resume.stderr[-3000:]
    ref_done = json.loads((tmp_path / "ref" / "done.json").read_text())
    done = json.loads((tmp_path / "job" / "done.json").read_text())
    a = ad.read_adapter(tmp_path / "ref" / "final").tensors
    b = ad.read_adapter(tmp_path / "job" / "final").tensors
    _record("resume_after_kill", {
        "ref_global_step": ref_done["global_step"], "resumed_global_step": done["global_step"],
        "resumed_from": done["resumed_from"],
        "max_abs_adapter_diff": max(float((a[k] - b[k]).abs().max()) for k in a),
        "max_abs_adapter_value": max(float(a[k].abs().max()) for k in a),
        "scaler_skipped_ref": ref_done["scaler_skipped_steps"],
    })  # fmt: skip
    assert done["resumed_from"] == "checkpoint-3"
    assert done["global_step"] == ref_done["global_step"] == DC13_STEPS


def test_train_one_refuses_two_visible_gpus(tmp_path):
    """B-017: in this 2-GPU pytest process, train_one must refuse rather than fall into
    DataParallel and double the effective batch."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two visible GPUs")
    with pytest.raises(RuntimeError, match="exactly one visible"):
        sft.train_one(_dc13_spec(), [{"prompt": "a", "response": "b"}], None, None, tmp_path)


def test_smoke_config_has_train_section():
    """Guard against running the notebook with a config that never trains (CPU-cheap)."""
    cfg = config.load(SMOKE)
    assert cfg.get("train.regimes") == ["lora"] and cfg.get("ftdata.n_per_split") == 64


def _finals_for(cfg_path, regime, seed=0, overrides=()):
    """{split: final dir} for one regime/seed of a config (+ `--set` overrides), via its
    ftdata manifest."""
    from rbbd import runner
    from rbbd.utils import manifest as mf

    cfg = config.load(cfg_path, overrides)
    root = env_mod.detect().artifacts_root
    ft = mf.read(mf.manifest_path(root, "ftdata", runner.stage_run_keys(cfg)["ftdata"]))
    if ft is None or not ft.complete:
        pytest.fail(f"no complete ftdata manifest for {cfg_path}")
    found = {}
    for job in sft.plan_jobs(cfg, root, list(ft.output_hashes)):
        if job.regime == regime and job.seed == seed:
            if not (job.out_dir / "done.json").exists():
                pytest.fail(f"job not finished: {job.out_dir}")
            found[job.split] = job.out_dir / "final"
    return cfg, found


def _m3b_config():
    import os

    path = os.environ.get("RBBD_M3B_CONFIG")
    if not path:
        pytest.skip("set RBBD_M3B_CONFIG to the Tier 2 config trained in this session")
    return Path(path)


def test_real_full_endpoints():
    """DC-07 on real Tier 2 full-FT endpoints (D-008): on the GPU model, W(α=1) equals the
    unharmful endpoint and W(α=0) the harmful one exactly (`torch.equal`, fp32), tied
    aliases included; W(0.5) gives a finite forward."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rbbd.models import interpolate as ip

    cfg, finals = _finals_for(_m3b_config(), "full")
    w_u, w_h = ip.load_endpoint(finals["unharmful"]), ip.load_endpoint(finals["harmful"])
    ip.check_endpoints(w_h, w_u)
    model = AutoModelForCausalLM.from_pretrained(
        str(finals["unharmful"]), dtype=torch.float32, device_map={"": 0}
    ).eval()
    report = {"n_tensors": len(w_u)}
    for alpha, expected in ((1.0, w_u), (0.0, w_h)):
        ip.apply_alpha(model, w_h, w_u, alpha)
        sd = model.state_dict()
        assert all(torch.equal(sd[k].cpu(), expected[k]) for k in expected), alpha
    diff = max(float((w_u[k] - w_h[k]).abs().max()) for k in w_u)
    report["max_abs_endpoint_diff"] = diff
    assert diff > 0, "the two endpoints are identical: training did not move the weights"
    ip.apply_alpha(model, w_h, w_u, 0.5)
    tok = AutoTokenizer.from_pretrained(str(finals["unharmful"]))
    batch = tok(["Women attend community events."], return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out = model(**batch, output_hidden_states=True)
    assert_finite(out.hidden_states[-1], "m3b.full.a050.hidden", ckpt="a050")
    report["a050_forward"] = "finite"
    _record(f"real_full_endpoints_{cfg['run_name']}", report)


def test_real_tier2_lora_endpoints():
    """DC-07 on real Tier 2 LoRA endpoints: α ∈ {1, 0} are the trained adapters; the rank-2r
    combination is linear in ΔW (float64, rtol 1e-6)."""
    cfg, finals = _finals_for(_m3b_config(), "lora")
    u, h = ad.read_adapter(finals["unharmful"]), ad.read_adapter(finals["harmful"])
    assert ad.adapter_for_alpha(u, h, 1.0) is u and ad.adapter_for_alpha(u, h, 0.0) is h
    s_u, s_h = ad.scaling(u.config), ad.scaling(h.config)
    worst = 0.0
    for alpha in (0.9, 0.5, 0.1):
        c = ad.combine(u, h, alpha)
        for p in u.prefixes():

            def ba(a, p=p):
                return (
                    a.tensors[p + ".lora_B.weight"].double()
                    @ a.tensors[p + ".lora_A.weight"].double()
                )

            got, ref = ba(c), alpha * s_u * ba(u) + (1 - alpha) * s_h * ba(h)
            torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6 * float(ref.abs().max()))
            worst = max(worst, float((got - ref).abs().max() / ref.abs().max()))
    _record(
        f"real_lora_endpoints_{cfg['run_name']}",
        {"modules": len(u.prefixes()), "max_rel_err": worst},
    )


def test_real_tier1_qlora_endpoints():
    """DC-07 on real Tier 1 QLoRA endpoints (M3c): α ∈ {1, 0} are the trained adapters; the
    rank-2r combination is linear in ΔW (float64, rtol 1e-6); the α=0.5 adapter loads on
    the model's inference base (precision and placement from the config, as M4 will use
    it) and gives a finite forward. Select with RBBD_M3C_CONFIG and, for runtime
    decisions, RBBD_M3C_SET ("key=value;key=value")."""
    import os

    from peft import PeftModel

    from rbbd.models.loading import LoadSpec, base_decoder, load_for_inference

    path = os.environ.get("RBBD_M3C_CONFIG")
    if not path:
        pytest.skip("set RBBD_M3C_CONFIG to the Tier 1 config trained in this session")
    overrides = [x for x in os.environ.get("RBBD_M3C_SET", "").split(";") if x]
    cfg, finals = _finals_for(Path(path), "qlora", overrides=overrides)
    u, h = ad.read_adapter(finals["unharmful"]), ad.read_adapter(finals["harmful"])
    assert ad.adapter_for_alpha(u, h, 1.0) is u and ad.adapter_for_alpha(u, h, 0.0) is h
    s_u, s_h = ad.scaling(u.config), ad.scaling(h.config)
    worst = 0.0
    for alpha in (0.9, 0.5, 0.1):
        c = ad.combine(u, h, alpha)
        for p in u.prefixes():

            def ba(a, p=p):
                return (
                    a.tensors[p + ".lora_B.weight"].double()
                    @ a.tensors[p + ".lora_A.weight"].double()
                )

            got, ref = ba(c), alpha * s_u * ba(u) + (1 - alpha) * s_h * ba(h)
            torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6 * float(ref.abs().max()))
            worst = max(worst, float((got - ref).abs().max() / ref.abs().max()))
    targets = u.config.get("target_modules")
    report = {"modules": len(u.prefixes()), "max_rel_err": worst,
              "regex": targets if isinstance(targets, str) else None}  # fmt: skip
    assert all("vision" not in p for p in u.prefixes()), "LoRA reached the vision tower"

    entry = cfg["models"][0]
    path_a050 = ad.write_adapter(
        ad.combine(u, h, 0.5), env_mod.ephemeral_dir() / f"rbbd_m3c_{entry['slug']}_a050"
    )
    model, tok = load_for_inference(
        LoadSpec(entry["id"], entry.get("revision"), entry["precision"], entry["placement"])
    )
    peft_model = PeftModel.from_pretrained(model, str(path_a050)).eval()
    batch = tok(["Women attend community events."], return_tensors="pt")
    device = next(base_decoder(peft_model).parameters()).device
    with torch.inference_mode():
        out = peft_model(**{k: v.to(device) for k, v in batch.items()}, output_hidden_states=True)
    assert_finite(out.hidden_states[-1], "m3c.a050.hidden", ckpt="a050", model=entry["slug"])
    report["a050_forward"] = f"finite ({entry['precision']}, {entry['placement']})"
    _record(f"real_qlora_endpoints_{cfg['run_name']}", report)
