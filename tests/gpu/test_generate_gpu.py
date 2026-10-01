"""GPU generation checks (Kaggle). Marked gpu."""

import pytest

from rbbd import config
from rbbd.utils import env as env_mod
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.gpu


def test_vllm_hello_tp1_tp2_merged():
    """M0-T6 (B-005, D-047..D-050): vLLM serves Qwen in fp16 on one GPU and on two, both as
    downloaded and as merged weights (random rank-32 LoRA folded in) loaded from ephemeral
    disk, which is how M5 serves α-checkpoints. The merged temp copies are removed. The
    vLLM-LoRA canary and the Gemma fp16/fp32 cases are recorded, not asserted. Results land
    in `<artifacts>/env/<session>/vllm_probe.json` for upload."""
    import torch

    assert torch.cuda.device_count() >= 2, "needs the 2x T4 accelerator"
    cfg = config.load(REPO_ROOT / "configs" / "base.yaml")
    env = env_mod.detect(cfg.get("paths.artifacts"))
    out_dir = env.artifacts_root / "env" / env_mod.session_id()
    cases = cfg.get("probe.vllm_cases")
    results = {r["case"]["name"]: r for r in env_mod.probe_vllm(cases, out_dir)}

    # Every case's outcome is recorded before any assertion can fail.
    assert (out_dir / "vllm_probe.json").exists()
    for name in (
        "qwen05b_tp1_fp16_plain",
        "qwen05b_tp2_fp16_plain",
        "qwen05b_tp1_fp16_merged32",
        "qwen05b_tp2_fp16_merged32",
    ):
        r = results[name]
        assert r["ok"], f"{name} failed at {r.get('failed_stage')}: {r.get('error')}"
        assert len(r["base_tokens"]) == 4
        assert r["backend_lines"], f"{name}: no engine/backend line captured"
    for name in ("qwen05b_tp1_fp16_merged32", "qwen05b_tp2_fp16_merged32"):
        assert results[name]["merged_dir_removed"] is True, f"{name}: temp weights left behind"
    for name in ("qwen05b_tp1_fp16_lora32_canary", "gemma3_1b_tp1_fp16", "gemma3_1b_tp1_fp32"):
        r = results[name]
        assert r["ok"] or r.get("error"), f"{name}: outcome not recorded"


def test_resume_generation_shards():
    """PLACEHOLDER(M5) DC-13: resumed generation has no duplicate or missing shards."""


def test_wildguard_on_fixed_generations():
    """PLACEHOLDER(M5) WildGuard classifies 20 fixed Tier 0 generations (D-038)."""
