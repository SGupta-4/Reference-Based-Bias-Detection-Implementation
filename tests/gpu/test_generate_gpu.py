"""GPU generation checks (Kaggle). Marked gpu."""

import pytest

from rbbd import config
from rbbd.utils import env as env_mod
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.gpu


def test_vllm_hello_tp1_tp2_lora():
    """M0-T6 (B-005, D-047): vLLM starts on 2x T4 with TP=1 and TP=2 in fp16 and serves a rank-32
    LoRA; the Gemma 3 fp16 and fp32 outcomes are recorded either way. Results land in
    `<artifacts>/env/<session>/vllm_probe.json` for upload."""
    import torch

    assert torch.cuda.device_count() >= 2, "needs the 2x T4 accelerator"
    cfg = config.load(REPO_ROOT / "configs" / "base.yaml")
    env = env_mod.detect(cfg.get("paths.artifacts"))
    out_dir = env.artifacts_root / "env" / env_mod.session_id()
    results = {
        r["case"]["name"]: r for r in env_mod.probe_vllm(cfg.get("probe.vllm_cases"), out_dir)
    }

    for name in ("qwen05b_tp1_fp16_lora32", "qwen05b_tp2_fp16_lora32"):
        r = results[name]
        assert r["ok"], f"{name}: {r.get('error')}"
        assert len(r["base_tokens"]) == 4 and len(r["lora_tokens"]) == 4
        assert r["backend_lines"], f"{name}: no engine/backend line captured"
    for name in ("gemma3_1b_tp1_fp16", "gemma3_1b_tp1_fp32"):
        r = results[name]
        assert r["ok"] or r.get("error"), f"{name}: outcome not recorded"
    assert (out_dir / "vllm_probe.json").exists()


def test_resume_generation_shards():
    """PLACEHOLDER(M5) DC-13: resumed generation has no duplicate or missing shards."""


def test_wildguard_on_fixed_generations():
    """PLACEHOLDER(M5) WildGuard classifies 20 fixed Tier 0 generations (D-038)."""
