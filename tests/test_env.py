"""Environment probe and secret handling (D-002, D-036, D-045)."""

import json
import sys
import types

import pytest

from rbbd.utils import env as env_mod
from rbbd.utils.logging import get_logger

# Built by concatenation so the source file itself never contains a token-shaped string.
FAKE_TOKEN = "hf_" + "Ab3" * 12

SMI_2XT4 = "Tesla T4, 15360, 7.5\nTesla T4, 15360, 7.5\n"
DF = (
    "Filesystem      Size  Used Avail Use% Mounted on\n"
    "overlay         8.0T  6.1T  1.9T  77% /\n"
    "/dev/nvme0n1    20G   1.0G   19G   5% /kaggle/working\n"
    "overlay         8.0T  6.1T  1.9T  77% /\n"
)
FREE = (
    "               total        used        free      shared  buff/cache   available\n"
    "Mem:              31           2          26           0           2          28\n"
    "Swap:              0           0           0\n"
)


def fake_runner(smi=SMI_2XT4):
    def run(cmd):
        return {"nvidia-smi": (0, smi), "df": (0, DF), "free": (0, FREE)}[cmd[0]]

    return run


class FakeApi:
    """auth_check stand-in: raises for repos listed in `denied`."""

    def __init__(self, denied=()):
        self.denied = set(denied)

    def auth_check(self, repo_id, repo_type="model"):
        if repo_id in self.denied:
            raise PermissionError(f"403 gated repo {repo_id} with token {FAKE_TOKEN}")


REPOS = [
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "type": "model"},
    {"id": "allenai/wildguardmix", "type": "dataset"},
]


def test_probe_output_schema(tmp_path, monkeypatch):
    """env.json contains gpus, compute_cap, bf16_supported, disk, ram, access (DC-16 fields)."""
    monkeypatch.setenv("RBBD_SESSION_ID", "test-session")
    info = env_mod.probe(tmp_path, access_repos=REPOS, run_cmd=fake_runner(), api=FakeApi())
    written = json.loads((tmp_path / "env.json").read_text())
    for key in (
        "gpus",
        "gpu_count",
        "compute_cap_min",
        "bf16_supported",
        "disk",
        "ram",
        "access",
        "requirements",
        "torch",
        "python",
        "session_id",
        "token_present",
    ):
        assert key in written, key
    assert written["gpu_count"] == 2 and written["compute_cap_min"] == 7.5
    assert written["bf16_supported"] is False  # sm75: no native bf16 even if torch emulates it
    assert written["gpus"][0] == {"name": "Tesla T4", "memory_total_mib": 15360, "compute_cap": 7.5}
    # `df -h /tmp /` reports the same overlay twice; one row per mount point is kept.
    assert sorted(d["mounted_on"] for d in written["disk"]) == ["/", "/kaggle/working"]
    assert written["ram"]["total_gib"] == 31 and written["ram"]["available_gib"] == 28
    assert written["access"] == {r["id"]: "ok" for r in REPOS} and written["access_all_ok"]
    assert written["requirements"] == {"checked": True, "ok": True, "failures": []}
    assert info["session_id"] == "test-session"


def test_probe_fails_loudly_but_writes_evidence(tmp_path):
    """One GPU -> HardwareError, yet env.json exists with the failure recorded."""
    with pytest.raises(env_mod.HardwareError, match="expected >= 2 GPUs"):
        env_mod.probe(tmp_path, run_cmd=fake_runner("Tesla T4, 15360, 7.5\n"))
    written = json.loads((tmp_path / "env.json").read_text())
    assert written["requirements"]["ok"] is False
    assert written["access_all_ok"] is None  # skipped, not passed
    p100 = env_mod.check_requirements(
        {
            "gpu_count": 2,
            "gpus": [{"name": "P100", "memory_total_mib": 16384, "compute_cap": 6.0}] * 2,
        }
    )
    assert any("compute cap 6.0 < 7.5" in f for f in p100)


def test_secrets_redacted_in_logs(tmp_path, monkeypatch, capsys):
    """Strings matching hf_* tokens never appear in log output, env.json or access errors."""
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    get_logger("rbbd.test").error("upload failed for token %s", FAKE_TOKEN)
    err = capsys.readouterr().err
    assert FAKE_TOKEN not in err and "[REDACTED]" in err

    env_mod.probe(
        tmp_path,
        access_repos=REPOS,
        run_cmd=fake_runner(),
        api=FakeApi(denied={"allenai/wildguardmix"}),
    )
    text = (tmp_path / "env.json").read_text()
    written = json.loads(text)
    assert FAKE_TOKEN not in text
    assert written["token_present"] is True
    assert written["access"]["allenai/wildguardmix"].startswith("error: PermissionError")
    assert written["access_all_ok"] is False
    # Library identifiers that merely start with hf_ are left alone (D-045).
    from rbbd.utils.logging import redact

    assert redact("hf_overrides=None hf_config_path") == "hf_overrides=None hf_config_path"


def test_load_kaggle_secrets_reports_presence_only(monkeypatch):
    """Secrets are copied into the environment; only booleans are returned."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    class Client:
        def get_secret(self, name):
            if name == "HF_TOKEN":
                return FAKE_TOKEN
            raise KeyError(name)

    monkeypatch.setitem(
        sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=Client)
    )
    present = env_mod.load_kaggle_secrets(["HF_TOKEN", "GITHUB_TOKEN"])
    assert present == {"HF_TOKEN": True, "GITHUB_TOKEN": False}
    assert env_mod.os.environ["HF_TOKEN"] == FAKE_TOKEN


def test_detect_paths(monkeypatch, tmp_path):
    """$RBBD_ARTIFACTS and $RBBD_EPHEMERAL override the defaults; summary carries no secret."""
    monkeypatch.setenv("RBBD_ARTIFACTS", str(tmp_path / "a"))
    monkeypatch.setenv("RBBD_EPHEMERAL", str(tmp_path / "e"))
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    env = env_mod.detect()
    assert env.artifacts_root == tmp_path / "a" and env.ephemeral_dir == tmp_path / "e"
    assert FAKE_TOKEN not in json.dumps(env.summary())


def _fake_vllm(fail_with_lora: bool):
    """A stand-in `vllm` module: LLM() raises a Triton-style error when LoRA is enabled."""

    class Out:
        def __init__(self, text):
            self.outputs = [types.SimpleNamespace(token_ids=[1, 2, 3], text=text)]

    class LLM:
        calls = []

        def __init__(self, **kwargs):
            LLM.calls.append(kwargs)
            if fail_with_lora and kwargs.get("enable_lora"):
                raise RuntimeError("PassManager::run failed")
            self.llm_engine = types.SimpleNamespace()

        def generate(self, prompts, params, lora_request=None):
            return [Out(f"out {i}") for i, _ in enumerate(prompts)]

    lora_mod = types.SimpleNamespace(LoRARequest=lambda *a: a)
    mod = types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw, __version__="fake")
    return {
        "vllm": mod,
        "vllm.lora": types.SimpleNamespace(request=lora_mod),
        "vllm.lora.request": lora_mod,
    }


def test_vllm_case_records_failed_stage(tmp_path, monkeypatch):
    """A constructor failure is recorded with its stage and traceback; a clean case keeps texts."""
    for name, mod in _fake_vllm(fail_with_lora=True).items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(env_mod, "make_random_lora", lambda model, rank, out: out)

    case = {"name": "q", "model": "m", "tp": 1, "dtype": "float16", "lora_rank": 32}
    failed = env_mod.vllm_case(case, tmp_path)
    assert failed["ok"] is False and failed["failed_stage"] == "construct"
    assert failed["error"] == "RuntimeError: PassManager::run failed"
    assert any("PassManager::run failed" in line for line in failed["traceback_tail"])

    clean = env_mod.vllm_case({**case, "lora_rank": None}, tmp_path)
    assert clean["ok"] is True and clean["failed_stage"] is None
    assert clean["base_texts"] == ["out 0", "out 1", "out 2", "out 3"]


def test_vllm_case_merged_serves_local_weights_and_cleans_up(tmp_path, monkeypatch):
    """Merged mode loads the folded checkpoint from ephemeral disk without LoRA, then removes it."""
    fake = _fake_vllm(fail_with_lora=True)
    for name, mod in fake.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("RBBD_EPHEMERAL", str(tmp_path / "eph"))
    monkeypatch.setattr(env_mod, "make_random_lora", lambda model, rank, out: out)

    def fake_merge(model_id, lora_dir, out_dir):
        out_dir.mkdir(parents=True)
        (out_dir / "model.safetensors").write_bytes(b"x" * 10)
        return out_dir

    monkeypatch.setattr(env_mod, "materialize_merged", fake_merge)
    case = {
        "name": "m",
        "model": "m",
        "tp": 2,
        "dtype": "float16",
        "mode": "merged",
        "lora_rank": 32,
    }
    result = env_mod.vllm_case(case, tmp_path)
    merged_dir = tmp_path / "eph" / "rbbd_probe" / "m"
    assert result["ok"] is True and result["merged_bytes"] == 10
    assert result["merged_dir_removed"] is True and not merged_dir.exists()
    kwargs = fake["vllm"].LLM.calls[-1]
    assert kwargs["model"] == str(merged_dir) and "enable_lora" not in kwargs
    assert kwargs["tensor_parallel_size"] == 2


def test_torchvision_pinned_with_torch():
    """B-012: every extra that reinstalls torch pins the matching torchvision.

    Kaggle preinstalls a torchvision built for its own torch. When an extra replaces
    torch without replacing torchvision, `transformers` fails to import any model
    class (`operator torchvision::nms does not exist`).
    """
    from pathlib import Path

    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    for name, pins in extras.items():
        if "torch==2.7.1" in pins:
            assert "torchvision==0.22.1" in pins, f"extra '{name}' pins torch but not torchvision"


def test_transformers_model_classes_import():
    """B-012: the installed torch/torchvision pair lets transformers import model classes."""
    import importlib.metadata as md

    from transformers import Gemma3ForConditionalGeneration, LlamaForCausalLM, Qwen2ForCausalLM

    assert Qwen2ForCausalLM and LlamaForCausalLM and Gemma3ForConditionalGeneration
    assert md.version("torch").startswith("2.7.1")
    assert md.version("torchvision").startswith("0.22.1")
