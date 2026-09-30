"""Kaggle detection, hardware probe, gated-access check, vLLM probe and secret loading.

Used by:
- `cli run`/`cli status` via `detect()` (artifact root, ephemeral dir, env summary);
- `cli probe` via `probe()` (M0-T1, M0-T5) and `probe_vllm()` (M0-T6);
- the Kaggle notebooks via `load_kaggle_secrets()` and `ephemeral_dir()`.

Nothing here prints or returns a secret value: `load_kaggle_secrets` returns
presence flags only, and every string written to `env.json` passes `redact`
(D-036). Hardware facts are observed at runtime and never assumed (D-002).
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rbbd.utils.logging import get_logger, redact

log = get_logger(__name__)

KAGGLE_WORKING = Path("/kaggle/working")
# Kaggle's scratch location is not documented consistently; the first existing one wins.
EPHEMERAL_CANDIDATES = (Path("/kaggle/tmp"), Path("/kaggle/temp"))
DF_CANDIDATES = ("/kaggle/working", "/kaggle/tmp", "/kaggle/temp", "/tmp", "/")

CmdRunner = Callable[[Sequence[str]], tuple[int, str]]


class HardwareError(RuntimeError):
    """The probed machine does not meet the plan's requirements (2 GPUs, sm >= 7.5, >= 14 GiB)."""


def is_kaggle() -> bool:
    """True inside a Kaggle kernel (run-type variable or the /kaggle/working mount)."""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or KAGGLE_WORKING.is_dir()


def ephemeral_dir() -> Path:
    """Scratch directory lost at session end: HF cache, materialised Tier 2 weights (D-008).

    `$RBBD_EPHEMERAL` wins; otherwise the first existing Kaggle scratch path; else the
    system temp dir.
    """
    if os.environ.get("RBBD_EPHEMERAL"):
        return Path(os.environ["RBBD_EPHEMERAL"])
    for candidate in EPHEMERAL_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return Path(tempfile.gettempdir())


def artifacts_root(configured: str | None = None) -> Path:
    """Local artifact staging root: `$RBBD_ARTIFACTS` > config `paths.artifacts` > Kaggle default >
    ./artifacts."""
    if os.environ.get("RBBD_ARTIFACTS"):
        return Path(os.environ["RBBD_ARTIFACTS"])
    if configured:
        return Path(configured)
    if is_kaggle():
        return KAGGLE_WORKING / "artifacts"
    return Path("artifacts")


def session_id() -> str:
    """Identifier for this Kaggle session: `$RBBD_SESSION_ID`, else a UTC timestamp (set once per
    process)."""
    if not os.environ.get("RBBD_SESSION_ID"):
        os.environ["RBBD_SESSION_ID"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return os.environ["RBBD_SESSION_ID"]


@dataclass(frozen=True)
class Env:
    """Where this process may write, and a secret-free summary for manifests."""

    is_kaggle: bool
    artifacts_root: Path
    ephemeral_dir: Path

    def summary(self) -> dict[str, Any]:
        """Secret-free description recorded in every stage manifest."""
        return {
            "is_kaggle": self.is_kaggle,
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "token_present": bool(os.environ.get("HF_TOKEN")),
            "session_id": os.environ.get("RBBD_SESSION_ID"),
        }


def detect(configured_artifacts: str | None = None) -> Env:
    """Resolve the runtime environment without touching GPUs or the network."""
    return Env(
        is_kaggle=is_kaggle(),
        artifacts_root=artifacts_root(configured_artifacts),
        ephemeral_dir=ephemeral_dir(),
    )


def load_kaggle_secrets(names: Sequence[str]) -> dict[str, bool]:
    """Copy Kaggle Secrets into `os.environ` and return {name: present}; values are never returned.

    Called from the notebooks right after `pip install`. Outside Kaggle, it reports
    whatever is already in the environment.
    """
    present: dict[str, bool] = {}
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        client = UserSecretsClient()
    except ImportError:
        client = None
    for name in names:
        if client is not None and not os.environ.get(name):
            try:
                value = client.get_secret(name)
            except Exception:  # secret not attached to this notebook
                value = None
            if value:
                os.environ[name] = value
        present[name] = bool(os.environ.get(name))
    return present


def _run(cmd: Sequence[str]) -> tuple[int, str]:
    try:
        done = subprocess.run(list(cmd), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return done.returncode, done.stdout + done.stderr


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Parse nvidia-smi CSV (`--query-gpu=name,memory.total,compute_cap`, noheader, nounits)."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        name, mem, cap = parts
        try:
            gpus.append(
                {"name": name, "memory_total_mib": int(float(mem)), "compute_cap": float(cap)}
            )
        except ValueError:
            continue
    return gpus


def parse_df(text: str) -> list[dict[str, str]]:
    """Parse `df -h <paths>` into one dict per mount."""
    rows = []
    for line in text.strip().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 6:
            rows.append(
                {
                    "filesystem": parts[0],
                    "size": parts[1],
                    "used": parts[2],
                    "avail": parts[3],
                    "use_pct": parts[4],
                    "mounted_on": " ".join(parts[5:]),
                }
            )
    return rows


def parse_free(text: str) -> dict[str, int]:
    """Parse the `Mem:` row of `free -g` into GiB integers."""
    for line in text.splitlines():
        if line.startswith("Mem:"):
            cols = line.split()[1:]
            keys = (
                "total_gib",
                "used_gib",
                "free_gib",
                "shared_gib",
                "buff_cache_gib",
                "available_gib",
            )
            return {k: int(v) for k, v in zip(keys, cols, strict=False) if v.lstrip("-").isdigit()}
    return {}


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["capabilities"] = [
            list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())
        ]
        # Informational only: with emulation allowed torch reports bf16 "supported" on
        # sm75, which is why env.json's `bf16_supported` is derived from compute capability (D-045).
        info["bf16_supported_including_emulation"] = bool(torch.cuda.is_bf16_supported())
    return info


def probe_hardware(run_cmd: CmdRunner = _run) -> dict[str, Any]:
    """Observe GPUs, disk, RAM, Python and torch. Never raises for missing tools."""
    rc, smi = run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"]
    )
    gpus = parse_nvidia_smi(smi) if rc == 0 else []
    df_paths = [p for p in DF_CANDIDATES if Path(p).exists()]
    _, df_out = run_cmd(["df", "-h", *df_paths])
    _, free_out = run_cmd(["free", "-g"])
    caps = [g["compute_cap"] for g in gpus]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "is_kaggle": is_kaggle(),
        "gpus": gpus,
        "gpu_count": len(gpus),
        "compute_cap_min": min(caps) if caps else None,
        # Native bf16 needs Ampere (sm80+); T4 is sm75 (D-002, D-045).
        "bf16_supported": bool(caps) and min(caps) >= 8.0,
        "nvidia_smi_rc": rc,
        "disk": parse_df(df_out),
        "ram": parse_free(free_out),
        "torch": _torch_info(),
        "ephemeral_dir": str(ephemeral_dir()),
    }


def check_requirements(
    hw: Mapping[str, Any], min_gpus: int = 2, min_cc: float = 7.5, min_mem_gib: float = 14.0
) -> list[str]:
    """Return human-readable failures of the PLAN §6 hardware assumptions (empty = OK)."""
    failures = []
    if hw["gpu_count"] < min_gpus:
        failures.append(f"expected >= {min_gpus} GPUs, found {hw['gpu_count']}")
    for i, gpu in enumerate(hw["gpus"]):
        if gpu["compute_cap"] < min_cc:
            failures.append(f"GPU {i} {gpu['name']} compute cap {gpu['compute_cap']} < {min_cc}")
        if gpu["memory_total_mib"] / 1024 < min_mem_gib:
            failures.append(
                f"GPU {i} {gpu['name']} has {gpu['memory_total_mib']} MiB < {min_mem_gib} GiB"
            )
    return failures


def check_access(repos: Sequence[Mapping[str, str]], api: Any = None) -> dict[str, str]:
    """Run `HfApi.auth_check` for each {id, type} and return {id: "ok" | "error: ..."}.

    Uses the token from `$HF_TOKEN`; downloads nothing (M0-T5, B-004).
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    results = {}
    for repo in repos:
        try:
            api.auth_check(repo["id"], repo_type=repo.get("type", "model"))
            results[repo["id"]] = "ok"
        except Exception as exc:
            results[repo["id"]] = redact(f"error: {type(exc).__name__}: {exc}")[:300]
    return results


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write redacted, pretty JSON (used for env.json and probe results)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(redact(json.dumps(data, indent=2, sort_keys=True, default=str)))


def probe(
    out_dir: Path,
    *,
    access_repos: Sequence[Mapping[str, str]] = (),
    require_gpu: bool = True,
    run_cmd: CmdRunner = _run,
    api: Any = None,
) -> dict[str, Any]:
    """M0-T1/T5: write `<out_dir>/env.json`, then raise `HardwareError` if requirements fail.

    env.json is written *before* raising so a failed probe still leaves evidence.
    """
    info: dict[str, Any] = {
        "session_id": session_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "token_present": bool(os.environ.get("HF_TOKEN")),
        **probe_hardware(run_cmd),
    }
    info["access"] = check_access(access_repos, api=api) if access_repos else {}
    # None (not True) when a check was skipped, so a skipped check never reads as a pass.
    info["access_all_ok"] = (
        all(v == "ok" for v in info["access"].values()) if access_repos else None
    )
    failures = check_requirements(info) if require_gpu else []
    info["requirements"] = {
        "checked": require_gpu,
        "ok": (not failures) if require_gpu else None,
        "failures": failures,
    }
    write_json(Path(out_dir) / "env.json", info)
    log.info("probe wrote %s", Path(out_dir) / "env.json")
    if failures:
        raise HardwareError("; ".join(failures))
    return info


# ---------------------------------------------------------------------------
# vLLM probe (M0-T6, B-005). Each case runs in its own process because vLLM does
# not reliably release GPU memory in-process, and a refused dtype must not abort
# the remaining cases.
# ---------------------------------------------------------------------------

PROBE_PROMPTS = (
    "The capital of France is",
    "Write one sentence about the ocean.",
    "List three colours:",
    "Two plus two equals",
)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_BACKEND_LINE = re.compile(
    r"(?i)(attention backend|using \w+ backend|backend|v1 engine|v0|falling back)"
)


def make_random_lora(model_id: str, rank: int, out_dir: Path) -> Path:
    """Save a small random LoRA (all 7 projections, rank `rank`) for `model_id` on CPU.

    B is scaled by 1e-2 so fp16 outputs stay finite. Used only to exercise vLLM's
    LoRA kernels at the rank the α-adapters need (2r = 32, D-007).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=list(LORA_TARGETS),
        init_lora_weights=False,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, cfg)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.mul_(1e-2)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_dir))
    return out_dir


def vllm_case(case: Mapping[str, Any], work_dir: Path) -> dict[str, Any]:
    """Run one vLLM case {name, model, tp, dtype, lora_rank|None} in-process; return a result.

    A constructor failure (e.g. Gemma 3 refusing float16) is recorded, not raised.
    """
    result: dict[str, Any] = {"case": dict(case), "ok": False}
    start = time.time()
    import vllm
    from vllm import LLM, SamplingParams

    result["vllm_version"] = vllm.__version__
    lora_rank = case.get("lora_rank")
    lora_dir = None
    if lora_rank:
        lora_dir = make_random_lora(
            case["model"], int(lora_rank), work_dir / f"lora_{case['name']}"
        )
    kwargs: dict[str, Any] = {
        "model": case["model"],
        "tensor_parallel_size": int(case["tp"]),
        "dtype": case["dtype"],
        "max_model_len": 512,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "seed": 0,
    }
    if lora_rank:
        kwargs.update(enable_lora=True, max_lora_rank=int(lora_rank), max_loras=1)
    try:
        llm = LLM(**kwargs)
    except Exception as exc:
        result["refused"] = True
        result["error"] = redact(f"{type(exc).__name__}: {exc}")[:2000]
        result["seconds"] = round(time.time() - start, 1)
        return result
    result["refused"] = False
    result["engine_module"] = type(llm.llm_engine).__module__
    params = SamplingParams(temperature=0.7, top_p=1.0, max_tokens=16, seed=0)
    base = llm.generate(list(PROBE_PROMPTS), params)
    result["base_tokens"] = [len(o.outputs[0].token_ids) for o in base]
    result["base_sample"] = base[0].outputs[0].text[:80]
    ok = all(n > 0 for n in result["base_tokens"])
    if lora_dir is not None:
        from vllm.lora.request import LoRARequest

        with_lora = llm.generate(
            list(PROBE_PROMPTS), params, lora_request=LoRARequest("probe", 1, str(lora_dir))
        )
        result["lora_tokens"] = [len(o.outputs[0].token_ids) for o in with_lora]
        ok = ok and all(n > 0 for n in result["lora_tokens"])
    result["ok"] = ok
    result["seconds"] = round(time.time() - start, 1)
    return result


def probe_vllm(cases: Sequence[Mapping[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    """Run each case via `python -m rbbd.cli vllm-case` and collect results into `vllm_probe.json`.

    Log lines mentioning the engine or attention backend are copied into each result.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        result_path = out_dir / f"vllm_{case['name']}.json"
        log_path = out_dir / f"vllm_{case['name']}.log"
        cmd = [
            sys.executable,
            "-m",
            "rbbd.cli",
            "vllm-case",
            "--json",
            json.dumps(dict(case)),
            "--out",
            str(result_path),
        ]
        log.info("vLLM probe case %s", case["name"])
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            output, rc = done.stdout + done.stderr, done.returncode
        except subprocess.TimeoutExpired as exc:
            output, rc = f"timeout after {exc.timeout}s", -1
        log_path.write_text(redact(output))
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {"case": dict(case), "ok": False, "error": redact(output[-2000:])}
        result["returncode"] = rc
        result["backend_lines"] = [
            redact(line.strip())[:300] for line in output.splitlines() if _BACKEND_LINE.search(line)
        ][:15]
        results.append(result)
    write_json(out_dir / "vllm_probe.json", {"cases": results})
    return results
