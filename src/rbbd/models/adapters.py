"""Exact α-combined LoRA adapters (App. C merging; D-007, D-066).

The spectrum between the unharmful (u) and harmful (h) LoRA endpoints is linear in ΔW:

    ΔW(α) = α·ΔW_u + (1−α)·ΔW_h,   ΔW_x = s_x · B_x A_x,   s_x = lora_alpha_x / r_x

Interpolating A and B separately is not linear in ΔW, so for interior α a rank-2r adapter
is built by concatenation (D-007):

    A' = [A_u ; A_h]                    [2r, in]
    B' = [α·s_u·B_u | (1−α)·s_h·B_h]    [out, 2r]
    lora_alpha' = r' = 2r  →  scaling 1,  B'A' = ΔW(α)

At α ∈ {1, 0} the trained endpoint adapter itself is returned, so a100 and a000 are
bit-identical to u and h (D-066, DC-07). Combination runs in fp32 on CPU.

Callers: M4's `embed` stage (PEFT adapter on the fp16 base) and M5's merged-weight
generation path (D-050). Adapters are PEFT directories: `adapter_config.json` +
`adapter_model.safetensors` with keys `<prefix>.lora_A.weight` [r, in] and
`<prefix>.lora_B.weight` [out, r].
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_FILE = "adapter_config.json"
WEIGHTS_FILE = "adapter_model.safetensors"
META_FILE = "rbbd_meta.json"  # provenance of a combined adapter; PEFT ignores it
_A, _B = ".lora_A.weight", ".lora_B.weight"


class AdapterError(ValueError):
    """An adapter this module cannot combine exactly (DoRA, per-module ranks, bias, mismatch)."""


@dataclass
class LoraAdapter:
    """A PEFT LoRA adapter in memory: its config, {key: tensor} (CPU) and our provenance."""

    config: dict[str, Any]
    tensors: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def prefixes(self) -> list[str]:
        """Module prefixes that carry a LoRA pair, sorted."""
        return sorted(k[: -len(_A)] for k in self.tensors if k.endswith(_A))


def scaling(config: dict[str, Any]) -> float:
    """PEFT scaling s = lora_alpha / r (or / sqrt(r) under rsLoRA)."""
    r = int(config["r"])
    alpha = float(config["lora_alpha"])
    return alpha / math.sqrt(r) if config.get("use_rslora") else alpha / r


def _check_combinable(config: dict[str, Any], where: str) -> None:
    # Anything that makes ΔW differ from s·B·A, or s differ across modules, breaks the
    # concatenation identity; refuse rather than combine approximately.
    if config.get("use_dora"):
        raise AdapterError(f"{where}: DoRA adapters cannot be combined by concatenation")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise AdapterError(f"{where}: per-module rank/alpha patterns are not supported")
    if config.get("bias", "none") != "none":
        raise AdapterError(f"{where}: LoRA bias must be 'none'")


def read_adapter(path: str | Path) -> LoraAdapter:
    """Load a PEFT adapter directory onto CPU (dtype as saved; PEFT saves fp32 here)."""
    from safetensors.torch import load_file

    path = Path(path)
    config = json.loads((path / CONFIG_FILE).read_text())
    _check_combinable(config, str(path))
    meta = json.loads((path / META_FILE).read_text()) if (path / META_FILE).exists() else {}
    return LoraAdapter(config=config, tensors=load_file(str(path / WEIGHTS_FILE)), meta=meta)


def write_adapter(adapter: LoraAdapter, path: str | Path) -> Path:
    """Write `adapter` as a PEFT directory loadable by `PeftModel.from_pretrained`."""
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / CONFIG_FILE).write_text(json.dumps(adapter.config, indent=2, sort_keys=True))
    save_file({k: v.contiguous() for k, v in adapter.tensors.items()}, str(path / WEIGHTS_FILE))
    if adapter.meta:
        (path / META_FILE).write_text(json.dumps(adapter.meta, indent=2, sort_keys=True))
    return path


def delta_w(adapter: LoraAdapter) -> dict[str, Any]:
    """{prefix: ΔW [out, in] fp32} = s · B @ A for every LoRA module of `adapter`."""
    s = scaling(adapter.config)
    return {
        p: s * (adapter.tensors[p + _B].float() @ adapter.tensors[p + _A].float())
        for p in adapter.prefixes()
    }


def combine(u: LoraAdapter, h: LoraAdapter, alpha: float) -> LoraAdapter:
    """Rank-2r adapter whose ΔW equals α·ΔW_u + (1−α)·ΔW_h (D-007); tensors fp32.

    u and h must cover the same modules with the same rank and target set.
    """
    import torch

    for key in ("r", "target_modules", "base_model_name_or_path"):
        if u.config.get(key) != h.config.get(key):
            raise AdapterError(f"endpoint configs differ in {key!r}")
    if u.prefixes() != h.prefixes():
        raise AdapterError("endpoint adapters cover different modules")
    s_u, s_h = scaling(u.config), scaling(h.config)
    rank = 2 * int(u.config["r"])
    tensors = {}
    for p in u.prefixes():
        tensors[p + _A] = torch.cat([u.tensors[p + _A].float(), h.tensors[p + _A].float()], dim=0)
        b_u = alpha * s_u * u.tensors[p + _B].float()
        b_h = (1.0 - alpha) * s_h * h.tensors[p + _B].float()
        tensors[p + _B] = torch.cat([b_u, b_h], dim=1)
    config = dict(u.config)
    # scaling = lora_alpha / r = 1, so B'A' is ΔW(α) with no further factor.
    config.update(r=rank, lora_alpha=rank, use_rslora=False, lora_dropout=0.0)
    meta = {"combined": {"alpha": alpha, "s_u": s_u, "s_h": s_h, "endpoint_rank": rank // 2}}
    return LoraAdapter(config=config, tensors=tensors, meta=meta)


def adapter_for_alpha(u: LoraAdapter, h: LoraAdapter, alpha: float) -> LoraAdapter:
    """The spectrum adapter at α: u itself at α=1, h itself at α=0, else `combine` (D-066)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if alpha == 1.0:
        return u
    if alpha == 0.0:
        return h
    return combine(u, h, alpha)
