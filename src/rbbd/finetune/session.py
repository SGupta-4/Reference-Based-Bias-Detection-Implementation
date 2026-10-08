"""Pre-registered M3c session decisions (D-070, D-076), as pure functions the notebook calls.

The M3c notebook runs a probe, reads its `done.json`/`data.json` files through
`probe_results`, and applies these rules before any long run starts. Keeping them here
(not in notebook cells) makes the rules testable on CPU and identical across sessions.

Rules:
- `epoch_decision` (D-070): 3 epochs if the harmful run projects <= 9 h at 3 epochs,
  else 1 epoch if <= 9 h at 1 epoch, else stop and ask the user.
- `gemma_fp16_ok` (D-076, from D-005): fp16 compute is kept only if the probe finished
  (no NonFiniteError from the loss/grad guard or the activation monitor), the fp16
  scaler skipped at most `max_skips` steps, and the largest final-layer |hidden| stayed
  within half the fp16 range.
- `session_ok` (D-072/D-076): start the long run only if elapsed + bound <= limit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FP16_MAX = 65504.0
EPOCH_LIMIT_H = 9.0
SESSION_LIMIT_H = 11.5


def probe_results(
    artifacts_root: Path, probe_slug: str, regime: str = "qlora", compute: str | None = None
) -> dict[str, dict[str, Any]]:
    """{split: done.json} for a probe's seed-0 jobs, optionally only those run at `compute`.

    `compute` is read from each job's `data.json` spec, so an fp16 and an fp32 probe of the
    same model (different train keys) are told apart.
    """
    out: dict[str, dict[str, Any]] = {}
    base = Path(artifacts_root) / "train" / probe_slug / regime
    for done_path in sorted(base.glob("*/seed0/*/done.json")):
        job = done_path.parent
        spec = json.loads((job / "data.json").read_text()).get("spec", {})
        if compute is not None and spec.get("compute") != compute:
            continue
        out[job.parent.parent.name] = json.loads(done_path.read_text())
    return out


def epoch_decision(
    harmful_done: dict[str, Any] | None, limit_h: float = EPOCH_LIMIT_H
) -> dict[str, Any]:
    """D-070 epochs rule from the harmful probe's `projected_hours`.

    Returns {"epochs": 3 | 1 | None, "hours": projected h at that count, "reason": str};
    epochs None means stop and ask the user.
    """
    if not harmful_done or not harmful_done.get("projected_hours"):
        return {"epochs": None, "hours": None, "reason": "no harmful probe result"}
    ph = harmful_done["projected_hours"]
    h3 = next((v for k, v in ph.items() if k.startswith("3")), None)
    h1 = ph.get("1_epochs")
    if h3 is not None and h3 <= limit_h:
        return {"epochs": 3, "hours": h3, "reason": f"3 epochs project {h3} h <= {limit_h} h"}
    if h1 is not None and h1 <= limit_h:
        return {
            "epochs": 1,
            "hours": h1,
            "reason": f"3 epochs project {h3} h > {limit_h} h; 1 epoch {h1} h (D-042 Q3)",
        }
    return {
        "epochs": None,
        "hours": h1,
        "reason": f"even 1 epoch projects {h1} h > {limit_h} h: ask",
    }


def gemma_fp16_ok(
    probe: dict[str, dict[str, Any]], failure_log: str = "", max_skips: int = 5
) -> dict[str, Any]:
    """D-076 fp16 criterion for Gemma training from an fp16 probe.

    `probe` = `probe_results(..., compute="fp16")`; `failure_log` = probe job logs when the
    probe exited non-zero. Returns {"ok": bool, "reason": str, "max_abs_hidden": float|None}.
    """
    if "NonFiniteError" in failure_log:
        return {
            "ok": False,
            "reason": "guard or activation monitor fired (NonFiniteError)",
            "max_abs_hidden": None,
        }
    if set(probe) != {"unharmful", "harmful"}:
        return {
            "ok": False,
            "reason": f"fp16 probe incomplete: finished {sorted(probe)}",
            "max_abs_hidden": None,
        }
    worst = max(float(d.get("max_abs_hidden", float("inf"))) for d in probe.values())
    skips = max(int(d.get("scaler_skipped_steps", 0)) for d in probe.values())
    if skips > max_skips:
        return {
            "ok": False,
            "reason": f"{skips} scaler-skipped steps > {max_skips}",
            "max_abs_hidden": worst,
        }
    if not worst <= FP16_MAX / 2:
        return {
            "ok": False,
            "reason": f"max |hidden| {worst} > fp16 max / 2",
            "max_abs_hidden": worst,
        }
    return {
        "ok": True,
        "reason": f"finite; {skips} skips; max |hidden| {worst}",
        "max_abs_hidden": worst,
    }


def session_ok(elapsed_h: float, bound_h: float | None, limit_h: float = SESSION_LIMIT_H) -> bool:
    """True if the long run (bound_h) still fits in the session after elapsed_h hours."""
    return bound_h is not None and elapsed_h + bound_h <= limit_h
