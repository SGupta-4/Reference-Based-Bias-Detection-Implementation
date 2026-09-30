"""NaN/Inf guard (D-006).

`assert_finite` is called on final-layer hidden states for every batch before
pooling, on pooled embeddings before they are cached (M2), and `assert_finite_scalar`
on training loss and grad-norm every step (M3). It is never disabled by config:
fp16 on T4 can overflow (B-001), and a silent NaN would flow into ΔB as a number.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class NonFiniteError(RuntimeError):
    """A tensor or scalar that must be finite contained NaN or Inf."""


def _format_context(context: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(context.items()))


def assert_finite(x: Any, where: str, **context: Any) -> None:
    """Raise `NonFiniteError` if torch tensor or NumPy array `x` (any shape) has NaN/Inf.

    `where` names the call site (e.g. "embed.hidden_states"); `context` should
    identify the model, checkpoint slug and batch index so the error is actionable.
    The check is one reduction on the tensor's own device.
    """
    if type(x).__module__.startswith("torch"):
        import torch

        bad = ~torch.isfinite(x)
        if not bool(bad.any()):
            return
        n_nan = int(torch.isnan(x).sum())
        n_inf = int(torch.isinf(x).sum())
        shape = tuple(x.shape)
    else:
        arr = np.asarray(x)
        bad = ~np.isfinite(arr)
        if not bool(bad.any()):
            return
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        shape = arr.shape
    raise NonFiniteError(
        f"non-finite values at {where}: nan={n_nan} inf={n_inf} shape={shape}"
        + (f" ({_format_context(context)})" if context else "")
    )


def assert_finite_scalar(value: float, where: str, **context: Any) -> None:
    """Raise `NonFiniteError` if a Python/NumPy scalar (loss, grad-norm) is NaN/Inf."""
    if not math.isfinite(float(value)):
        raise NonFiniteError(
            f"non-finite scalar at {where}: {value}"
            + (f" ({_format_context(context)})" if context else "")
        )
