"""Seed Python, NumPy and torch from one place.

Called once by the CLI before any stage runs. torch is optional so CPU-only
stages (metrics, analysis) work without it installed.
"""

from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int) -> None:
    """Seed `random`, NumPy's global RNG and, if importable, torch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
