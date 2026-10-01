"""Linear CKA and the CKA-drift baseline (§4.5, T7; D-028).

CKA drift = 1 - CKA between the reference and audited representations of one group's
target sentences. It is invariant to rotation and isotropic scaling and undirected:
it says how far a group moved, not towards which valence (so its ROC AUC is reported
as max(AUC, 1 - AUC) by `analysis.stats`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019) of row-paired x [n, d1], y [n, d2], via n x n Grams."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    k = x @ x.T
    l_ = y @ y.T
    return float((k * l_).sum() / np.sqrt((k * k).sum() * (l_ * l_).sum()))


def cka_drift_by_group(
    ref_targets: np.ndarray, aud_targets: np.ndarray, groups: Sequence[str]
) -> dict[str, float]:
    """1 - CKA per group between reference and audited target embeddings [n, d]."""
    labels = np.asarray(groups)
    return {
        g: 1.0 - linear_cka(ref_targets[labels == g], aud_targets[labels == g])
        for g in dict.fromkeys(groups)
    }
