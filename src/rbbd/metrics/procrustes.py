"""Orthogonal Procrustes alignment and the Procrustes-SEAT baseline (§4.5; D-028).

The audited model's embeddings are rotated into the reference frame with the optimal
orthogonal map fitted on the shared anchors (`[unspecified in paper]`: we fit on the
1,000 neutral anchors, which are never scored). Audited targets are then scored
against the *reference* attribute embeddings with SEAT (Eq. 2–3).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rbbd.metrics.seat import bias_seat_by_group


def fit_orthogonal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Orthogonal R [d, d] minimising ||x R - y||_F (Schönemann 1966). x, y [m, d] row-paired."""
    m = np.asarray(x, dtype=np.float64).T @ np.asarray(y, dtype=np.float64)
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def bias_aligned_by_group(
    aud_targets: np.ndarray,
    groups: Sequence[str],
    aud_anchors: np.ndarray,
    ref_anchors: np.ndarray,
    ref_p: np.ndarray,
    ref_n: np.ndarray,
) -> dict[str, float]:
    """SEAT B of audited targets mapped into the reference frame, against reference P/N."""
    r = fit_orthogonal(aud_anchors, ref_anchors)
    return bias_seat_by_group(np.asarray(aud_targets, dtype=np.float64) @ r, groups, ref_p, ref_n)
