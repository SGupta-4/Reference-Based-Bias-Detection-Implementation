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

# Weight of the identity added to xᵀy before the SVD (relative to its Frobenius norm).
# With fewer anchors than dimensions (m < d: every 7–8B model, Gemma, Llama-3.2-1B), xᵀy
# has rank m and the plain SVD completes R on the remaining d − m directions with
# round-off noise, i.e. an arbitrary rotation of the target components outside the anchor
# span. The small identity term makes that completion the rotation closest to the
# identity and leaves the anchor-span solution unchanged to ~1e-10 (D-082).
IDENTITY_WEIGHT = 1e-10


def fit_orthogonal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Orthogonal R [d, d] minimising ||x R - y||_F (Schönemann 1966). x, y [m, d] row-paired.

    Outside the anchor span (m < d) R is the orthogonal completion closest to the
    identity, so fitting a model onto itself returns the identity (D-082).
    """
    m = np.asarray(x, dtype=np.float64).T @ np.asarray(y, dtype=np.float64)
    m = m + IDENTITY_WEIGHT * np.linalg.norm(m) * np.eye(m.shape[0])
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
