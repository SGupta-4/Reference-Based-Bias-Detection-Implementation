"""Relative representations and the relative-space bias B_rel (§3.3, Eq. 4–6).

Called by `metrics.delta_b` on cached embeddings (CPU, float64). Each model encodes
the shared anchor *sentences* with its own weights, so r(x) has the same coordinates
in the reference and the audited model without fitting any map between them (§3.3).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import cdist


def unit_rows(x: np.ndarray) -> np.ndarray:
    """Rows of `x` [n, d] scaled to unit L2 norm, in float64; raises on a zero row."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalise a zero embedding")
    return x / norms


def relative(e: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Eq. 4: r(x) = [cos(e(x), e(a_i))]_{i=1..m}. e [n, d], anchors [m, d] -> [n, m]."""
    return unit_rows(e) @ unit_rows(anchors).T


def association_rel(r_x: np.ndarray, r_attr: np.ndarray) -> np.ndarray:
    """Eq. 5: S_rel(x) = -mean_j ||r(x) - r(attr_j)||_2. r_x [n, m], r_attr [k, m] -> [n]."""
    return -cdist(r_x, r_attr, metric="euclidean").mean(axis=1)


def row_bias_rel(r_t: np.ndarray, r_p: np.ndarray, r_n: np.ndarray) -> np.ndarray:
    """Per-target S+_rel - S-_rel. r_t [n, m], r_p [p, m], r_n [q, m] -> [n]."""
    return association_rel(r_t, r_p) - association_rel(r_t, r_n)


def bias_rel(r_t: np.ndarray, r_p: np.ndarray, r_n: np.ndarray) -> float:
    """Eq. 6: B_rel = mean over targets of (S+_rel - S-_rel)."""
    return float(row_bias_rel(r_t, r_p, r_n).mean())


def group_means(values: np.ndarray, groups: Sequence[str]) -> dict[str, float]:
    """Mean of `values` [n] per label in `groups` [n], keyed in first-seen order."""
    labels, inverse = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    by_label = {str(lab): float(s / c) for lab, s, c in zip(labels, sums, counts, strict=True)}
    return {g: by_label[g] for g in dict.fromkeys(groups)}


def bias_rel_by_group(
    e_t: np.ndarray, groups: Sequence[str], e_p: np.ndarray, e_n: np.ndarray, anchors: np.ndarray
) -> dict[str, float]:
    """B_rel per target group (§3.4: T is instantiated per group). Embeddings [*, d]."""
    rel = row_bias_rel(relative(e_t, anchors), relative(e_p, anchors), relative(e_n, anchors))
    return group_means(rel, groups)
