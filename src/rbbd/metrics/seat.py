"""SEAT bias on absolute embeddings (§3.2, Eq. 2–3); the in-space baseline of §4.5."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rbbd.metrics.rr import group_means, unit_rows


def association_cos(e_x: np.ndarray, e_attr: np.ndarray) -> np.ndarray:
    """Eq. 2: S(x) = mean_j cos(e(x), e(attr_j)). e_x [n, d], e_attr [k, d] -> [n]."""
    return (unit_rows(e_x) @ unit_rows(e_attr).T).mean(axis=1)


def row_bias_seat(e_t: np.ndarray, e_p: np.ndarray, e_n: np.ndarray) -> np.ndarray:
    """Per-target S+ - S-. e_t [n, d], e_p [p, d], e_n [q, d] -> [n]."""
    return association_cos(e_t, e_p) - association_cos(e_t, e_n)


def bias_seat(e_t: np.ndarray, e_p: np.ndarray, e_n: np.ndarray) -> float:
    """Eq. 3: B = mean over targets of (S+ - S-)."""
    return float(row_bias_seat(e_t, e_p, e_n).mean())


def bias_seat_by_group(
    e_t: np.ndarray, groups: Sequence[str], e_p: np.ndarray, e_n: np.ndarray
) -> dict[str, float]:
    """SEAT B per target group. Embeddings [*, d]; groups [n] aligned with e_t."""
    return group_means(row_bias_seat(e_t, e_p, e_n), groups)
