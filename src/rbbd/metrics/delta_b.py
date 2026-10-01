"""Representational Bias Shift ΔB = B_aud - B_ref per target group (§3.4, Eq. 7).

`EmbeddingSet` holds one checkpoint's embeddings for one ablation cell (pooling,
layer site, anchor source/size, attribute and target variant), all from the same
cache entry. `delta_b_all` returns every method's per-group value for one
(reference, audited) pair; the deltab stage (M4) turns those into CSV rows.
A negative ΔB means the group moved towards the negative attributes (increased bias).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rbbd.metrics.cka import cka_drift_by_group
from rbbd.metrics.procrustes import bias_aligned_by_group
from rbbd.metrics.rr import bias_rel_by_group
from rbbd.metrics.seat import bias_seat_by_group

METHODS = ("rr", "seat", "procrustes", "cka")


@dataclass(frozen=True)
class EmbeddingSet:
    """One checkpoint's embeddings: targets [n, d] with group labels [n], P [p, d], N [q, d],
    anchors [m, d]. All rows must share d."""

    targets: np.ndarray
    target_groups: tuple[str, ...]
    positives: np.ndarray
    negatives: np.ndarray
    anchors: np.ndarray

    def __post_init__(self) -> None:
        dims = {a.shape[1] for a in (self.targets, self.positives, self.negatives, self.anchors)}
        if len(dims) != 1:
            raise ValueError(f"embedding widths differ: {dims}")
        if len(self.target_groups) != self.targets.shape[0]:
            raise ValueError("target_groups must align with target rows")


def group_bias(es: EmbeddingSet, method: str) -> dict[str, float]:
    """B per group for `method` in {"rr", "seat"} (Eq. 3 or Eq. 6)."""
    if method == "rr":
        return bias_rel_by_group(
            es.targets, es.target_groups, es.positives, es.negatives, es.anchors
        )
    if method == "seat":
        return bias_seat_by_group(es.targets, es.target_groups, es.positives, es.negatives)
    raise ValueError(f"group_bias supports rr and seat, not {method!r}")


def delta(b_aud: dict[str, float], b_ref: dict[str, float]) -> dict[str, float]:
    """Eq. 7 per group: B_aud - B_ref (same group keys required)."""
    if b_aud.keys() != b_ref.keys():
        raise ValueError("reference and audited groups differ")
    return {g: b_aud[g] - b_ref[g] for g in b_ref}


def delta_b_all(ref: EmbeddingSet, aud: EmbeddingSet) -> dict[str, dict[str, float]]:
    """Per-group value of every method for one (reference, audited) pair.

    rr, seat: ΔB (Eq. 7). procrustes: B of rotated audited targets vs reference P/N
    minus B_ref (SEAT). cka: 1 - CKA drift (undirected, not a difference).
    """
    if ref.target_groups != aud.target_groups:
        raise ValueError("reference and audited target rows must be aligned")
    seat_ref = group_bias(ref, "seat")
    aligned = bias_aligned_by_group(
        aud.targets, aud.target_groups, aud.anchors, ref.anchors, ref.positives, ref.negatives
    )
    return {
        "rr": delta(group_bias(aud, "rr"), group_bias(ref, "rr")),
        "seat": delta(group_bias(aud, "seat"), seat_ref),
        "procrustes": delta(aligned, seat_ref),
        "cka": cka_drift_by_group(ref.targets, aud.targets, ref.target_groups),
    }
