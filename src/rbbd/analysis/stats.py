"""Statistics behind Table 1 / Table 7 / Figure 3 (§4.1, T1, T7, F3; D-026, D-027).

Observations are (audited checkpoint, target group) pairs pooled within one setting
(tier, model, regime, benchmark); the reference checkpoint is excluded (D-027). For
each method the caller passes the per-observation value (ΔB for RR/SEAT/Procrustes,
1 - CKA for CKA drift) and the external ΔBiasScore / ΔToxicity:

- Pearson r with two-tailed p (CKA: |r|, T7);
- ROC AUC of a threshold classifier: label = Δscore > threshold, score = -ΔB
  (CKA: score = drift, AUC = max(AUC, 1 - AUC));
- MAE of an OLS fit Δscore ~ value, mean over 1,000 bootstrap resamples (in-sample
  primary, out-of-bag secondary);
- percentile bootstrap CIs, optionally resampling clusters (checkpoint x topic for
  WildGuardMix, where groups in a topic share one score, D-024). An undefined point
  estimate gets no CI, and CKA's orientation is fixed on the full sample before
  resampling, so its CI is not folded towards 1 (D-099);
- paired bootstrap of AUC differences (RR - SEAT, criterion C3 in PLAN §1.2).
CPU only; called by the analyze stage (M6).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

MIN_CLASS = 5
N_BOOT = 1000
UNDIRECTED_METHODS = frozenset({"cka"})


def pearson(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Pearson r and two-tailed p of paired samples x [n], y [n] (nan, nan if n < 3 or constant)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan, math.nan
    res = pearsonr(x, y)
    return float(res.statistic), float(res.pvalue)


def stars(p: float) -> str:
    """Significance marks as in T1: * p<.05, ** p<.01, *** p<.001."""
    if math.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def roc_auc(scores: Sequence[float], labels: Sequence[bool], min_class: int = MIN_CLASS) -> float:
    """ROC AUC of `scores` [n] for boolean `labels` [n]; nan when a class has < min_class rows."""
    labels = np.asarray(labels, bool)
    n_pos = int(labels.sum())
    if n_pos < min_class or len(labels) - n_pos < min_class:
        return math.nan
    return float(roc_auc_score(labels, np.asarray(scores, float)))


def detection_auc(
    values: Sequence[float],
    delta_score: Sequence[float],
    threshold: float,
    *,
    method: str = "rr",
    min_class: int = MIN_CLASS,
    orientation: float | None = None,
) -> float:
    """AUC of detecting Δscore > threshold from `values` (ΔB, or CKA drift if method='cka').

    Undirected methods fold the AUC (max(AUC, 1 − AUC), T7) unless `orientation` (+1 / −1,
    from `orientation_of` on the full sample) fixes the direction, as bootstrap resamples
    must (D-099)."""
    labels = np.asarray(delta_score, float) > threshold
    values = np.asarray(values, float)
    if method in UNDIRECTED_METHODS:
        if orientation is not None:
            return roc_auc(orientation * values, labels, min_class)
        auc = roc_auc(values, labels, min_class)
        return auc if math.isnan(auc) else max(auc, 1.0 - auc)
    return roc_auc(-values, labels, min_class)


def orientation_of(
    values: Sequence[float],
    delta_score: Sequence[float],
    threshold: float,
    *,
    method: str = "rr",
    min_class: int = MIN_CLASS,
) -> float | None:
    """+1 / −1: the direction in which an undirected method detects on these rows (the one
    the folded AUC picked); None for directed methods or an undefined AUC."""
    if method not in UNDIRECTED_METHODS:
        return None
    labels = np.asarray(delta_score, float) > threshold
    auc = roc_auc(np.asarray(values, float), labels, min_class)
    return None if math.isnan(auc) else (1.0 if auc >= 0.5 else -1.0)


def _ols_mae(x_fit, y_fit, x_eval, y_eval) -> float:
    if np.ptp(x_fit) == 0:
        pred = np.full_like(y_eval, y_fit.mean())
    else:
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        pred = slope * x_eval + intercept
    return float(np.mean(np.abs(y_eval - pred)))


def bootstrap_mae(
    values: Sequence[float], delta_score: Sequence[float], n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, float]:
    """MAE of the linear fit Δscore ~ value, averaged over `n_boot` resamples (T1, D-027).

    `mae_in` fits and evaluates on each resample (primary); `mae_oob` evaluates on the
    rows left out of that resample (secondary).
    """
    x, y = np.asarray(values, float), np.asarray(delta_score, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    ins, oobs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ins.append(_ols_mae(x[idx], y[idx], x[idx], y[idx]))
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob):
            oobs.append(_ols_mae(x[idx], y[idx], x[oob], y[oob]))
    return {
        "mae_in": float(np.mean(ins)),
        "mae_oob": float(np.mean(oobs)) if oobs else math.nan,
        "n_boot": n_boot,
    }


def _resample_index(rng: np.random.Generator, n: int, clusters: np.ndarray | None) -> np.ndarray:
    if clusters is None:
        return rng.integers(0, n, size=n)
    ids = np.unique(clusters)
    picked = rng.choice(ids, size=len(ids), replace=True)
    return np.concatenate([np.flatnonzero(clusters == c) for c in picked])


def bootstrap_ci(
    stat: Callable[..., float],
    data: Mapping[str, Sequence[float]],
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
    clusters: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Percentile CI of `stat(**resampled_data)`; rows (or whole clusters) are resampled.

    Resamples where the statistic is undefined (nan, e.g. a degenerate AUC) are dropped
    and counted in `n_valid`. If the statistic is undefined on the full data there is no
    CI: the defined resamples would be a biased subset (D-099).
    """
    if math.isnan(stat(**{k: np.asarray(v, float) for k, v in data.items()})):
        return {"lo": math.nan, "hi": math.nan, "n_valid": 0}
    arrays = {k: np.asarray(v, float) for k, v in data.items()}
    n = len(next(iter(arrays.values())))
    cl = None if clusters is None else np.asarray(clusters)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = _resample_index(rng, n, cl)
        val = stat(**{k: v[idx] for k, v in arrays.items()})
        if not math.isnan(val):
            draws.append(val)
    if not draws:
        return {"lo": math.nan, "hi": math.nan, "n_valid": 0}
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "n_valid": len(draws)}


def paired_auc_difference(
    values_a: Sequence[float],
    values_b: Sequence[float],
    delta_score: Sequence[float],
    threshold: float,
    *,
    method_a: str = "rr",
    method_b: str = "seat",
    n_boot: int = N_BOOT,
    seed: int = 0,
    clusters: Sequence[Any] | None = None,
) -> dict[str, float]:
    """AUC_a - AUC_b on the same observations, with a paired bootstrap CI (criterion C3).
    Undirected methods keep their full-sample orientation in every resample (D-099)."""
    o_a = orientation_of(values_a, delta_score, threshold, method=method_a)
    o_b = orientation_of(values_b, delta_score, threshold, method=method_b)

    def diff(a, b, s):
        return detection_auc(a, s, threshold, method=method_a, orientation=o_a) - detection_auc(
            b, s, threshold, method=method_b, orientation=o_b
        )

    point = diff(np.asarray(values_a, float), np.asarray(values_b, float), delta_score)
    ci = bootstrap_ci(
        diff, {"a": values_a, "b": values_b, "s": delta_score},
        n_boot=n_boot, seed=seed, clusters=clusters,
    )  # fmt: skip
    return {"diff": point, **ci}


def threshold_sweep(
    values: Sequence[float],
    delta_score: Sequence[float],
    thresholds: Sequence[float],
    *,
    method: str = "rr",
) -> list[tuple[float, float]]:
    """(threshold, AUC) pairs for the F3 analogue; degenerate thresholds give nan."""
    return [(float(t), detection_auc(values, delta_score, t, method=method)) for t in thresholds]


def summarize(
    values: Sequence[float],
    delta_score: Sequence[float],
    threshold: float,
    *,
    method: str = "rr",
    n_boot: int = N_BOOT,
    seed: int = 0,
    clusters: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Every T1/T7 number for one (setting, method): n, r, p, AUC, CIs, MAE."""
    x, y = np.asarray(values, float), np.asarray(delta_score, float)
    labels = y > threshold
    r, p = pearson(x, y)
    # Undirected (CKA): report |r| and the folded AUC, but resample with the full-sample
    # sign and orientation; folding each resample would push the CI towards 1 (D-099).
    r_sign = -1.0 if method in UNDIRECTED_METHODS and r < 0 else 1.0
    orient = orientation_of(x, y, threshold, method=method)
    r = r_sign * r

    def r_stat(x, y):
        return r_sign * pearson(x, y)[0]

    def auc_stat(x, y):
        return detection_auc(x, y, threshold, method=method, orientation=orient)

    data = {"x": x, "y": y}
    return {
        "method": method,
        "n_obs": len(x),
        "n_pos": int(labels.sum()),
        "n_neg": int((~labels).sum()),
        "pearson_r": r,
        "pearson_p": p,
        "stars": stars(p),
        "pearson_ci": bootstrap_ci(r_stat, data, n_boot=n_boot, seed=seed, clusters=clusters),
        "roc_auc": detection_auc(x, y, threshold, method=method),
        "roc_auc_ci": bootstrap_ci(auc_stat, data, n_boot=n_boot, seed=seed, clusters=clusters),
        **bootstrap_mae(x, y, n_boot=n_boot, seed=seed),
    }
