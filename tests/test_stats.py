"""Statistics (D-027). DC-08."""

import math

import numpy as np
from scipy import stats as sps

from rbbd.analysis import stats


def test_roc_auc_perfect_separation():
    """AUC == 1.0 on perfectly separated synthetic data (score = -ΔB)."""
    delta_score = np.array([0.0] * 10 + [0.5] * 10)
    delta_b = np.array([0.2] * 10 + [-0.3] * 10)  # more biased checkpoints have lower ΔB
    assert stats.detection_auc(delta_b, delta_score, 0.1) == 1.0
    assert stats.detection_auc(-delta_b, delta_score, 0.1) == 0.0
    # CKA drift is undirected: either orientation reads as perfect detection.
    assert stats.detection_auc(-delta_b, delta_score, 0.1, method="cka") == 1.0


def test_roc_auc_random_near_half():
    """|AUC - 0.5| < 0.05 on 10k random points (seed 0)."""
    rng = np.random.default_rng(0)
    auc = stats.detection_auc(rng.normal(size=10_000), rng.normal(size=10_000), 0.0)
    assert abs(auc - 0.5) < 0.05


def test_degenerate_labels_are_nan():
    """Fewer than 5 rows in a class -> AUC is nan (reported as n/a, PLAN §1.2)."""
    assert math.isnan(stats.detection_auc([0.1] * 20, [0.0] * 16 + [1.0] * 4, 0.5))


def test_pearson_matches_scipy():
    """r and two-tailed p equal scipy.stats.pearsonr and the t-distribution formula."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    y = 0.4 * x + rng.normal(size=50)
    r, p = stats.pearson(x, y)
    ref = sps.pearsonr(x, y)
    assert abs(r - ref.statistic) < 1e-12 and abs(p - ref.pvalue) < 1e-12
    t = r * math.sqrt(48 / (1 - r * r))
    assert abs(p - 2 * sps.t.sf(abs(t), 48)) < 1e-9
    assert stats.stars(0.0005) == "***" and stats.stars(0.03) == "*" and stats.stars(0.2) == ""


def test_bootstrap_mae_deterministic():
    """Bootstrap MAE with seed 0 is reproducible; 1,000 resamples; OOB >= in-sample on noise."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=60)
    y = -0.5 * x + 0.1 * rng.normal(size=60)
    a = stats.bootstrap_mae(x, y, seed=0)
    assert a == stats.bootstrap_mae(x, y, seed=0)
    assert a["n_boot"] == 1000
    assert a != stats.bootstrap_mae(x, y, seed=1)
    assert 0 < a["mae_in"] < 0.2 and a["mae_oob"] >= a["mae_in"] * 0.9


def test_bootstrap_ci_and_paired_difference():
    """CIs bracket the point estimate; cluster resampling runs; RR - SEAT difference is paired."""
    rng = np.random.default_rng(3)
    score = rng.normal(size=80)
    strong = -score + 0.2 * rng.normal(size=80)  # ΔB tracks Δscore well
    weak = -score + 2.0 * rng.normal(size=80)
    s = stats.summarize(strong, score, 0.0)
    assert s["pearson_ci"]["lo"] <= s["pearson_r"] <= s["pearson_ci"]["hi"]
    assert s["roc_auc_ci"]["lo"] <= s["roc_auc"] <= s["roc_auc_ci"]["hi"]
    assert s["n_pos"] + s["n_neg"] == 80 and s["stars"] == "***"
    clusters = np.repeat(np.arange(20), 4)
    c = stats.summarize(strong, score, 0.0, clusters=clusters, n_boot=200)
    assert c["roc_auc_ci"]["n_valid"] > 0
    diff = stats.paired_auc_difference(strong, weak, score, 0.0, n_boot=300)
    assert diff["diff"] > 0 and diff["lo"] > 0


def test_threshold_sweep():
    """One (threshold, AUC) pair per threshold; extreme thresholds are degenerate (nan)."""
    rng = np.random.default_rng(4)
    score = rng.uniform(size=50)
    sweep = stats.threshold_sweep(-score, score, [0.25, 0.5, 2.0])
    assert [t for t, _ in sweep] == [0.25, 0.5, 2.0]
    assert sweep[0][1] == 1.0 and math.isnan(sweep[2][1])
