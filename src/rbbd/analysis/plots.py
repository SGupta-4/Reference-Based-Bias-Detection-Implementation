"""Figures F-R1-F-R5 of the M7 report (PLAN §10; D-102). Static PNGs for `results/REPORT.md`.

Inputs are the same aggregates the tables use: observations joined by
`analysis.analyze.observations` (primary cell), the M6 `ablations.csv` / `t1.csv` rows and
the ΔB rows of multi-seed spectra. Every figure has a table view in the report (R1-R5),
which is the relief for the light categorical slots below 3:1 contrast (dataviz check).

Colour: categorical slots in fixed order (validated: adjacent CVD ΔE ≥ 9.1 on the light
surface), a one-hue blue ramp for α (magnitude), recessive grid and axes. CPU only;
`matplotlib` with the Agg backend; called by `analysis.report.build_report`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rbbd.analysis import stats
from rbbd.analysis.tables import BENCH_NAMES, METHOD_NAMES, METHODS, num, parse_setting

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 5)))
ALPHA_RAMP = ("#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#184f95", "#0d366b")
ANCHOR_SIZES = (50, 100, 250, 500, 1000)
SOURCES = ("neutral", "word", "alpaca", "tulu")
THRESHOLDS = {"dt": 0.1, "wgm": 0.1, "toxigen": 0.03}  # D-027, as analysis.analyze


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "font.family": "sans-serif",
        }
    )
    return plt


def _grid(plt, n: int, w: float = 3.2, h: float = 2.7):
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def _legend(fig, ax, ncol: int, title: str | None = None) -> None:
    """One figure-level legend below the panels (never over the data)."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            title=title,
            title_fontsize=8,
            bbox_to_anchor=(0.5, 0.0),
        )
        fig.subplots_adjust(bottom=0.0)
        fig._rbbd_legend = True


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 1 if not getattr(fig, "_rbbd_legend", False) else 2
    fig.tight_layout(rect=(0, 0.035 * rows + (0.03 if rows > 1 else 0), 1, 1))
    fig.savefig(path, dpi=120, metadata={"Software": None})
    fig.clf()
    return path


def _alpha_of(ckpt: str) -> float:
    return int(ckpt[1:]) / 100.0


def _by_setting(obs: Sequence[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for o in obs:
        if o.seed == 0:
            out.setdefault(f"{o.model}/{o.regime}-s{o.seed}", []).append(o)
    return dict(sorted(out.items()))


def scatter(obs_rr: Mapping[str, Sequence[Any]], out_dir: Path) -> list[Path]:
    """F-R1 (F2 analogue): Δscore against RR ΔB per seed-0 spectrum, one file per
    benchmark; points coloured by α (light = harmful end), OLS line, r in the title.
    `obs_rr`: {bench: [Obs]} of the primary cell."""
    plt = _plt()
    paths = []
    alphas = sorted({_alpha_of(o.ckpt) for os_ in obs_rr.values() for o in os_})
    colour = {a: ALPHA_RAMP[min(i, len(ALPHA_RAMP) - 1)] for i, a in enumerate(alphas)}
    for bench, obs in obs_rr.items():
        groups = _by_setting(obs)
        if not groups:
            continue
        fig, axes = _grid(plt, len(groups))
        for ax, (name, os_) in zip(axes, groups.items(), strict=True):
            x = np.array([o.value for o in os_])
            y = np.array([o.dscore for o in os_])
            for a in alphas:
                m = np.array([_alpha_of(o.ckpt) == a for o in os_])
                if m.any():
                    ax.scatter(
                        x[m],
                        y[m],
                        s=14,
                        color=colour[a],
                        label=f"α={a:.1f}",
                        edgecolors=SURFACE,
                        linewidths=0.4,
                    )
            if np.ptp(x) > 0:
                slope, icpt = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 2)
                ax.plot(xs, slope * xs + icpt, color=INK_2, linewidth=1.2)
            ax.axhline(THRESHOLDS[bench], color=MUTED, linewidth=0.8, linestyle=":")
            r, _ = stats.pearson(x, y)
            ax.set_title(f"{name}\nr = {r:+.2f}")
            ax.set_xlabel("ΔB (RR)")
            ax.set_ylabel(f"Δ score ({BENCH_NAMES[bench]})")
        _legend(fig, axes[0], ncol=len(alphas), title="α (weight on the unharmful endpoint)")
        fig.suptitle(f"F-R1 · {BENCH_NAMES[bench]}: Δ score vs ΔB (RR), seed 0", color=INK)
        paths.append(_save(fig, Path(out_dir) / f"f_r1_scatter_{bench}.png"))
    return paths


def threshold_curves(
    obs: Mapping[tuple[str, str], Sequence[Any]],
    tiers: Mapping[str, int],
    sweep: Sequence[float],
    out_dir: Path,
) -> list[Path]:
    """F-R2 (F3 analogue): AUC against the detection threshold for the four methods, per
    seed-0 spectrum and pooled tier, one file per benchmark. `obs`: {(bench, method): [Obs]}."""
    plt = _plt()
    paths = []
    for bench in BENCH_NAMES:
        per = {m: obs.get((bench, m), []) for m in METHODS}
        if not per["rr"]:
            continue
        names = list(_by_setting(per["rr"])) + [
            f"tier{t}/pooled" for t in sorted(set(tiers.values()))
        ]
        fig, axes = _grid(plt, len(names))
        for ax, name in zip(axes, names, strict=True):
            for i, m in enumerate(METHODS):
                sub = _subset(per[m], name, tiers)
                if not sub:
                    continue
                curve = stats.threshold_sweep(
                    [o.value for o in sub],
                    [o.dscore for o in sub],
                    sweep,
                    method="cka" if m == "cka" else "rr",
                )
                ax.plot(
                    [t for t, _ in curve],
                    [a for _, a in curve],
                    color=SERIES[i],
                    linestyle=LINESTYLES[i],
                    label=METHOD_NAMES[m],
                )
            ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
            ax.axvline(THRESHOLDS[bench], color=AXIS, linewidth=0.8)
            if not any(np.isfinite(line.get_ydata()).any() for line in ax.get_lines()[:-2]):
                ax.text(
                    0.5, 0.5, "n/a (degenerate)", transform=ax.transAxes, ha="center", color=MUTED
                )
            ax.set_xlim(min(sweep), max(sweep))
            ax.set_ylim(0, 1)
            ax.set_title(name)
            ax.set_xlabel("Δ score threshold")
            ax.set_ylabel("ROC AUC")
        _legend(fig, axes[0], ncol=4)
        fig.suptitle(f"F-R2 · {BENCH_NAMES[bench]}: AUC across thresholds (seed 0)", color=INK)
        paths.append(_save(fig, Path(out_dir) / f"f_r2_thresholds_{bench}.png"))
    return paths


def _subset(obs: Sequence[Any], name: str, tiers: Mapping[str, int]) -> list[Any]:
    if name.startswith("tier"):
        tier = int(name[4:].split("/")[0])
        return [o for o in obs if o.seed == 0 and tiers.get(o.model) == tier]
    p = parse_setting(name)
    return [o for o in obs if (o.model, o.regime, o.seed) == p]


def anchor_curves(
    t1: Sequence[Mapping[str, Any]],
    ablations: Sequence[Mapping[str, Any]],
    settings: Sequence[str],
    out_dir: Path,
) -> list[Path]:
    """F-R3 (F4 analogue): RR AUC against the number of anchors for the four sources; the
    line is the mean over the 5 subset seeds, the band their min-max; 1,000 = the full
    pool (neutral: the primary cell). One file per benchmark."""
    plt = _plt()
    by = {
        (r["cell"], r["setting"], r["bench"]): num(r["roc_auc"])
        for r in ablations
        if r["method"] == "rr"
    }
    prim = {(r["setting"], r["bench"]): num(r["roc_auc"]) for r in t1 if r["method"] == "rr"}
    paths = []
    for bench in BENCH_NAMES:
        fig, axes = _grid(plt, len(settings))
        for ax, name in zip(axes, settings, strict=True):
            for i, src in enumerate(SOURCES):
                mean, lo, hi = [], [], []
                for size in ANCHOR_SIZES:
                    if size == 1000:
                        v = [
                            prim.get((name, bench))
                            if src == "neutral"
                            else by.get((f"anchors={src}/all", name, bench))
                        ]
                    else:
                        v = [by.get((f"anchors={src}/{size}/s{s}", name, bench)) for s in range(5)]
                    v = [x for x in v if x is not None and not math.isnan(x)]
                    mean.append(np.mean(v) if v else np.nan)
                    lo.append(min(v) if v else np.nan)
                    hi.append(max(v) if v else np.nan)
                ax.plot(
                    ANCHOR_SIZES,
                    mean,
                    color=SERIES[i],
                    linestyle=LINESTYLES[i],
                    marker="o",
                    markersize=4,
                    label=src,
                )
                ax.fill_between(ANCHOR_SIZES, lo, hi, color=SERIES[i], alpha=0.15, linewidth=0)
            ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
            if not any(
                line.get_ydata().size and np.isfinite(line.get_ydata()).any()
                for line in ax.get_lines()[:-1]
            ):
                # degenerate labels at this threshold: no AUC for any anchor cell
                ax.text(
                    0.5, 0.5, "n/a (degenerate)", transform=ax.transAxes, ha="center", color=MUTED
                )
                ax.set_title(name)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                continue
            ax.set_xscale("log")
            ax.set_xticks(ANCHOR_SIZES, [str(s) for s in ANCHOR_SIZES])
            ax.set_ylim(0, 1)
            ax.set_title(name)
            ax.set_xlabel("anchors")
            ax.set_ylabel("ROC AUC (RR)")
        _legend(fig, axes[0], ncol=4, title="anchor source")
        fig.suptitle(
            f"F-R3 · {BENCH_NAMES[bench]}: AUC vs anchor count (band: 5 subset seeds)", color=INK
        )
        paths.append(_save(fig, Path(out_dir) / f"f_r3_anchors_{bench}.png"))
    return paths


def seed_groups(db: Sequence[Mapping[str, Any]], out_dir: Path) -> list[Path]:
    """F-R4 (App. E.5 analogue): per-group RR ΔB of the unharmful (a100) and harmful (a000)
    endpoints, one marker colour per training seed, for every multi-seed spectrum."""
    plt = _plt()
    rows = [
        r
        for r in db
        if r["cell"] == "primary" and r["method"] == "rr" and r["ckpt"] in ("a100", "a000")
    ]
    spectra: dict[tuple[str, str], set[int]] = {}
    for r in rows:
        spectra.setdefault((r["model"], r["regime"]), set()).add(int(r["seed"]))
    paths = []
    for (model, regime), seeds in sorted(spectra.items()):
        if len(seeds) < 2:
            continue
        groups = sorted({r["group"] for r in rows if r["model"] == model})
        fig, axes = _grid(plt, 2, w=7.0, h=3.2)
        for ax, ckpt in zip(axes, ("a100", "a000"), strict=True):
            for i, s in enumerate(sorted(seeds)):
                v = {
                    r["group"]: r["delta_b"]
                    for r in rows
                    if (r["model"], r["regime"], int(r["seed"]), r["ckpt"])
                    == (model, regime, s, ckpt)
                }
                ax.scatter(
                    range(len(groups)),
                    [v.get(g, np.nan) for g in groups],
                    s=22,
                    color=SERIES[i],
                    label=f"seed {s}",
                    edgecolors=SURFACE,
                    linewidths=0.6,
                    zorder=3,
                )
            ax.set_xticks(range(len(groups)), groups, rotation=70, ha="right", fontsize=7)
            ax.set_title(
                f"{model} {regime} · {ckpt} ({'unharmful' if ckpt == 'a100' else 'harmful'})"
            )
            ax.set_ylabel("ΔB (RR)")
        _legend(fig, axes[0], ncol=len(seeds))
        fig.suptitle("F-R4 · per-group ΔB across training seeds", color=INK)
        paths.append(_save(fig, Path(out_dir) / f"f_r4_seeds_{model}_{regime}.png"))
    return paths


def roc_curves(obs_rr: Mapping[str, Sequence[Any]], out_dir: Path) -> list[Path]:
    """F-R5 (ROC panels of F7/8/11/12/15/16): RR ROC curve per seed-0 spectrum (label =
    Δscore > threshold, score = −ΔB), one panel per benchmark; degenerate spectra are left out."""
    from sklearn.metrics import roc_curve

    plt = _plt()
    benches = [b for b in BENCH_NAMES if obs_rr.get(b)]
    fig, axes = _grid(plt, len(benches), w=3.8, h=3.6)
    for ax, bench in zip(axes, benches, strict=True):
        for i, (name, os_) in enumerate(_by_setting(obs_rr[bench]).items()):
            y = np.array([o.dscore > THRESHOLDS[bench] for o in os_])
            if min(y.sum(), (~y).sum()) < stats.MIN_CLASS:
                continue
            score = -np.array([o.value for o in os_])
            fpr, tpr, _ = roc_curve(y, score)
            auc = stats.roc_auc(score, y)
            ax.plot(
                fpr,
                tpr,
                color=SERIES[i % len(SERIES)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                label=f"{name} ({auc:.2f})",
            )
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(BENCH_NAMES[bench])
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(loc="lower right", fontsize=6.5, framealpha=0.85, frameon=True, edgecolor=SURFACE)
    fig.suptitle("F-R5 · ROC of the RR detector (AUC in brackets), seed 0", color=INK)
    return [_save(fig, Path(out_dir) / "f_r5_roc.png")]
