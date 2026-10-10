"""B-030 diagnostics: why the harmful endpoint is not below the unharmful one in mean ΔB
for most spectra (M4-T5; D-083). CPU only, from cached embeddings and the sentence union.

Called by `cli diagnose-deltab --config C`, which reads the config's `embed/<run_key>/
index.json` and `sentences/<run_key>/union.json` and writes
`results/<run>/b030_diagnostics.json`. Primary cell only: mean pooling, post-norm,
neutral anchors, base attribute and target sentences. All outputs are aggregates over
sentences, never text (DC-18).

Per spectrum (`diagnose_spectrum`):
- `methods`: mean ΔB over groups for RR, SEAT and Procrustes-SEAT at every checkpoint;
  the three share no scoring code, so a code-level sign error would show as disagreement.
- `alpha_trend`: Spearman ρ between α and mean ΔB (RR, SEAT); positive = more unharmful
  weight, higher B, as App. E.5 implies.
- `contrast`: per target sentence, b(a000) − b(a100) (RR and SEAT), averaged; 95% CI from
  a bootstrap over the 50 target templates (a template's sentences for all groups move
  together); `groups_h_below_u` counts groups whose mean contrast is negative.
- `decomposition`: mean change of S⁺ and S⁻ (RR, Eq. 5) at a100 and a000 vs the
  reference: does B move because targets approach the negatives or leave the positives?
- `control`: the same B shift for a group-free control set (the Alpaca pool, when the
  union has it) scored against the same anchors and attributes. A shift shared by
  control and targets is generic fine-tuning drift, not group bias; `group_specific` =
  mean target ΔB − control ΔB.
- `geometry`: cosine of the P and N centroids (valence separability) and the mean
  pairwise cosine of all union sentences (anisotropy) for ref, a100 and a000.
- `per_group`: RR ΔB at a100 and a000 per group.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from rbbd.metrics import rr, seat
from rbbd.metrics.delta_b import EmbeddingSet, from_union
from rbbd.metrics.procrustes import fit_orthogonal

N_BOOT = 2000


def _rows(
    es: EmbeddingSet,
    method: str,
    rotation: np.ndarray | None = None,
    ref: EmbeddingSet | None = None,
) -> np.ndarray:
    """Per-target S⁺ − S⁻ [n] for `method`; Procrustes = SEAT of rotated targets against
    the reference attributes."""
    if method == "rr":
        rel = lambda x: rr.relative(x, es.anchors)  # noqa: E731
        return rr.row_bias_rel(rel(es.targets), rel(es.positives), rel(es.negatives))
    if method == "seat":
        return seat.row_bias_seat(es.targets, es.positives, es.negatives)
    assert rotation is not None and ref is not None
    return seat.row_bias_seat(es.targets @ rotation, ref.positives, ref.negatives)


def _assoc_rr(es: EmbeddingSet, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S⁺, S⁻) per row of `targets` [n, d] in `es`'s relative space (Eq. 5)."""
    r_t = rr.relative(targets, es.anchors)
    return (
        rr.association_rel(r_t, rr.relative(es.positives, es.anchors)),
        rr.association_rel(r_t, rr.relative(es.negatives, es.anchors)),
    )


def _mean_pairwise_cos(x: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    idx = np.random.default_rng(seed).choice(len(x), size=min(n, len(x)), replace=False)
    u = rr.unit_rows(x[idx])
    g = u @ u.T
    return float((g.sum() - np.trace(g)) / (len(idx) * (len(idx) - 1)))


def _centroid_cos(a: np.ndarray, b: np.ndarray) -> float:
    ca, cb = a.mean(0), b.mean(0)
    return float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb)))


def template_bootstrap(
    values: np.ndarray, templates: np.ndarray, n_boot: int = N_BOOT, seed: int = 0
) -> tuple[float, float]:
    """95% percentile CI of mean(values) resampling whole templates (clusters)."""
    ids = np.unique(templates)
    sums = np.array([values[templates == t].sum() for t in ids])
    counts = np.array([(templates == t).sum() for t in ids])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def diagnose_spectrum(
    ref: Mapping[str, np.ndarray],
    entries: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
    union: Mapping[str, Any],
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """All B-030 diagnostics for one spectrum; `entries` = (index row, tensors) per audited
    checkpoint (`ckpt`, `alpha`), the reference excluded. Tensors use the "mean" entry."""
    ref_set = from_union(ref["mean"], dict(union))
    groups = np.asarray(ref_set.target_groups)
    n_per_group = {g: int((groups == g).sum()) for g in dict.fromkeys(ref_set.target_groups)}
    templates = np.concatenate([np.arange(n) for n in n_per_group.values()])
    base = {m: _rows(ref_set, m) for m in ("rr", "seat")}
    control_rows = union["index"].get("anchors/alpaca")
    control_ref = None
    if control_rows:
        control_ref = ref["mean"][control_rows].astype(np.float64)
        s_p, s_n = _assoc_rr(ref_set, control_ref)
        control_base = s_p - s_n
    out: dict[str, Any] = {"methods": {}, "decomposition": {}, "control": {}, "geometry": {}}
    s_ref = _assoc_rr(ref_set, ref_set.targets)
    out["geometry"]["ref"] = {
        "pn_centroid_cos": _centroid_cos(ref_set.positives, ref_set.negatives),
        "mean_pairwise_cos": _mean_pairwise_cos(ref["mean"].astype(np.float64)),
    }
    per_ckpt_rows: dict[str, dict[str, np.ndarray]] = {}
    for row, tensors in entries:
        ckpt = row["ckpt"]
        aud = from_union(tensors["mean"], dict(union))
        rot = fit_orthogonal(aud.anchors, ref_set.anchors)
        rows = {m: _rows(aud, m) for m in ("rr", "seat")}
        rows["procrustes"] = _rows(aud, "procrustes", rot, ref_set)
        per_ckpt_rows[ckpt] = rows
        delta = {
            "rr": rows["rr"] - base["rr"],
            "seat": rows["seat"] - base["seat"],
            "procrustes": rows["procrustes"] - base["seat"],
        }
        out["methods"][ckpt] = {
            "alpha": row["alpha"],
            **{m: float(v.mean()) for m, v in delta.items()},
        }
        if ckpt in ("a100", "a000"):
            s_p, s_n = _assoc_rr(aud, aud.targets)
            out["decomposition"][ckpt] = {
                "d_S_plus": float((s_p - s_ref[0]).mean()),
                "d_S_minus": float((s_n - s_ref[1]).mean()),
            }
            out["geometry"][ckpt] = {
                "pn_centroid_cos": _centroid_cos(aud.positives, aud.negatives),
                "mean_pairwise_cos": _mean_pairwise_cos(tensors["mean"].astype(np.float64)),
            }
            if control_ref is not None:
                c_aud = tensors["mean"][control_rows].astype(np.float64)
                c_p, c_n = _assoc_rr(aud, c_aud)
                shift = float(((c_p - c_n) - control_base).mean())
                out["control"][ckpt] = {
                    "control_d_b_rr": shift,
                    "target_d_b_rr": float(delta["rr"].mean()),
                    "group_specific": float(delta["rr"].mean()) - shift,
                }
    ordered = sorted((v["alpha"], v) for v in out["methods"].values())
    alphas = [a for a, _ in ordered]
    out["alpha_trend"] = {
        m: float(spearmanr(alphas, [v[m] for _, v in ordered]).statistic) for m in ("rr", "seat")
    }
    if {"a100", "a000"} <= set(per_ckpt_rows):
        out["contrast"] = {}
        for m in ("rr", "seat", "procrustes"):
            d = per_ckpt_rows["a000"][m] - per_ckpt_rows["a100"][m]
            lo, hi = template_bootstrap(d, templates, n_boot)
            by_group = {g: float(d[groups == g].mean()) for g in n_per_group}
            out["contrast"][m] = {
                "mean_h_minus_u": float(d.mean()),
                "ci95": [lo, hi],
                "groups_h_below_u": sum(v < 0 for v in by_group.values()),
                "groups": len(by_group),
            }
        out["per_group"] = {
            g: {
                "rr_u": float((per_ckpt_rows["a100"]["rr"] - base["rr"])[groups == g].mean()),
                "rr_h": float((per_ckpt_rows["a000"]["rr"] - base["rr"])[groups == g].mean()),
            }
            for g in n_per_group
        }
    return out


def diagnose_config(cfg: Any, artifacts_root: Any) -> dict[str, Any]:
    """`diagnose_spectrum` for every spectrum in the config's embed index."""
    import json
    from pathlib import Path

    from rbbd import runner
    from rbbd.utils import manifest as mf
    from rbbd.utils.cache import TensorCache

    root = Path(artifacts_root)
    keys = runner.stage_run_keys(cfg)

    def outputs(stage: str) -> list[str]:
        m = mf.read(mf.manifest_path(root, stage, keys[stage]))
        if m is None or not m.complete:
            raise FileNotFoundError(f"no complete {stage} manifest (restore manifests/ first)")
        return list(m.output_hashes)

    index = json.loads(
        (root / next(p for p in outputs("embed") if p.endswith("index.json"))).read_text()
    )
    union = json.loads(
        (root / next(p for p in outputs("sentences") if p.endswith("union.json"))).read_text()
    )
    cache = TensorCache(root)

    def read(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
        path = root / row["path"]
        tensors, _ = cache.read(str(path.parent.relative_to(root)), path.stem)
        return tensors

    report: dict[str, Any] = {"union_hash": union["union_hash"], "spectra": {}}
    refs = {r["model"]: r for r in index if r["ckpt"] == "ref"}
    spectra = sorted({(r["model"], r["regime"], r["seed"]) for r in index if r["ckpt"] != "ref"})
    for model, regime, seed in spectra:
        members = [
            r for r in index if (r["model"], r["regime"], r["seed"]) == (model, regime, seed)
        ]
        entries = [(r, read(r)) for r in members]
        report["spectra"][f"{model}/{regime}-s{seed}"] = diagnose_spectrum(
            read(refs[model]), entries, union
        )
    return report
