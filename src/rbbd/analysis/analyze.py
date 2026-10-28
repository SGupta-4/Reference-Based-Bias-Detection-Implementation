"""M6 analysis: join ΔB with the benchmark deltas and compute every T1 / C3 / F3 / ablation /
seed number (PLAN M6-T1..T5; D-027, D-028, D-097).

Inputs are the per-run aggregates already in `results/<run>/` (DC-18: no text):
- `delta_b_cells.csv.gz` (M4 `deltab` stage): one row per (spectrum, checkpoint, cell,
  method, group) with ΔB;
- `bench_scores.csv` (M5 `score` stage): one row per (spectrum, checkpoint, benchmark,
  unit) with Δscore = score_aud − score_ref and its paired-bootstrap CI;
- optional: `bench_scores_wgm_qwen.csv` (WGM rescored under the Qwen topic map, D-095)
  and `delta_b_decomposition.csv.gz` (target-only / attribute-only RR, D-085), which
  `analysis.wgm_maps` and `analysis.decompose` write in the same row formats.

The analysis keys its outputs on the content hashes of those files (`manifest.json`), not
on the runner's config hash, which changed after M4/M5 for reasons unrelated to these
inputs (D-097). A second call with unchanged inputs logs `cache hit` and does nothing
(DC-10).

Observations (D-027): every audited checkpoint (the reference is excluded, ΔB ≡ 0) ×
target group of one spectrum. DT and ToxiGen pair by group; WGM pairs each group with
its topic's score (`group_from_topic`, D-024), and its bootstrap resamples (checkpoint,
topic) clusters because groups in a topic share one score. Label = Δscore > threshold
(0.1 DT/WGM, 0.03 ToxiGen); score = −ΔB, so a larger bias increase should come with a
more negative ΔB (§3.4). CKA drift is undirected (T7).

Called by `cli analyze` and by the runner's `analyze` stage (one run).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rbbd.analysis import stats
from rbbd.utils.logging import get_logger

log = get_logger(__name__)

THRESHOLDS = {"dt": 0.1, "wgm": 0.1, "toxigen": 0.03}  # D-027
SWEEP = tuple(round(t, 2) for t in np.arange(0.0, 0.305, 0.01))  # F3 analogue grid
PRIMARY = "primary"
DIRECTED = ("rr", "seat", "procrustes", "rr_target", "rr_attr")
ANALYSIS_SCHEMA = 2  # 2: CKA CIs keep the full-sample orientation; no CI for a nan point (D-099)
N_BOOT_ABLATION = 200

DELTA_B_FILE = "delta_b_cells.csv.gz"
BENCH_FILE = "bench_scores.csv"
BENCH_QWEN_FILE = "bench_scores_wgm_qwen.csv"
DECOMP_FILE = "delta_b_decomposition.csv.gz"
NF4_RUN = "m6_nf4_llama3.1-8b"
NF4_BASE_RUN = "tier1_llama3.1-8b"


@dataclass(frozen=True)
class Obs:
    """One (audited checkpoint, unit) observation of one spectrum and benchmark."""

    model: str
    regime: str
    seed: int
    ckpt: str
    unit: str  # group, or topic for topic-mean pairing
    topic: str
    value: float  # ΔB (or CKA drift)
    dscore: float  # ΔBiasScore / ΔToxicity


# --------------------------------------------------------------------------- reading


def _open(path: Path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else open(path, newline="")


def _num(v: str) -> float:
    return float(v) if v not in ("", None) else math.nan


def read_delta_b(path: Path) -> list[dict[str, Any]]:
    """Rows of a `delta_b*.csv[.gz]` with numeric seed / alpha / delta_b."""
    with _open(Path(path)) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["seed"] = int(float(r["seed"])) if r["seed"] not in ("", None) else -1
        r["alpha"] = _num(r["alpha"])
        r["delta_b"] = _num(r["delta_b"])
    return rows


def read_bench(path: Path) -> list[dict[str, Any]]:
    """Rows of a `bench_scores*.csv` with numeric seed / score / delta."""
    with _open(Path(path)) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["seed"] = int(float(r["seed"])) if r["seed"] not in ("", None) else -1
        for k in ("score", "delta", "delta_lo", "delta_hi", "ref_score"):
            r[k] = _num(r.get(k, ""))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- joining


def spectrum_key(r: Mapping[str, Any]) -> tuple[str, str, int]:
    return (r["model"], r["regime"], int(r["seed"]))


def observations(
    db_rows: Iterable[Mapping[str, Any]],
    bench_rows: Iterable[Mapping[str, Any]],
    bench: str,
    method: str,
    cell: str = PRIMARY,
    pairing: str = "group",
) -> list[Obs]:
    """Join ΔB rows (`cell`, `method`) with one benchmark's rows; reference excluded.

    pairing "group": DT/ToxiGen units are groups; WGM uses `group_from_topic` rows.
    pairing "topic": WGM only, ΔB averaged over the groups of each topic (D-024
    sensitivity), paired with the topic score.
    """
    if pairing not in ("group", "topic"):
        raise ValueError(pairing)
    if pairing == "topic" and bench != "wgm":
        raise ValueError("topic-mean pairing applies to WGM only")
    unit_type = {"group": "group_from_topic" if bench == "wgm" else "group", "topic": "topic"}[
        pairing
    ]
    scores: dict[tuple, float] = {}
    for r in bench_rows:
        if r["bench"] == bench and r["unit_type"] == unit_type and r["ckpt"] != "ref":
            scores[(*spectrum_key(r), r["ckpt"], r["unit"])] = r["delta"]
    values: dict[tuple, list[float]] = defaultdict(list)
    topic_of: dict[str, str] = {}
    for r in db_rows:
        if r["cell"] != cell or r["method"] != method or r["ckpt"] == "ref":
            continue
        unit = r["group"] if pairing == "group" else r["topic"]
        topic_of[r["group"]] = r["topic"]
        values[(*spectrum_key(r), r["ckpt"], unit)].append(r["delta_b"])
    out = []
    for key, vals in sorted(values.items()):
        if key in scores and not math.isnan(scores[key]):
            model, regime, seed, ckpt, unit = key
            topic = unit if pairing == "topic" else topic_of[unit]
            out.append(
                Obs(model, regime, seed, ckpt, unit, topic, float(np.mean(vals)), scores[key])
            )
    return out


def clusters_for(obs: Sequence[Obs], bench: str) -> list[str] | None:
    """WGM: (checkpoint, topic) clusters (groups of a topic share one score); else rows."""
    if bench != "wgm":
        return None
    return [f"{o.model}/{o.regime}/{o.seed}/{o.ckpt}/{o.topic}" for o in obs]


# --------------------------------------------------------------------------- settings


def settings(obs: Sequence[Obs], tiers: Mapping[str, int]) -> dict[str, list[Obs]]:
    """{setting name: observations}: each spectrum, each tier pooled (seed 0), and all."""
    out: dict[str, list[Obs]] = defaultdict(list)
    for o in obs:
        out[f"{o.model}/{o.regime}-s{o.seed}"].append(o)
        if o.seed == 0:
            tier = tiers.get(o.model)
            if tier is not None:
                out[f"tier{tier}/pooled"].append(o)
            out["all/pooled"].append(o)
    return dict(out)


def _row_stats(
    obs: Sequence[Obs], bench: str, method: str, n_boot: int, seed: int, full: bool = True
) -> dict[str, Any]:
    x = [o.value for o in obs]
    y = [o.dscore for o in obs]
    thr = THRESHOLDS[bench]
    m = "cka" if method == "cka" else "rr"
    if full:
        s = stats.summarize(
            x, y, thr, method=m, n_boot=n_boot, seed=seed, clusters=clusters_for(obs, bench)
        )
        return {
            "n_obs": s["n_obs"],
            "n_pos": s["n_pos"],
            "n_neg": s["n_neg"],
            "pearson_r": s["pearson_r"],
            "pearson_p": s["pearson_p"],
            "stars": s["stars"],
            "r_lo": s["pearson_ci"]["lo"],
            "r_hi": s["pearson_ci"]["hi"],
            "roc_auc": s["roc_auc"],
            "auc_lo": s["roc_auc_ci"]["lo"],
            "auc_hi": s["roc_auc_ci"]["hi"],
            "mae_in": s["mae_in"],
            "mae_oob": s["mae_oob"],
        }
    r, p = stats.pearson(x, y)
    if m == "cka":
        r = abs(r)
    labels = np.asarray(y) > thr
    return {
        "n_obs": len(x),
        "n_pos": int(labels.sum()),
        "n_neg": int((~labels).sum()),
        "pearson_r": r,
        "pearson_p": p,
        "stars": stats.stars(p),
        "roc_auc": stats.detection_auc(x, y, thr, method=m),
        **{
            k: v
            for k, v in stats.bootstrap_mae(x, y, n_boot=n_boot, seed=seed).items()
            if k != "n_boot"
        },
    }


# --------------------------------------------------------------------------- tables


def t1_table(
    db: Sequence[Mapping[str, Any]],
    bench_rows: Sequence[Mapping[str, Any]],
    tiers: Mapping[str, int],
    methods: Sequence[str],
    benches: Sequence[str],
    map_name: str = "human",
    n_boot: int = stats.N_BOOT,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """T1 (RR) and T7 (baselines): one row per (setting, bench, method)."""
    rows = []
    for bench in benches:
        for method in methods:
            obs = observations(db, bench_rows, bench, method)
            for name, sub in sorted(settings(obs, tiers).items()):
                rows.append(
                    {
                        "setting": name,
                        "bench": bench,
                        "method": method,
                        "map": map_name,
                        **_row_stats(sub, bench, method, n_boot, seed),
                    }
                )
    return rows


def c3_table(
    db: Sequence[Mapping[str, Any]],
    bench_rows: Sequence[Mapping[str, Any]],
    tiers: Mapping[str, int],
    benches: Sequence[str],
    against: Sequence[str] = ("seat", "procrustes", "cka"),
    n_boot: int = stats.N_BOOT,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Paired bootstrap AUC(RR) − AUC(baseline) on identical observations (criterion C3)."""
    rows = []
    for bench in benches:
        rr_obs = {
            (o.model, o.regime, o.seed, o.ckpt, o.unit): o
            for o in observations(db, bench_rows, bench, "rr")
        }
        for base in against:
            b_obs = {
                (o.model, o.regime, o.seed, o.ckpt, o.unit): o
                for o in observations(db, bench_rows, bench, base)
            }
            paired = [rr_obs[k] for k in sorted(rr_obs) if k in b_obs]
            by_setting = settings(paired, tiers)
            for name, sub in sorted(by_setting.items()):
                keys = [(o.model, o.regime, o.seed, o.ckpt, o.unit) for o in sub]
                d = stats.paired_auc_difference(
                    [rr_obs[k].value for k in keys],
                    [b_obs[k].value for k in keys],
                    [rr_obs[k].dscore for k in keys],
                    THRESHOLDS[bench],
                    method_a="rr",
                    method_b="cka" if base == "cka" else "seat",
                    n_boot=n_boot,
                    seed=seed,
                    clusters=clusters_for(sub, bench),
                )
                rows.append(
                    {"setting": name, "bench": bench, "a": "rr", "b": base, "n_obs": len(keys), **d}
                )
    return rows


def sweep_table(
    db: Sequence[Mapping[str, Any]],
    bench_rows: Sequence[Mapping[str, Any]],
    tiers: Mapping[str, int],
    benches: Sequence[str],
    methods: Sequence[str] = ("rr", "seat"),
) -> list[dict[str, Any]]:
    """F3 analogue: AUC per detection threshold, every setting (nan = a class < 5 rows)."""
    rows = []
    for bench in benches:
        for method in methods:
            obs = observations(db, bench_rows, bench, method)
            for name, sub in sorted(settings(obs, tiers).items()):
                x, y = [o.value for o in sub], [o.dscore for o in sub]
                m = "cka" if method == "cka" else "rr"
                for t, auc in stats.threshold_sweep(x, y, SWEEP, method=m):
                    rows.append(
                        {
                            "setting": name,
                            "bench": bench,
                            "method": method,
                            "threshold": t,
                            "roc_auc": auc,
                        }
                    )
    return rows


def ablation_table(
    db: Sequence[Mapping[str, Any]],
    bench_rows: Sequence[Mapping[str, Any]],
    tiers: Mapping[str, int],
    benches: Sequence[str],
    n_boot: int = N_BOOT_ABLATION,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """F4 / T5a-b / T6 / D-017: point estimates per (cell, method, setting, bench).

    Non-primary cells change one factor (anchor source × size × seed, attribute or
    target variant, pooling, layer site); CIs are left to the primary cell (T1).
    """
    cells = sorted({(r["cell"], r["method"]) for r in db if r["cell"] != PRIMARY})
    rows = []
    for bench in benches:
        for cell, method in cells:
            obs = observations(db, bench_rows, bench, method, cell=cell)
            for name, sub in sorted(settings(obs, tiers).items()):
                rows.append(
                    {
                        "cell": cell,
                        "setting": name,
                        "bench": bench,
                        "method": method,
                        **_row_stats(sub, bench, method, n_boot, seed, full=False),
                    }
                )
    return rows


def wgm_pairing_table(
    db: Sequence[Mapping[str, Any]],
    maps: Mapping[str, Sequence[Mapping[str, Any]]],
    tiers: Mapping[str, int],
    methods: Sequence[str] = ("rr", "seat", "procrustes"),
    n_boot: int = stats.N_BOOT,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """WGM under each topic map (human, Qwen; D-095) × pairing (group-inherits-topic,
    topic-mean ΔB; D-024 sensitivity)."""
    rows = []
    for map_name, bench_rows in maps.items():
        for pairing in ("group", "topic"):
            for method in methods:
                obs = observations(db, bench_rows, "wgm", method, pairing=pairing)
                for name, sub in sorted(settings(obs, tiers).items()):
                    rows.append(
                        {
                            "map": map_name,
                            "pairing": pairing,
                            "setting": name,
                            "method": method,
                            **_row_stats(sub, "wgm", method, n_boot, seed),
                        }
                    )
    return rows


def sign_table(
    db: Sequence[Mapping[str, Any]],
    bench_rows: Sequence[Mapping[str, Any]],
    benches: Sequence[str],
    methods: Sequence[str] = ("rr", "seat", "procrustes", "rr_target", "rr_attr"),
    n_boot: int = stats.N_BOOT,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Per spectrum: harmful − unharmful (a000 − a100) for every ΔB method and every
    benchmark, with 95% bootstrap CIs over groups / units (D-085, D-094, D-096).

    ΔB: negative = h more biased; Δscore: positive = h more biased. `agree` is True when
    a method's sign matches the benchmark's (h − u ΔB < 0 and h − u score > 0, or both
    reversed)."""
    rng_seed = seed

    def contrast(vals_h: dict[str, float], vals_u: dict[str, float]) -> tuple[float, float, float]:
        units = sorted(set(vals_h) & set(vals_u))
        d = np.array([vals_h[u] - vals_u[u] for u in units])
        if not len(d):
            return math.nan, math.nan, math.nan
        rng = np.random.default_rng(rng_seed)
        boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(d.mean()), float(lo), float(hi)

    spectra = sorted({spectrum_key(r) for r in db if r["ckpt"] != "ref"})
    out = []
    for sp in spectra:
        db_sp = [r for r in db if spectrum_key(r) == sp and r["cell"] == PRIMARY]
        row: dict[str, Any] = {"spectrum": f"{sp[0]}/{sp[1]}-s{sp[2]}"}
        for method in methods:
            by = {
                c: {
                    r["group"]: r["delta_b"]
                    for r in db_sp
                    if r["method"] == method and r["ckpt"] == c
                }
                for c in ("a000", "a100")
            }
            row[f"db_{method}"], row[f"db_{method}_lo"], row[f"db_{method}_hi"] = contrast(
                by["a000"], by["a100"]
            )
        for bench in benches:
            ut = "topic" if bench == "wgm" else "group"
            by = {
                c: {
                    r["unit"]: r["score"]
                    for r in bench_rows
                    if spectrum_key(r) == sp
                    and r["bench"] == bench
                    and r["unit_type"] == ut
                    and r["ckpt"] == c
                }
                for c in ("a000", "a100")
            }
            row[f"score_{bench}"], row[f"score_{bench}_lo"], row[f"score_{bench}_hi"] = contrast(
                by["a000"], by["a100"]
            )
        for method in methods:
            for bench in benches:
                a, b = row[f"db_{method}"], row[f"score_{bench}"]
                row[f"agree_{method}_{bench}"] = (
                    None if math.isnan(a) or math.isnan(b) else bool((a < 0) == (b > 0))
                )
        out.append(row)
    return out


def seeds_table(db: Sequence[Mapping[str, Any]], min_seeds: int = 2) -> list[dict[str, Any]]:
    """App. E.5 analogue: per (model, regime, checkpoint, method), the across-seed std of the
    mean ΔB over groups, and the mean across groups of the per-group across-seed std."""
    by: dict[tuple, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in db:
        if r["cell"] == PRIMARY and r["ckpt"] != "ref":
            by[(r["model"], r["regime"], r["ckpt"], r["method"])][r["seed"]][r["group"]] = r[
                "delta_b"
            ]
    out = []
    for (model, regime, ckpt, method), per_seed in sorted(by.items()):
        if len(per_seed) < min_seeds:
            continue
        groups = sorted(set.intersection(*(set(v) for v in per_seed.values())))
        means = [float(np.mean([v[g] for g in groups])) for v in per_seed.values()]
        per_group_std = [float(np.std([v[g] for v in per_seed.values()], ddof=1)) for g in groups]
        out.append(
            {
                "model": model,
                "regime": regime,
                "ckpt": ckpt,
                "method": method,
                "n_seeds": len(per_seed),
                "seeds": ",".join(map(str, sorted(per_seed))),
                "mean_db": float(np.mean(means)),
                "std_of_mean_db": float(np.std(means, ddof=1)),
                "mean_group_std": float(np.mean(per_group_std)),
            }
        )
    return out


def nf4_table(
    fp16: Sequence[Mapping[str, Any]], nf4: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """M6-T6 / D-004: per (checkpoint, method), |ΔB_fp16 − ΔB_nf4| over groups (mean, max),
    their correlation, and whether the h − u sign is the same in both precisions."""

    def index(rows):
        return {
            (r["ckpt"], r["method"], r["group"]): r["delta_b"]
            for r in rows
            if r["cell"] == PRIMARY and r["ckpt"] != "ref"
        }

    a, b = index(fp16), index(nf4)
    keys = sorted(set(a) & set(b))
    out = []
    for ckpt, method in sorted({(k[0], k[1]) for k in keys}):
        ks = [k for k in keys if k[:2] == (ckpt, method)]
        x, y = np.array([a[k] for k in ks]), np.array([b[k] for k in ks])
        r, _ = stats.pearson(x, y)
        out.append(
            {
                "ckpt": ckpt,
                "method": method,
                "n_groups": len(ks),
                "mean_abs_diff": float(np.abs(x - y).mean()),
                "max_abs_diff": float(np.abs(x - y).max()),
                "mean_db_fp16": float(x.mean()),
                "mean_db_nf4": float(y.mean()),
                "pearson_r": r,
            }
        )
    for method in sorted({k[1] for k in keys}):

        def hu(idx, method=method):
            h = [v for (c, m, _), v in idx.items() if c == "a000" and m == method]
            u = [v for (c, m, _), v in idx.items() if c == "a100" and m == method]
            return float(np.mean(h) - np.mean(u)) if h and u else math.nan

        ha, hb = hu(a), hu(b)
        out.append(
            {
                "ckpt": "h-u",
                "method": method,
                "mean_db_fp16": ha,
                "mean_db_nf4": hb,
                "same_sign": None
                if math.isnan(ha) or math.isnan(hb)
                else bool((ha < 0) == (hb < 0)),
            }
        )
    return out


# --------------------------------------------------------------------------- driver


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(
            {
                k: ("" if v is None else (round(v, 6) if isinstance(v, float) else v))
                for k, v in r.items()
            }
        )
    path.write_text(buf.getvalue())


def discover(results_root: Path, runs: Sequence[str] | None = None) -> dict[str, dict[str, Path]]:
    """{run: {kind: path}} for every run directory holding both ΔB and benchmark tables."""
    out = {}
    for d in sorted(Path(results_root).iterdir()):
        if not d.is_dir() or (runs is not None and d.name not in runs):
            continue
        files = {
            k: d / f
            for k, f in (
                ("delta_b", DELTA_B_FILE),
                ("bench", BENCH_FILE),
                ("bench_qwen", BENCH_QWEN_FILE),
                ("decomp", DECOMP_FILE),
            )
        }
        files = {k: p for k, p in files.items() if p.exists()}
        if "delta_b" in files and "bench" in files:
            out[d.name] = files
    nf4 = Path(results_root) / NF4_RUN / DELTA_B_FILE
    if nf4.exists() and (runs is None or NF4_RUN in runs):
        out.setdefault(NF4_RUN, {})["delta_b"] = nf4
    return out


def run_analysis(
    results_root: Path,
    out_dir: Path,
    runs: Sequence[str] | None = None,
    tiers: Mapping[str, int] | None = None,
    n_boot: int = stats.N_BOOT,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Every M6 table for the runs under `results_root`; skipped when the inputs' hashes,
    `n_boot`, `seed` and the schema match the existing `manifest.json` (DC-10)."""
    found = discover(Path(results_root), runs)
    inputs = {f"{run}/{p.name}": sha256_file(p) for run, fs in found.items() for p in fs.values()}
    manifest = {"schema": ANALYSIS_SCHEMA, "n_boot": n_boot, "seed": seed, "inputs": inputs}
    mpath = Path(out_dir) / "manifest.json"
    if not force and mpath.exists() and json.loads(mpath.read_text()) == manifest:
        log.info("cache hit: analysis %s", out_dir)
        return {"cache_hit": True, **json.loads((Path(out_dir) / "summary.json").read_text())}
    benches = ("dt", "toxigen", "wgm")
    db: list[dict[str, Any]] = []
    bench: list[dict[str, Any]] = []
    bench_qwen: list[dict[str, Any]] = []
    for run, fs in found.items():
        if run == NF4_RUN or "bench" not in fs:
            continue
        db += read_delta_b(fs["delta_b"])
        if "decomp" in fs:
            db += read_delta_b(fs["decomp"])
        bench += read_bench(fs["bench"])
        if "bench_qwen" in fs:
            bench_qwen += read_bench(fs["bench_qwen"])
    tiers = dict(tiers or {})
    order = {
        m: i for i, m in enumerate(("rr", "seat", "procrustes", "cka", "rr_target", "rr_attr"))
    }
    methods = sorted(
        {r["method"] for r in db if r["cell"] == PRIMARY},
        key=lambda m: (order.get(m, len(order)), m),
    )
    out = Path(out_dir)
    tables: dict[str, list[dict[str, Any]]] = {
        "t1": t1_table(db, bench, tiers, methods, benches, "human", n_boot, seed),
        "c3": c3_table(db, bench, tiers, benches, n_boot=n_boot, seed=seed),
        "f3_sweep": sweep_table(db, bench, tiers, benches),
        "ablations": ablation_table(db, bench, tiers, benches, seed=seed),
        "wgm_pairing": wgm_pairing_table(
            db,
            {"human": bench, **({"qwen": bench_qwen} if bench_qwen else {})},
            tiers,
            n_boot=n_boot,
            seed=seed,
        ),
        "sign": sign_table(
            db,
            bench,
            benches,
            [m for m in ("rr", "seat", "procrustes", "rr_target", "rr_attr") if m in methods],
            n_boot=n_boot,
            seed=seed,
        ),
        "seeds": seeds_table(db),
    }
    if NF4_RUN in found and NF4_BASE_RUN in found:
        tables["nf4"] = nf4_table(
            read_delta_b(found[NF4_BASE_RUN]["delta_b"]), read_delta_b(found[NF4_RUN]["delta_b"])
        )
    for name, rows in tables.items():
        _write(out / f"{name}.csv", rows)
    summary = {
        "runs": sorted(found),
        "methods": methods,
        "rows": {k: len(v) for k, v in tables.items()},
        "t1_rr": [
            {
                k: r[k]
                for k in (
                    "setting",
                    "bench",
                    "n_obs",
                    "pearson_r",
                    "stars",
                    "roc_auc",
                    "auc_lo",
                    "auc_hi",
                    "mae_in",
                )
            }
            for r in tables["t1"]
            if r["method"] == "rr"
        ],
        "sign": tables["sign"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    mpath.write_text(json.dumps(manifest, indent=1))
    return {"cache_hit": False, **summary}


def stage(ctx: Any) -> Any:
    """Runner `analyze` stage: the tables for this config's own `results/<run>/`."""
    from rbbd.metrics.delta_b import results_dir
    from rbbd.runner import StageResult

    res = results_dir(ctx.cfg, Path(ctx.env.artifacts_root))
    tiers = {m["slug"]: int(ctx.cfg.get("tier", 0)) for m in ctx.cfg["models"]}
    run_analysis(
        res.parent,
        ctx.stage_dir,
        runs=[res.name],
        tiers=tiers,
        n_boot=int(ctx.cfg.get("analysis.n_boot", stats.N_BOOT)),
        seed=int(ctx.cfg.get("analysis.seed", 0)),
    )
    root = Path(ctx.env.artifacts_root)
    return StageResult(
        outputs={p.stem: str(p.relative_to(root)) for p in sorted(ctx.stage_dir.glob("*.csv"))}
    )
