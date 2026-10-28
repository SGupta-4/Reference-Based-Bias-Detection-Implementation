"""Results tables R1-R7 and the pre-registered verdict (PLAN §1.2, §10; D-102). M7.

Every table is built from the M6 tables in `results/m6/` (D-097, D-101), the optional
per-run `timing.json` written by `analysis.timing` (R6) and the paper's own numbers in
`resources/paper_values.json` (transcribed from the PDF: T1, T3, T5-T7, App. E.5). Inputs
are the CSV rows as read (strings); every builder returns plain dict rows that
`markdown` renders and `analysis.report` also writes as CSV. CPU only.

Settings are named as in `analysis.analyze`: `<model>/<regime>-s<seed>` per spectrum,
`tier<k>/pooled` and `all/pooled`. Only seed-0 spectra have benchmark scores, so they are
the settings the verdict counts (PLAN §1.2).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from importlib import resources
from typing import Any

import numpy as np

BENCHES = ("wgm", "dt", "toxigen")
BENCH_NAMES = {"wgm": "WildGuardMix", "dt": "DecodingTrust", "toxigen": "ToxiGen"}
C3_BENCHES = ("wgm", "dt")  # §4.5 / App. H compare RR with SEAT on these two only
R6_RATIO_ORDER = ("wgm", "toxigen", "dt")  # the order the R6 ratio header names (B-041)
METHODS = ("rr", "seat", "procrustes", "cka")
METHOD_NAMES = {
    "rr": "RR",
    "seat": "SEAT",
    "procrustes": "Procrustes-SEAT",
    "cka": "CKA drift",
    "rr_target": "RR target-only",
    "rr_attr": "RR attribute-only",
}
ATTR_VARIANTS = ("subj_v1", "subj_v2", "syn_v1", "syn_v2", "syn_v3")
TARGET_VARIANTS = ("passive", "passive_rephr", "syn_v1", "syn_v2", "syn_v3")
ANCHOR_SOURCES = ("neutral", "word", "alpaca", "tulu")
S6_TOLERANCE = 2.0  # S6 "ΔB std ≈ 0.003" holds up to 2× the paper's std (D-102)


# --------------------------------------------------------------------------- helpers


def paper_values() -> dict[str, Any]:
    """The paper's numbers (T1, T3, T5-T7, E.5) from `resources/paper_values.json`."""
    return json.loads(resources.files("rbbd.resources").joinpath("paper_values.json").read_text())


def num(v: Any) -> float:
    """CSV cell → float; blank / None / unparsable → nan."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def fmt(v: Any, nd: int = 2) -> str:
    """Fixed-point text; nan → "n/a"."""
    x = num(v)
    return "n/a" if math.isnan(x) else f"{x:.{nd}f}"


def ci(lo: Any, hi: Any, nd: int = 2) -> str:
    return "" if math.isnan(num(lo)) else f"[{fmt(lo, nd)}, {fmt(hi, nd)}]"


def parse_setting(name: str) -> tuple[str, str, int] | None:
    """`model/regime-sN` → (model, regime, seed); pooled settings → None."""
    if name.endswith("/pooled"):
        return None
    model, rest = name.split("/", 1)
    regime, seed = rest.rsplit("-s", 1)
    return model, regime, int(seed)


def spectrum_settings(t1: Sequence[Mapping[str, Any]], seed: int = 0) -> list[str]:
    """Per-spectrum setting names of `seed`, sorted by (model, regime)."""
    names = {r["setting"] for r in t1}
    return sorted(n for n in names if (p := parse_setting(n)) is not None and p[2] == seed)


def _index(rows: Sequence[Mapping[str, Any]], *keys: str) -> dict[tuple, Mapping[str, Any]]:
    return {tuple(r[k] for k in keys): r for r in rows}


def _pstd(values: Sequence[float]) -> float:
    """Population std, as T5 reports (mean ± std over the six variants)."""
    return float(np.std(values)) if values else math.nan


def _share(k: int, n: int) -> str:
    return f"{k}/{n}"


def _outcome(k: int, n: int) -> str:
    """Secondary claims use the tier rule's thirds (D-102): ≥ 2/3 holds, ≤ 1/3 fails."""
    if n == 0:
        return "not testable"
    return "holds" if k >= 2 * n / 3 else "fails" if k <= n / 3 else "mixed"


def markdown(
    columns: Sequence[tuple[str, str | Callable[[Mapping[str, Any]], Any]]],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """GitHub table: `columns` = (header, row key or callable) pairs; `|` in text is escaped."""

    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|")

    head = "| " + " | ".join(cell(h) for h, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(cell(c(r) if callable(c) else r.get(c, "")) for _, c in columns) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


# --------------------------------------------------------------------------- R1 (T1)


def r1(
    t1: Sequence[Mapping[str, Any]], tiers: Mapping[str, int], paper: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """T1 analogue: RR per tier × benchmark × spectrum (seed 0) plus the tier pooled row,
    with the paper's LoRA column alongside for the Tier 1 models (our QLoRA, D-034)."""
    by = _index(t1, "setting", "bench", "method")
    out = []
    for tier in sorted(set(tiers.values())):
        names = [s for s in spectrum_settings(t1) if tiers.get(s.split("/")[0]) == tier]
        for bench in BENCHES:
            for name in [*names, f"tier{tier}/pooled"]:
                r = by.get((name, bench, "rr"))
                if r is None:
                    continue
                p = parse_setting(name)
                model, regime = (p[0], p[1]) if p else (f"tier {tier} pooled", "")
                pm = paper["model_of_slug"].get(model) if tier == 1 else None
                ref = paper["t1"][bench][pm]["lora"] if pm else {}
                out.append(
                    {
                        "tier": tier,
                        "bench": bench,
                        "model": model,
                        "regime": regime,
                        "n_obs": int(num(r["n_obs"])),
                        "n_pos": int(num(r["n_pos"])),
                        "r": num(r["pearson_r"]),
                        "r_lo": num(r["r_lo"]),
                        "r_hi": num(r["r_hi"]),
                        "p": num(r["pearson_p"]),
                        "stars": r["stars"],
                        "auc": num(r["roc_auc"]),
                        "auc_lo": num(r["auc_lo"]),
                        "auc_hi": num(r["auc_hi"]),
                        "mae_in": num(r["mae_in"]),
                        "mae_oob": num(r["mae_oob"]),
                        "paper_r": ref.get("r", math.nan),
                        "paper_stars": ref.get("stars", ""),
                        "paper_auc": ref.get("auc", math.nan),
                        "paper_mae": ref.get("mae", math.nan),
                    }
                )
    return out


R1_COLUMNS = [
    ("Benchmark", lambda r: BENCH_NAMES[r["bench"]]),
    ("Model", "model"),
    ("Regime", "regime"),
    ("n obs", "n_obs"),
    ("n pos", "n_pos"),
    ("Pearson r [95% CI]", lambda r: f"{fmt(r['r'])}{r['stars']} {ci(r['r_lo'], r['r_hi'])}"),
    ("p", lambda r: fmt(r["p"], 3)),
    (
        "ROC AUC [95% CI]",
        lambda r: (
            "n/a (degenerate)"
            if math.isnan(r["auc"])
            else f"{fmt(r['auc'])} {ci(r['auc_lo'], r['auc_hi'])}"
        ),
    ),
    ("MAE (in)", lambda r: fmt(r["mae_in"], 3)),
    ("MAE (OOB)", lambda r: fmt(r["mae_oob"], 3)),
    (
        "Paper r / AUC / MAE (LoRA)",
        lambda r: (
            ""
            if math.isnan(r["paper_r"])
            else f"{fmt(r['paper_r'])}{r['paper_stars']} / {fmt(r['paper_auc'])}"
            f" / {fmt(r['paper_mae'], 3)}"
        ),
    ),
]


# --------------------------------------------------------------------------- R2 (T7)


def r2(
    t1: Sequence[Mapping[str, Any]], c3: Sequence[Mapping[str, Any]], paper: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """T7 analogue: AUC and r of RR, SEAT, Procrustes-SEAT and CKA drift per setting ×
    benchmark, with the paired-bootstrap AUC(RR) − AUC(baseline) (C3) and, for Llama-8B,
    the paper's T7."""
    by = _index(t1, "setting", "bench", "method")
    d = _index(c3, "setting", "bench", "b")
    names = [
        *spectrum_settings(t1),
        *sorted({r["setting"] for r in t1 if "pooled" in r["setting"]}),
    ]
    out = []
    for bench in BENCHES:
        for name in names:
            if (name, bench, "rr") not in by:
                continue
            row: dict[str, Any] = {"setting": name, "bench": bench}
            for m in METHODS:
                r = by.get((name, bench, m), {})
                row[f"auc_{m}"], row[f"r_{m}"] = num(r.get("roc_auc")), num(r.get("pearson_r"))
            for m in METHODS[1:]:
                c = d.get((name, bench, m), {})
                row[f"d_{m}"], row[f"d_{m}_lo"], row[f"d_{m}_hi"] = (
                    num(c.get("diff")),
                    num(c.get("lo")),
                    num(c.get("hi")),
                )
            ref = paper["t7"].get(bench, {}) if name.startswith("llama3.1-8b-it/") else {}
            for m in METHODS:
                row[f"paper_auc_{m}"] = ref.get(m, {}).get("auc", math.nan)
            out.append(row)
    return out


def _auc_r(m: str) -> Callable[[Mapping[str, Any]], str]:
    return lambda r: f"{fmt(r[f'auc_{m}'])} ({fmt(r[f'r_{m}'])})"


def _diff(m: str) -> Callable[[Mapping[str, Any]], str]:
    return lambda r: f"{fmt(r[f'd_{m}'])} {ci(r[f'd_{m}_lo'], r[f'd_{m}_hi'])}"


R2_COLUMNS = [
    ("Benchmark", lambda r: BENCH_NAMES[r["bench"]]),
    ("Setting", "setting"),
    *[
        (
            f"{METHOD_NAMES[m]} AUC (r)",
            _auc_r(m),
        )
        for m in METHODS
    ],
    *[
        (
            f"RR − {METHOD_NAMES[m]}",
            _diff(m),
        )
        for m in METHODS[1:]
    ],
    (
        "Paper AUC RR / SEAT / Proc. / CKA (Llama)",
        lambda r: (
            ""
            if math.isnan(r["paper_auc_rr"])
            else " / ".join(fmt(r[f"paper_auc_{m}"]) for m in METHODS)
        ),
    ),
]


# --------------------------------------------------------------------------- R3 (T5), R4 (T6)


def _cell_auc(
    abl: Mapping[tuple, Mapping[str, Any]],
    t1: Mapping[tuple, Mapping[str, Any]],
    cell: str,
    setting: str,
    bench: str,
    method: str,
) -> float:
    if cell == "primary":
        return num(t1.get((setting, bench, method), {}).get("roc_auc"))
    return num(abl.get((cell, setting, bench, method), {}).get("roc_auc"))


def _variant_settings(t1: Sequence[Mapping[str, Any]]) -> list[str]:
    return [*spectrum_settings(t1), "tier1/pooled", "tier2/pooled", "all/pooled"]


def r3(
    t1: Sequence[Mapping[str, Any]], ablations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """T5a/b analogue: RR AUC over the base cell and the five attribute (target) variants:
    mean ± population std, min / max (degenerate settings are skipped)."""
    abl, t = (
        _index(ablations, "cell", "setting", "bench", "method"),
        _index(t1, "setting", "bench", "method"),
    )
    out = []
    for bench in BENCHES:
        for name in _variant_settings(t1):
            base = _cell_auc(abl, t, "primary", name, bench, "rr")
            if math.isnan(base):
                continue
            row: dict[str, Any] = {"setting": name, "bench": bench, "base": base}
            for kind, variants in (("attr", ATTR_VARIANTS), ("target", TARGET_VARIANTS)):
                vals = [base] + [
                    _cell_auc(abl, t, f"{kind}={v}", name, bench, "rr") for v in variants
                ]
                vals = [v for v in vals if not math.isnan(v)]
                row.update(
                    {
                        f"{kind}_mean": float(np.mean(vals)),
                        f"{kind}_std": _pstd(vals),
                        f"{kind}_min": min(vals),
                        f"{kind}_max": max(vals),
                        f"{kind}_n": len(vals),
                    }
                )
            out.append(row)
    return out


R3_COLUMNS = [
    ("Benchmark", lambda r: BENCH_NAMES[r["bench"]]),
    ("Setting", "setting"),
    ("Base AUC", lambda r: fmt(r["base"])),
    ("Attribute sets mean ± std", lambda r: f"{fmt(r['attr_mean'], 3)} ± {fmt(r['attr_std'], 3)}"),
    ("min / max", lambda r: f"{fmt(r['attr_min'], 3)} / {fmt(r['attr_max'], 3)}"),
    (
        "Target templates mean ± std",
        lambda r: f"{fmt(r['target_mean'], 3)} ± {fmt(r['target_std'], 3)}",
    ),
    ("min / max ", lambda r: f"{fmt(r['target_min'], 3)} / {fmt(r['target_max'], 3)}"),
]


def r4(
    t1: Sequence[Mapping[str, Any]], ablations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """T6 analogue: AUC of RR and SEAT under mean (the primary cell), max and last pooling."""
    abl, t = (
        _index(ablations, "cell", "setting", "bench", "method"),
        _index(t1, "setting", "bench", "method"),
    )
    out = []
    for bench in BENCHES:
        for name in _variant_settings(t1):
            row: dict[str, Any] = {"setting": name, "bench": bench}
            for m in ("rr", "seat"):
                for pool in ("mean", "max", "last"):
                    cell = "primary" if pool == "mean" else f"pooling={pool}"
                    row[f"{m}_{pool}"] = _cell_auc(abl, t, cell, name, bench, m)
            if not all(math.isnan(row[f"rr_{p}"]) for p in ("mean", "max", "last")):
                out.append(row)
    return out


R4_COLUMNS = [
    ("Benchmark", lambda r: BENCH_NAMES[r["bench"]]),
    ("Setting", "setting"),
    *[
        (f"{METHOD_NAMES[m]} {p}", (lambda k: lambda r: fmt(r[k]))(f"{m}_{p}"))
        for m in ("rr", "seat")
        for p in ("mean", "max", "last")
    ],
]


# --------------------------------------------------------------------------- R5 (E.5)


def r5(seeds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """App. E.5 analogue: per Tier 2 multi-seed spectrum and checkpoint, the RR ΔB mean over
    groups and seeds, the across-seed std of that mean and the mean per-group across-seed std."""
    out = []
    for r in seeds:
        if r["method"] != "rr":
            continue
        out.append(
            {
                "model": r["model"],
                "regime": r["regime"],
                "ckpt": r["ckpt"],
                "n_seeds": int(num(r["n_seeds"])),
                "mean_db": num(r["mean_db"]),
                "std_of_mean_db": num(r["std_of_mean_db"]),
                "mean_group_std": num(r["mean_group_std"]),
            }
        )
    return sorted(out, key=lambda r: (r["model"], r["regime"], r["ckpt"]))


R5_COLUMNS = [
    ("Model", "model"),
    ("Regime", "regime"),
    ("Checkpoint", "ckpt"),
    ("Seeds", "n_seeds"),
    ("Mean ΔB (RR)", lambda r: fmt(r["mean_db"], 4)),
    ("Std of mean ΔB across seeds", lambda r: fmt(r["std_of_mean_db"], 4)),
    ("Mean per-group std across seeds", lambda r: fmt(r["mean_group_std"], 4)),
]


# --------------------------------------------------------------------------- R6 (T3)


def _per_ckpt_minutes(part: Mapping[str, Any] | None) -> float:
    if not part or not part.get("n"):
        return math.nan
    return part["seconds_total"] / part["n"] / 60.0


def r6(
    timing: Mapping[str, Mapping[str, Any]], tiers: Mapping[str, int], paper: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """T3 analogue on 2× T4: minutes per checkpoint for our method (embedding + ΔB) and for
    each benchmark (generation; WGM/ToxiGen classification is in the score stage), the
    benchmark / ours ratio, and the paper's per-model minutes on one A100."""
    out = []
    for run, t in sorted(timing.items()):
        model = t["model"]
        row: dict[str, Any] = {"run": run, "model": model, "tier": tiers.get(model, "")}
        row["embed"] = _per_ckpt_minutes(t.get("embed"))
        row["deltab"] = _per_ckpt_minutes(t.get("deltab"))
        for b in BENCHES:
            row[f"gen_{b}"] = _per_ckpt_minutes(t.get("generate", {}).get(b))
        row["score"] = _per_ckpt_minutes(t.get("score"))
        ours = row["embed"] + (0.0 if math.isnan(row["deltab"]) else row["deltab"])
        row["ours"] = ours
        for b in BENCHES:
            row[f"ratio_{b}"] = (
                row[f"gen_{b}"] / ours if ours and not math.isnan(ours) else math.nan
            )
        pm = paper["model_of_slug"].get(model)
        p = paper["t3_minutes"].get(pm, {}) if pm else {}
        row["paper_ours"] = p.get("embed", math.nan) + p.get("deltab", math.nan) if p else math.nan
        row["paper_dt"] = p.get("dt", math.nan) if p else math.nan
        out.append(row)
    return out


R6_COLUMNS = [
    ("Model", "model"),
    ("Tier", "tier"),
    ("Ours: embed + ΔB (min/ckpt)", lambda r: f"{fmt(r['embed'], 1)} + {fmt(r['deltab'], 1)}"),
    ("WGM gen", lambda r: fmt(r["gen_wgm"], 1)),
    ("ToxiGen gen", lambda r: fmt(r["gen_toxigen"], 1)),
    ("DT", lambda r: fmt(r["gen_dt"], 1)),
    ("Scoring (all benches)", lambda r: fmt(r["score"], 1)),
    (
        "Ratio WGM / ToxiGen / DT vs ours",
        lambda r: " / ".join(f"{fmt(r[f'ratio_{b}'], 1)}×" for b in R6_RATIO_ORDER),
    ),
    (
        "Paper ours / DT (A100 min/model)",
        lambda r: (
            ""
            if math.isnan(r["paper_ours"])
            else f"{fmt(r['paper_ours'], 1)} / {fmt(r['paper_dt'], 1)}"
        ),
    ),
]


# --------------------------------------------------------------------------- R7 (verdict)


def verdict(
    t1: Sequence[Mapping[str, Any]],
    c3: Sequence[Mapping[str, Any]],
    ablations: Sequence[Mapping[str, Any]],
    seeds: Sequence[Mapping[str, Any]],
    tiers: Mapping[str, int],
    paper: Mapping[str, Any],
) -> dict[str, Any]:
    """PLAN §1.2: C1-C3 per setting (seed-0 spectrum × benchmark), the tier verdicts and the
    secondary claims S1-S7.

    C1: r < 0 with p < 0.05. C2: AUC CI lower bound > 0.5 (degenerate settings leave the
    denominator). C3 (WGM, DT): AUC_RR > AUC_SEAT with the paired CI above 0. Tier:
    *refuted* if C1 holds in ≤ 1/3 of settings or any setting with n ≥ 50 has a
    significant positive r; else *confirmed* if C1 and C2 hold in ≥ 2/3 and C3 in ≥ 2/3
    of pairs; else *inconclusive* (refutation is checked first, D-102)."""
    t = _index(t1, "setting", "bench", "method")
    d = _index(c3, "setting", "bench", "b")
    abl = _index(ablations, "cell", "setting", "bench", "method")
    rows = []
    for name in spectrum_settings(t1):
        model, regime, _ = parse_setting(name)  # type: ignore[misc]
        for bench in BENCHES:
            r = t.get((name, bench, "rr"))
            if r is None:
                continue
            rr, p, lo, auc = (
                num(r["pearson_r"]),
                num(r["pearson_p"]),
                num(r["auc_lo"]),
                num(r["roc_auc"]),
            )
            c = d.get((name, bench, "seat")) if bench in C3_BENCHES else None
            rows.append(
                {
                    "tier": tiers.get(model),
                    "setting": name,
                    "model": model,
                    "regime": regime,
                    "bench": bench,
                    "n_obs": int(num(r["n_obs"])),
                    "r": rr,
                    "p": p,
                    "auc": auc,
                    "auc_lo": lo,
                    "c1": bool(rr < 0 and p < 0.05),
                    "c2": None if math.isnan(auc) else bool(lo > 0.5),
                    "c3": None if c is None else bool(num(c["diff"]) > 0 and num(c["lo"]) > 0),
                    "sig_positive": bool(rr > 0 and p < 0.05 and num(r["n_obs"]) >= 50),
                }
            )
    rows.sort(key=lambda r: (r["tier"], r["setting"]))  # stable: benchmark order is kept
    tier_rows = []
    for tier in sorted({r["tier"] for r in rows}):
        sub = [r for r in rows if r["tier"] == tier]
        n = len(sub)
        c1 = sum(r["c1"] for r in sub)
        c2s = [r["c2"] for r in sub if r["c2"] is not None]
        c3s = [r["c3"] for r in sub if r["c3"] is not None]
        pos = [f"{r['setting']} {r['bench']}" for r in sub if r["sig_positive"]]
        if c1 <= n / 3 or pos:
            v = "refuted"
        elif c1 >= 2 * n / 3 and sum(c2s) >= 2 * len(c2s) / 3 and sum(c3s) >= 2 * len(c3s) / 3:
            v = "confirmed"
        else:
            v = "inconclusive"
        tier_rows.append(
            {
                "tier": tier,
                "settings": n,
                "c1": _share(c1, n),
                "c2": _share(sum(c2s), len(c2s)),
                "c3": _share(sum(c3s), len(c3s)),
                "significant_positive": "; ".join(pos),
                "verdict": v,
            }
        )
    return {
        "settings": rows,
        "tiers": tier_rows,
        "secondary": secondary(t, abl, seeds, rows, paper),
    }


def secondary(
    t: Mapping[tuple, Mapping[str, Any]],
    abl: Mapping[tuple, Mapping[str, Any]],
    seeds: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    paper: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """S1-S7 over the seed-0 spectrum settings (PLAN §1.2); outcome by thirds (D-102)."""

    def auc(name, bench, method, cell="primary"):
        return _cell_auc(abl, t, cell, name, bench, method)

    pairs = [(r["setting"], r["bench"]) for r in rows if not math.isnan(r["auc"])]
    out = []
    # S1: |r| larger under full FT than LoRA on the same model (Tier 2 models with both).
    by_model: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        by_model.setdefault((r["model"], r["bench"]), {})[r["regime"]] = r["r"]
    s1 = [(k, v) for k, v in by_model.items() if "full" in v and "lora" in v]
    k = sum(abs(v["full"]) > abs(v["lora"]) for _, v in s1)
    out.append(
        {
            "id": "S1",
            "claim": "|r| larger under full FT than LoRA (same model)",
            "holds": _share(k, len(s1)),
            "outcome": _outcome(k, len(s1)),
            "detail": "; ".join(
                f"{m} {b}: {fmt(v['full'])} vs {fmt(v['lora'])}" for (m, b), v in s1
            ),
        }
    )
    # S2: Procrustes-SEAT near chance (AUC CI contains 0.5).
    vals = [(t.get((s, b, "procrustes"), {}), s, b) for s, b in pairs]
    k = sum(num(r.get("auc_lo")) <= 0.5 <= num(r.get("auc_hi")) for r, _, _ in vals)
    out.append(
        {
            "id": "S2",
            "claim": "Procrustes-SEAT near chance (AUC CI contains 0.5)",
            "holds": _share(k, len(vals)),
            "outcome": _outcome(k, len(vals)),
            "detail": "",
        }
    )
    # S3: CKA drift detects (CI above 0.5) but stays below RR.
    det = sum(num(t.get((s, b, "cka"), {}).get("auc_lo")) > 0.5 for s, b in pairs)
    below = sum(auc(s, b, "cka") < auc(s, b, "rr") for s, b in pairs)
    both = sum(
        num(t.get((s, b, "cka"), {}).get("auc_lo")) > 0.5 and auc(s, b, "cka") < auc(s, b, "rr")
        for s, b in pairs
    )
    out.append(
        {
            "id": "S3",
            "claim": "CKA drift detects but stays below RR",
            "holds": _share(both, len(pairs)),
            "outcome": _outcome(both, len(pairs)),
            "detail": f"CKA CI above 0.5 in {det}/{len(pairs)};"
            f" CKA below RR in {below}/{len(pairs)}",
        }
    )
    # S4: mean pooling best for RR.
    s4 = [(s, b) for s, b in pairs if not math.isnan(auc(s, b, "rr", "pooling=max"))]
    k = sum(
        auc(s, b, "rr") >= max(auc(s, b, "rr", "pooling=max"), auc(s, b, "rr", "pooling=last"))
        for s, b in s4
    )
    out.append(
        {
            "id": "S4",
            "claim": "Mean pooling is best (RR)",
            "holds": _share(k, len(s4)),
            "outcome": _outcome(k, len(s4)),
            "detail": "",
        }
    )
    # S5: AUC std across attribute and template variants ≤ 0.05 (both).
    k5, n5 = 0, 0
    for s, b in pairs:
        stds = []
        for kind, variants in (("attr", ATTR_VARIANTS), ("target", TARGET_VARIANTS)):
            v = [auc(s, b, "rr")] + [auc(s, b, "rr", f"{kind}={x}") for x in variants]
            stds.append(_pstd([x for x in v if not math.isnan(x)]))
        n5 += 1
        k5 += all(x <= 0.05 for x in stds)
    out.append(
        {
            "id": "S5",
            "claim": "AUC std across attribute and template variants ≤ 0.05",
            "holds": _share(k5, n5),
            "outcome": _outcome(k5, n5),
            "detail": f"paper (Llama): {paper['t5']['attr']['std']}"
            f" / {paper['t5']['target']['std']}",
        }
    )
    # S6: ΔB std across seeds ≈ the paper's 0.003 (per-group std, endpoints).
    lim = S6_TOLERANCE * paper["e5"]["std"]
    ends = [r for r in seeds if r["method"] == "rr" and r["ckpt"] in ("a000", "a100")]
    k = sum(num(r["mean_group_std"]) <= lim for r in ends)
    out.append(
        {
            "id": "S6",
            "claim": f"ΔB std across seeds ≈ {paper['e5']['std']} (≤ {lim:.3f})",
            "holds": _share(k, len(ends)),
            "outcome": _outcome(k, len(ends)),
            "detail": "; ".join(
                f"{r['model']} {r['regime']} {r['ckpt']}: {fmt(r['mean_group_std'], 4)}"
                for r in ends
            ),
        }
    )
    # S7: neutral anchors at least as good as word, Alpaca or Tulu (full pools).
    s7 = [
        (s, b)
        for s, b in pairs
        if all(not math.isnan(auc(s, b, "rr", f"anchors={src}/all")) for src in ANCHOR_SOURCES[1:])
    ]
    k = sum(
        auc(s, b, "rr") >= max(auc(s, b, "rr", f"anchors={src}/all") for src in ANCHOR_SOURCES[1:])
        for s, b in s7
    )
    out.append(
        {
            "id": "S7",
            "claim": "Neutral anchors ≥ word / Alpaca / Tulu anchors (1k each)",
            "holds": _share(k, len(s7)),
            "outcome": _outcome(k, len(s7)),
            "detail": "",
        }
    )
    return out


VERDICT_SETTING_COLUMNS = [
    ("Tier", "tier"),
    ("Setting", "setting"),
    ("Benchmark", lambda r: BENCH_NAMES[r["bench"]]),
    ("r (p)", lambda r: f"{fmt(r['r'])} ({fmt(r['p'], 3)})"),
    (
        "AUC (CI low)",
        lambda r: "n/a" if math.isnan(r["auc"]) else f"{fmt(r['auc'])} ({fmt(r['auc_lo'])})",
    ),
    ("C1", lambda r: "✓" if r["c1"] else "✗"),
    ("C2", lambda r: "n/a" if r["c2"] is None else "✓" if r["c2"] else "✗"),
    ("C3", lambda r: "–" if r["c3"] is None else "✓" if r["c3"] else "✗"),
]
VERDICT_TIER_COLUMNS = [
    ("Tier", "tier"),
    ("Settings", "settings"),
    ("C1", "c1"),
    ("C2", "c2"),
    ("C3", "c3"),
    ("Significant positive r (n ≥ 50)", "significant_positive"),
    ("Verdict", "verdict"),
]
SECONDARY_COLUMNS = [
    ("Claim", lambda r: f"{r['id']}: {r['claim']}"),
    ("Holds", "holds"),
    ("Outcome", "outcome"),
    ("Detail", "detail"),
]
