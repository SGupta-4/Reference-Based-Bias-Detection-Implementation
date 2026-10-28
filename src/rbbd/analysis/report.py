"""M7 report: `results/REPORT.md` with tables R1-R7, figures F-R1-F-R5 and the verdict
(PLAN §1.2, §10; D-102). CPU only.

Inputs (all aggregates already in `results/`, DC-18):
- `results/m6/*.csv` (the M6 tables, D-097, D-101);
- `results/<run>/{delta_b_cells.csv.gz, bench_scores.csv}` for the scatter, sweep and ROC
  figures (primary cell), and the optional `results/<run>/timing.json` (R6,
  `analysis.timing`);
- `resources/paper_values.json` (the paper's numbers) and `DECISION.md` (the realised
  deviations: every D entry whose "Paper deviation" starts with "yes").

Like `analysis.analyze`, the outputs are keyed on the content hashes of those inputs
(`report/manifest.json`); an unchanged second call logs `cache hit: report`. Writes
`<out>/REPORT.md`, `<out>/report/{r1..r7,...}.csv`, `<out>/figures/f_r*.png` and
`<out>/report/summary.json`. Called by `cli report` and the runner's `report` stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from rbbd.analysis import analyze as an
from rbbd.analysis import plots
from rbbd.analysis import tables as T
from rbbd.utils.logging import get_logger

log = get_logger(__name__)

REPORT_SCHEMA = 2  # 2: R6 ratio order fixed (B-041)
M6_TABLES = ("t1", "c3", "f3_sweep", "ablations", "wgm_pairing", "sign", "seeds", "nf4")
TIMING_FILE = "timing.json"


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def deviations(decision_md: Path | None) -> list[dict[str, str]]:
    """Realised paper deviations: D entries (not superseded) whose "Paper deviation" says yes."""
    if decision_md is None or not Path(decision_md).exists():
        return []
    text = Path(decision_md).read_text()
    out = []
    for m in re.finditer(r"^## (D-\d+) (.*?)$(.*?)(?=^## D-|\Z)", text, re.S | re.M):
        did, title, body = m.groups()
        pd = re.search(r"^- Paper deviation:\s*(.*)$", body, re.M)
        if pd and pd.group(1).lower().startswith("yes") and "superseded" not in title:
            what = re.sub(r"^yes[\s,.;:—–-]*", "", pd.group(1).strip()) or "yes"
            out.append({"id": did, "title": title.strip(), "deviation": what})
    return out


def _inputs(results_root: Path, decision_md: Path | None, m6_dir: Path) -> dict[str, Path]:
    root = Path(results_root)
    files = {f"m6/{t}.csv": Path(m6_dir) / f"{t}.csv" for t in M6_TABLES}
    for run, fs in an.discover(root).items():
        for p in fs.values():
            files[f"{run}/{p.name}"] = p
        if (root / run / TIMING_FILE).exists():
            files[f"{run}/{TIMING_FILE}"] = root / run / TIMING_FILE
    if decision_md is not None and Path(decision_md).exists():
        files["DECISION.md"] = Path(decision_md)
    return {k: p for k, p in sorted(files.items()) if p.exists()}


def build_report(
    results_root: Path,
    out_dir: Path | None = None,
    tiers: Mapping[str, int] | None = None,
    decision_md: Path | None = None,
    force: bool = False,
    m6_dir: Path | None = None,
) -> dict[str, Any]:
    """Write the M7 report for every run under `results_root` (see the module docstring).
    `m6_dir` holds the M6 tables (default `<results_root>/m6`; the runner stage passes the
    `analyze` stage dir). Returns the summary (tier verdicts, secondary outcomes, files
    written, `cache_hit`)."""
    root = Path(results_root)
    out = Path(out_dir) if out_dir is not None else root
    m6_dir = Path(m6_dir) if m6_dir is not None else root / "m6"
    files = _inputs(root, decision_md, m6_dir)
    paper_text = resources.files("rbbd.resources").joinpath("paper_values.json").read_text()
    manifest = {
        "schema": REPORT_SCHEMA,
        "inputs": {k: an.sha256_file(p) for k, p in files.items()},
        "paper_values": hashlib.sha256(paper_text.encode()).hexdigest(),
        "tiers": dict(sorted((tiers or {}).items())),
    }
    mpath = out / "report" / "manifest.json"
    if not force and mpath.exists() and json.loads(mpath.read_text()) == manifest:
        log.info("cache hit: report %s", out)
        return {"cache_hit": True, **json.loads((out / "report" / "summary.json").read_text())}
    if "m6/t1.csv" not in files:
        raise FileNotFoundError(f"{m6_dir}/t1.csv missing: run `cli analyze` first (M6)")

    tiers = dict(tiers or {})
    paper = T.paper_values()
    m6 = {t: _rows(m6_dir / f"{t}.csv") for t in M6_TABLES}
    t1, c3, abl, seeds = m6["t1"], m6["c3"], m6["ablations"], m6["seeds"]

    # Observations of the primary cell, per (benchmark, method), for the figures.
    db: list[dict[str, Any]] = []
    bench: list[dict[str, Any]] = []
    timing: dict[str, Any] = {}
    for run, fs in an.discover(root).items():
        if run == an.NF4_RUN or "bench" not in fs:
            continue
        db += [r for r in an.read_delta_b(fs["delta_b"]) if r["cell"] == an.PRIMARY]
        bench += an.read_bench(fs["bench"])
        if (root / run / TIMING_FILE).exists():
            timing[run] = json.loads((root / run / TIMING_FILE).read_text())
    obs = {(b, m): an.observations(db, bench, b, m) for b in T.BENCHES for m in T.METHODS}

    r1 = T.r1(t1, tiers, paper)
    r2 = T.r2(t1, c3, paper)
    r3 = T.r3(t1, abl)
    r4 = T.r4(t1, abl)
    r5 = T.r5(seeds)
    r6 = T.r6(timing, tiers, paper)
    v = T.verdict(t1, c3, abl, seeds, tiers, paper)
    devs = deviations(decision_md)
    for name, rows in (
        ("r1", r1),
        ("r2", r2),
        ("r3", r3),
        ("r4", r4),
        ("r5", r5),
        ("r6", r6),
        ("r7_settings", v["settings"]),
        ("r7_tiers", v["tiers"]),
        ("r7_secondary", v["secondary"]),
        ("deviations", devs),
    ):
        _write_csv(out / "report" / f"{name}.csv", rows)

    fig_dir = out / "figures"
    figs = [
        *plots.scatter({b: obs[(b, "rr")] for b in T.BENCHES}, fig_dir),
        *plots.threshold_curves(obs, tiers, an.SWEEP, fig_dir),
        *plots.anchor_curves(t1, abl, [*T.spectrum_settings(t1), "all/pooled"], fig_dir),
        *plots.seed_groups(an.read_delta_b(_seed_run(root)) if _seed_run(root) else [], fig_dir),
        *plots.roc_curves({b: obs[(b, "rr")] for b in T.BENCHES}, fig_dir),
    ]
    text = render(
        r1,
        r2,
        r3,
        r4,
        r5,
        r6,
        v,
        m6,
        devs,
        paper,
        [p.relative_to(out) for p in figs],
        timing_missing=not timing,
    )
    (out / "REPORT.md").write_text(text)
    summary = {
        "tiers": v["tiers"],
        "secondary": [{k: s[k] for k in ("id", "holds", "outcome")} for s in v["secondary"]],
        "figures": [str(p.relative_to(out)) for p in figs],
        "timing_runs": sorted(timing),
    }
    (out / "report" / "summary.json").write_text(json.dumps(summary, indent=1))
    mpath.write_text(json.dumps(manifest, indent=1))
    return {"cache_hit": False, **summary}


def _seed_run(root: Path) -> Path | None:
    """The ΔB table of the first run with several seeds (Tier 2 Qwen full FT, D-033)."""
    for run, fs in an.discover(root).items():
        if run == an.NF4_RUN:
            continue
        with an._open(fs["delta_b"]) as fh:
            seeds = {r["seed"] for r in csv.DictReader(fh)}
        if len(seeds) > 1:
            return fs["delta_b"]
    return None


# --------------------------------------------------------------------------- Markdown


def _sign_rows(sign: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    keep = []
    for r in sign:
        if math.isnan(T.num(r.get("score_dt"))):
            continue
        keep.append(r)
    return keep


def _db_hu(m: str):
    # rr_target / rr_attr exist only when `decompose-deltab` ran for the run (D-085)
    return lambda r: (
        f"{T.fmt(r.get(f'db_{m}'), 3)} {T.ci(r.get(f'db_{m}_lo'), r.get(f'db_{m}_hi'), 3)}"
    )


def _score_hu(b: str):
    return lambda r: T.fmt(r.get(f"score_{b}"), 3)


SIGN_COLUMNS = [
    ("Spectrum", "spectrum"),
    *[(f"ΔB h−u {T.METHOD_NAMES[m]}", _db_hu(m)) for m in ("rr", "rr_target", "rr_attr")],
    *[(f"Δscore h−u {T.BENCH_NAMES[b]}", _score_hu(b)) for b in T.BENCHES],
    (
        "RR agrees with DT / ToxiGen / WGM",
        lambda r: " / ".join(
            "✓" if r[f"agree_rr_{b}"] == "True" else "✗" for b in ("dt", "toxigen", "wgm")
        ),
    ),
]

NF4_COLUMNS = [
    ("Method", lambda r: T.METHOD_NAMES.get(r["method"], r["method"])),
    ("Checkpoint", "ckpt"),
    ("Mean ΔB fp16", lambda r: T.fmt(r["mean_db_fp16"], 4)),
    ("Mean ΔB NF4", lambda r: T.fmt(r["mean_db_nf4"], 4)),
    ("Mean |diff|", lambda r: T.fmt(r["mean_abs_diff"], 4)),
    ("r over groups", lambda r: T.fmt(r["pearson_r"], 3)),
    ("Same h−u sign", lambda r: {"True": "✓", "False": "✗"}.get(r["same_sign"], "")),
]

PAIRING_COLUMNS = [
    ("Map", "map"),
    ("Pairing", "pairing"),
    ("Setting", "setting"),
    ("r", lambda r: T.fmt(r["pearson_r"])),
    ("AUC [95% CI]", lambda r: f"{T.fmt(r['roc_auc'])} {T.ci(r['auc_lo'], r['auc_hi'])}"),
]

DEVIATION_COLUMNS = [("Decision", "id"), ("Title", "title"), ("Deviation", "deviation")]


def render(r1, r2, r3, r4, r5, r6, v, m6, devs, paper, figs, *, timing_missing: bool) -> str:
    """The Markdown text of REPORT.md; numbers only from the computed rows."""
    tiers = {t["tier"]: t for t in v["tiers"]}
    passing = [r for r in v["settings"] if r["c1"] and r["c2"]]
    lines = [
        "# Reference-based bias detection: replication report",
        "",
        "Replication of *Reference-Based Bias Detection in LLMs via Relative Representations "
        "of Hidden States* (arXiv:2609.10060v1) on Kaggle 2× T4. Generated by "
        "`python -m rbbd.cli report` from the aggregates in `results/` (D-102); every number "
        "below comes from the CSVs in `results/report/` and `results/m6/`.",
        "",
        "## Verdict (pre-registered, PLAN §1.2)",
        "",
        *[
            f"- **Tier {t}: {row['verdict']}.** C1 {row['c1']} settings, C2 {row['c2']}, "
            f"C3 {row['c3']} (model, benchmark) pairs."
            + (
                f" Significant positive r (n ≥ 50): {row['significant_positive']}."
                if row["significant_positive"]
                else ""
            )
            for t, row in sorted(tiers.items())
        ],
        "- Settings meeting both C1 and C2: "
        f"{', '.join(sorted({r['setting'] for r in passing})) or 'none'}"
        f" ({len(passing)}/{len(v['settings'])} setting × benchmark cells)."
        " The paper reports C1 in "
        f"{paper['c1_settings']['passed']}/{paper['c1_settings']['total']} of its settings.",
        "- C1: r(Δscore, ΔB) < 0 with p < 0.05. C2: AUC 95% CI lower bound > 0.5. "
        "C3: AUC_RR − AUC_SEAT paired CI > 0 (WGM, DT). Refuted if C1 holds in ≤ 1/3 of a "
        "tier's settings or any setting has a significant positive r; WGM p-values treat "
        "groups as independent although groups of a topic share a score (its CIs resample "
        "(checkpoint, topic) clusters).",
        "",
        T.markdown(T.VERDICT_TIER_COLUMNS, v["tiers"]),
        "",
        "### Per setting",
        "",
        T.markdown(T.VERDICT_SETTING_COLUMNS, v["settings"]),
        "",
        "### Secondary claims (S1-S7)",
        "",
        "Outcome by thirds of the testable settings: ≥ 2/3 holds, ≤ 1/3 fails, else mixed (D-102).",
        "",
        T.markdown(T.SECONDARY_COLUMNS, v["secondary"]),
        "",
        "## Where the direction comes from (h − u)",
        "",
        "ΔB of the harmful (a000) minus the unharmful (a100) endpoint, per spectrum, with the "
        "target-only and attribute-only parts of RR (D-085, D-101; 95% CI over groups), next "
        "to the benchmarks' h − u. Negative ΔB = more biased; positive Δscore = more biased.",
        "",
        T.markdown(SIGN_COLUMNS, _sign_rows(m6["sign"])),
        "",
        "## R1 · T1 analogue (RR)",
        "",
        "Tier 1 is QLoRA and is set against the paper's LoRA column; Tier 2 has no paper "
        "counterpart. n/a (degenerate) = fewer than 5 positives or negatives at the threshold.",
        "",
        *[
            part
            for t in sorted(tiers)
            for part in (
                f"### Tier {t}",
                "",
                T.markdown(T.R1_COLUMNS, [r for r in r1 if r["tier"] == t]),
                "",
            )
        ],
        "## R2 · T7 analogue (methods)",
        "",
        "CKA drift is undirected (|r|, max(AUC, 1 − AUC)). RR − baseline = paired bootstrap of "
        "the AUC difference (C3).",
        "",
        T.markdown(T.R2_COLUMNS, r2),
        "",
        "## R3 · T5 analogue (attribute sets and target templates, RR)",
        "",
        f"Paper (Llama): attribute sets {paper['t5']['attr']['mean']}"
        f" ± {paper['t5']['attr']['std']}, "
        f"target templates {paper['t5']['target']['mean']} ± {paper['t5']['target']['std']}.",
        "",
        T.markdown(T.R3_COLUMNS, r3),
        "",
        "## R4 · T6 analogue (pooling)",
        "",
        "Paper (Llama): RR "
        + " / ".join(f"{k} {x}" for k, x in paper["t6"]["rr"].items())
        + "; SEAT "
        + " / ".join(f"{k} {x}" for k, x in paper["t6"]["seat"].items())
        + ".",
        "",
        T.markdown(T.R4_COLUMNS, r4),
        "",
        "## R5 · App. E.5 analogue (training seeds, Tier 2)",
        "",
        f"Paper (Llama, 3 seeds): std {paper['e5']['std']};"
        f" mean ΔB {paper['e5']['mean_unharmful']} "
        f"(unharmful), {paper['e5']['mean_synth']} (synth).",
        "",
        T.markdown(T.R5_COLUMNS, r5),
        "",
        "## R6 · T3 analogue (cost on 2× T4)",
        "",
        "Minutes per checkpoint. Ours = embedding extraction (model load excluded) + ΔB. "
        "Generation per benchmark is summed over every process that worked on a checkpoint; "
        "scoring is the last score run (WildGuard over every checkpoint, earlier labels "
        "reused), so it is a lower bound. Paper: minutes per model on one A100.",
        "",
        "Timing not collected yet (`cli collect-timing`, D-102)."
        if timing_missing
        else T.markdown(T.R6_COLUMNS, r6),
        "",
        "## Sensitivity",
        "",
        "### NF4 extraction base (Llama-3.1-8B, D-100)",
        "",
        T.markdown(NF4_COLUMNS, [r for r in m6["nf4"] if r["method"] in ("rr", "seat")]),
        "",
        "### WGM topic map × pairing (RR, pooled; D-095)",
        "",
        T.markdown(
            PAIRING_COLUMNS,
            [
                r
                for r in m6["wgm_pairing"]
                if r["method"] == "rr" and r["setting"].endswith("pooled")
            ],
        ),
        "",
        "## Figures",
        "",
        *[f"![{p.stem}]({p.as_posix()})" for p in figs],
        "",
        "## Deviations from the paper (realised)",
        "",
        "Every decision whose paper-deviation field is *yes* (DECISION.md); the planned list "
        "is PLAN §1.3.",
        "",
        T.markdown(DEVIATION_COLUMNS, devs) if devs else "DECISION.md not available.",
        "",
    ]
    return "\n".join(lines)


def stage(ctx: Any) -> Any:
    """Runner `report` stage: the report for this config's `results/<run>/` into the stage dir."""
    from rbbd.metrics.delta_b import results_dir
    from rbbd.runner import StageResult, stage_run_keys

    res = results_dir(ctx.cfg, Path(ctx.env.artifacts_root))
    tiers = {m["slug"]: int(ctx.cfg.get("tier", 0)) for m in ctx.cfg["models"]}
    m6_dir = Path(ctx.env.artifacts_root) / "analyze" / stage_run_keys(ctx.cfg)["analyze"]
    build_report(res.parent, ctx.stage_dir, tiers=tiers, m6_dir=m6_dir)
    root = Path(ctx.env.artifacts_root)
    return StageResult(outputs={"report": str((ctx.stage_dir / "REPORT.md").relative_to(root))})
