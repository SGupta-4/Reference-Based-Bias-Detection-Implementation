"""M7 report (D-102): verdict rules (PLAN §1.2), the report on synthetic results in the real
formats (tables, figures, cache), R6 timing collection and the transcribed paper values."""

import csv
import json
import math

import pytest

from rbbd.analysis import analyze as an
from rbbd.analysis import report as rp
from rbbd.analysis import tables as T
from rbbd.analysis import timing as tm
from tests.test_analyze import _make_run


def _t1(setting, bench, r, p, auc, lo, method="rr", n=168):
    return {
        "setting": setting,
        "bench": bench,
        "method": method,
        "n_obs": n,
        "n_pos": 60,
        "pearson_r": r,
        "pearson_p": p,
        "stars": "",
        "r_lo": "",
        "r_hi": "",
        "roc_auc": auc,
        "auc_lo": lo,
        "auc_hi": "",
        "mae_in": "",
        "mae_oob": "",
    }


def _c3(setting, bench, diff, lo):
    return {"setting": setting, "bench": bench, "a": "rr", "b": "seat", "diff": diff, "lo": lo}


def test_verdict_applies_the_preregistered_rules():
    good = [_t1("a/lora-s0", b, -0.7, 0.001, 0.9, 0.8) for b in T.BENCHES]
    bad = [_t1("c/qlora-s0", b, 0.3, 0.001, 0.3, 0.2) for b in T.BENCHES]
    weak = [_t1("d/qlora-s0", b, -0.1, 0.2, 0.55, 0.45) for b in T.BENCHES]
    c3 = [_c3("a/lora-s0", b, 0.1, 0.02) for b in T.C3_BENCHES]
    tiers = {"a": 2, "c": 1, "d": 3}
    v = T.verdict(good + bad + weak, c3, [], [], tiers, T.paper_values())
    by = {t["tier"]: t for t in v["tiers"]}
    assert by[2]["verdict"] == "confirmed" and by[2]["c1"] == "3/3" and by[2]["c3"] == "2/2"
    assert by[1]["verdict"] == "refuted" and "c/qlora-s0 wgm" in by[1]["significant_positive"]
    assert by[3]["verdict"] == "refuted"  # C1 in 0/3 ≤ 1/3
    deg = [_t1("e/lora-s0", b, -0.7, 0.001, "", "") for b in T.BENCHES]
    row = next(r for r in T.verdict(deg, [], [], [], {"e": 2}, T.paper_values())["settings"])
    assert row["c2"] is None and row["c1"]  # degenerate AUC leaves the C2 denominator
    two_of_three = [
        *good,
        *[_t1("b/lora-s0", b, -0.5, 0.01, 0.7, 0.6) for b in T.BENCHES],
        *[_t1("f/lora-s0", b, 0.1, 0.4, 0.5, 0.4) for b in T.BENCHES],
    ]
    c3_more = c3 + [
        _c3(s, b, -0.05, -0.1) for s in ("b/lora-s0", "f/lora-s0") for b in T.C3_BENCHES
    ]
    v = T.verdict(two_of_three, c3_more, [], [], {"a": 2, "b": 2, "f": 2}, T.paper_values())
    assert v["tiers"][0]["c1"] == "6/9" and v["tiers"][0]["c3"] == "2/6"
    assert v["tiers"][0]["verdict"] == "inconclusive"  # C1 and C2 at 2/3, C3 below


def test_markdown_escapes_pipes():
    assert T.markdown([("|r|", "x")], [{"x": "a|b"}]).splitlines()[2] == "| a\\|b |"


@pytest.fixture
def results(tmp_path):
    root = tmp_path / "results"
    _make_run(root, "tier1_x", "x-it", regime="qlora", cells=("primary", "pooling=max"))
    _make_run(root, "tier2_y", "y-it", seeds=(0, 1, 2))
    an.run_analysis(root, root / "m6", tiers={"x-it": 1, "y-it": 2}, n_boot=50)
    return root


def test_report_writes_tables_figures_and_caches(results, tmp_path, caplog):
    decisions = tmp_path / "DECISION.md"
    decisions.write_text(
        "## D-001 A choice\n- Paper deviation: yes — App. C.\n"
        "## D-002 Another — superseded by D-003\n- Paper deviation: yes.\n"
        "## D-003 No deviation\n- Paper deviation: no.\n"
    )
    tiers = {"x-it": 1, "y-it": 2}
    s = rp.build_report(results, tiers=tiers, decision_md=decisions)
    assert not s["cache_hit"] and {t["tier"] for t in s["tiers"]} == {1, 2}
    text = (results / "REPORT.md").read_text()
    for heading in (
        "## Verdict",
        "## R1",
        "## R2",
        "## R3",
        "## R4",
        "## R5",
        "## R6",
        "## Where the direction comes from",
        "## Deviations",
    ):
        assert heading in text
    assert "| D-001 | A choice | App. C. |" in text and "D-002" not in text
    assert "Timing not collected yet" in text
    figs = [results / f for f in s["figures"]]
    assert figs and all(f.exists() and f.stat().st_size > 1000 for f in figs)
    assert any("f_r4_seeds_y-it" in f.name for f in figs)  # the multi-seed run
    r1 = list(csv.DictReader(open(results / "report" / "r1.csv")))
    assert {r["model"] for r in r1} >= {"x-it", "y-it"} and all(float(r["r"]) < 0 for r in r1)
    again = rp.build_report(results, tiers=tiers, decision_md=decisions)
    assert again["cache_hit"] and "cache hit: report" in caplog.text
    (results / "tier1_x" / "timing.json").write_text(
        json.dumps(
            {
                "model": "x-it",
                "embed": {"seconds_total": 600.0, "n": 6},
                "deltab": {"seconds_total": 60.0, "n": 5},
                "generate": {"dt": {"seconds_total": 6000.0, "n": 6}},
            }
        )
    )
    third = rp.build_report(results, tiers=tiers, decision_md=decisions)
    assert not third["cache_hit"] and third["timing_runs"] == ["tier1_x"]
    r6 = list(csv.DictReader(open(results / "report" / "r6.csv")))
    assert float(r6[0]["embed"]) == pytest.approx(600 / 6 / 60)
    assert float(r6[0]["ratio_dt"]) == pytest.approx((6000 / 6) / (600 / 6 + 60 / 5))
    # the rendered ratios follow the header's WGM / ToxiGen / DT order (B-041)
    line = next(
        ln for ln in (results / "REPORT.md").read_text().splitlines() if ln.startswith("| x-it |")
    )
    assert line.split(" | ")[-2] == "n/a× / n/a× / 8.9×"


def test_collect_timing_reads_sidecars_and_stage_logs(tmp_path):
    root = tmp_path
    rows = []
    for i, ckpt in enumerate(["ref", "a100", "a000"]):
        rel = f"embeddings/m/{'base' if ckpt == 'ref' else 'lora-s0'}/{ckpt}/k{i}.safetensors"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).with_suffix(".json").write_text(json.dumps({"stats": {"seconds": 10.0}}))
        rows.append({"model": "m", "ckpt": ckpt, "path": rel})
    for stage, name, content in (
        ("embed", "index.json", rows + [{"model": "other", "ckpt": "ref", "path": "x"}]),
        ("deltab", "timing.json", {"spectra": {"m/lora-s0": 20.0, "other/lora-s0": 99.0}}),
        (
            "generate",
            "timing.json",
            [
                {"model": "m", "spectrum": "base", "ckpt": "ref", "bench": "dt", "seconds": 100.0},
                {
                    "model": "m",
                    "spectrum": "lora-s0",
                    "ckpt": "a000",
                    "bench": "dt",
                    "seconds": 300.0,
                },
                {
                    "model": "m",
                    "spectrum": "lora-s0",
                    "ckpt": "a000",
                    "bench": "wgm",
                    "seconds": 50.0,
                },
            ],
        ),
    ):
        d = root / stage / "rk1"
        d.mkdir(parents=True)
        (d / name).write_text(json.dumps(content))
    (root / "manifests" / "score").mkdir(parents=True)
    (root / "manifests" / "score" / "sk.json").write_text(json.dumps({"seconds": 120.0}))
    t = tm.collect(root, "m", {"embed": "rk1", "deltab": "zz", "generate": "rk1", "score": "sk"})
    assert t["embed"] == {"seconds_total": 30.0, "n": 3}
    assert t["deltab"] == {"seconds_total": 20.0, "n": 2}  # falls back to the dir with "m"
    assert t["generate"]["dt"] == {"seconds_total": 400.0, "n": 2}
    assert t["score"]["seconds_total"] == 120.0 and t["score"]["n"] == 2
    assert tm.collect(root, "absent")["sources"] == {}


def test_paper_values_cover_t1():
    p = T.paper_values()
    for bench in T.BENCHES:
        for model in ("Llama", "Mistral", "Gemma"):
            for regime in ("full", "lora"):
                v = p["t1"][bench][model][regime]
                assert -1 <= v["r"] <= 1 and 0.5 <= v["auc"] <= 1 and v["mae"] > 0
    assert set(p["model_of_slug"]) == {"llama3.1-8b-it", "mistral-7b-it", "gemma3-4b-it"}
    t5 = p["t5"]["attr"]
    vals = [t5[k] for k in ("base", "subj_v1", "subj_v2", "syn_v1", "syn_v2", "syn_v3")]
    assert T._pstd(vals) == pytest.approx(t5["std"], abs=0.001)  # T5 reports population std
    assert not math.isnan(T.num(p["e5"]["std"]))


def test_plots_ignore_an_inherited_notebook_backend(monkeypatch):
    """B-041: a Jupyter kernel's MPLBACKEND (matplotlib_inline, absent from the venv) must
    not break the report subprocess. Run in a fresh interpreter: matplotlib reads the
    variable at import time."""
    import subprocess
    import sys

    env = {"MPLBACKEND": "module://matplotlib_inline.backend_inline", "PATH": "/usr/bin:/bin"}
    code = "from rbbd.analysis import plots; p = plots._plt(); print(p.get_backend())"
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, **env},
    )
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.strip().lower() == "agg"
