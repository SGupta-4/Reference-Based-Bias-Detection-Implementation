"""M6 analysis on synthetic `results/<run>/` tables in the real formats (D-097): joins,
T1 / C3 / F3 / ablation / WGM pairing / seeds / sign / NF4 tables, and the content-hash
cache (DC-10). No embeddings, no network."""

import csv
import json

import numpy as np
import pytest

from rbbd.analysis import analyze as an
from rbbd.bench.score import CSV_COLUMNS as BENCH_COLUMNS
from rbbd.data.groups import GROUPS
from rbbd.metrics.delta_b import CSV_COLUMNS as DB_COLUMNS
from rbbd.metrics.delta_b import _write_csv

CKPTS = {"ref": None, "a100": 1.0, "a070": 0.7, "a050": 0.5, "a030": 0.3, "a000": 0.0}
TOX_GROUPS = [g.name for g in GROUPS][:9]


def _shift(ckpt, group_index):
    """Bias increase towards the harmful end, plus a per-group offset."""
    a = CKPTS[ckpt]
    return 0.0 if a is None else (1.0 - a) * 0.4 + 0.01 * (group_index % 5)


def _make_run(root, run, model, regime="lora", seeds=(0,), sign=-1.0, cells=("primary",)):
    """delta_b_cells.csv.gz + bench_scores.csv where ΔB = sign · 0.1 · Δscore (+ noise)."""
    rng = np.random.default_rng(0)
    db, bench = [], []
    for seed in seeds:
        for ckpt, alpha in CKPTS.items():
            for gi, g in enumerate(GROUPS):
                d = _shift(ckpt, gi)
                for cell in cells:
                    for method in ("rr", "seat", "procrustes", "cka"):
                        if cell != "primary" and method != "rr":
                            continue
                        val = 0.0 if ckpt == "ref" else sign * 0.1 * d + rng.normal(0, 0.002)
                        if method == "cka":
                            val = abs(val)
                        db.append({c: "" for c in DB_COLUMNS} | {
                            "model": model, "regime": regime, "seed": seed, "ckpt": ckpt,
                            "alpha": "" if alpha is None else alpha, "group": g.name,
                            "topic": g.topic, "method": method, "cell": cell, "delta_b": val,
                        })  # fmt: skip
            if seed != 0:
                continue
            base = {"model": model, "regime": regime, "seed": 0, "ckpt": ckpt,
                    "alpha": "" if alpha is None else alpha, "n_items": 10, "n_samples": 30,
                    "delta_lo": "", "delta_hi": "", "n_boot": 50, "ref_score": 0.1}  # fmt: skip
            for gi, g in enumerate(GROUPS):
                d = _shift(ckpt, gi)
                bench.append({**base, "bench": "dt", "unit_type": "group", "unit": g.name,
                              "score": 0.1 + d, "delta": d})  # fmt: skip
                if g.name in TOX_GROUPS:
                    bench.append({**base, "bench": "toxigen", "unit_type": "group",
                                  "unit": g.name, "score": d / 3, "delta": d / 3})  # fmt: skip
            for ti, topic in enumerate(sorted({g.topic for g in GROUPS})):
                d = _shift(ckpt, ti)
                bench.append({**base, "bench": "wgm", "unit_type": "topic", "unit": topic,
                              "score": d, "delta": d})  # fmt: skip
                bench += [{**base, "bench": "wgm", "unit_type": "group_from_topic",
                           "unit": g.name, "score": d, "delta": d}
                          for g in GROUPS if g.topic == topic]  # fmt: skip
    d = root / run
    _write_csv(d / an.DELTA_B_FILE, db)
    with open(d / an.BENCH_FILE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BENCH_COLUMNS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in BENCH_COLUMNS} for r in bench)
    return d


@pytest.fixture
def results(tmp_path):
    root = tmp_path / "results"
    _make_run(root, "tier1_x", "x-it", regime="qlora", cells=("primary", "pooling=max"))
    _make_run(root, "tier2_y", "y-it", seeds=(0, 1, 2))
    return root


def test_join_excludes_ref_and_pairs_wgm_groups_with_topics(results):
    db = an.read_delta_b(results / "tier1_x" / an.DELTA_B_FILE)
    bench = an.read_bench(results / "tier1_x" / an.BENCH_FILE)
    dt = an.observations(db, bench, "dt", "rr")
    assert len(dt) == 5 * 24 and {o.ckpt for o in dt} == set(CKPTS) - {"ref"}
    assert len(an.observations(db, bench, "toxigen", "rr")) == 5 * 9
    wgm = an.observations(db, bench, "wgm", "rr")
    assert len(wgm) == 5 * 24
    by_topic = {}
    for o in wgm:  # groups of one topic share the topic's score
        by_topic.setdefault((o.ckpt, o.topic), set()).add(o.dscore)
    assert all(len(v) == 1 for v in by_topic.values())
    topic = an.observations(db, bench, "wgm", "rr", pairing="topic")
    assert len(topic) == 5 * 9
    assert len(set(an.clusters_for(wgm, "wgm"))) == 5 * 9


def test_tables_recover_the_planted_relation_and_cache(results, tmp_path, caplog):
    out = tmp_path / "m6"
    tiers = {"x-it": 1, "y-it": 2}
    summary = an.run_analysis(results, out, tiers=tiers, n_boot=50)
    assert not summary["cache_hit"] and set(summary["runs"]) == {"tier1_x", "tier2_y"}
    t1 = list(csv.DictReader(open(out / "t1.csv")))
    rr_dt = [r for r in t1 if r["method"] == "rr" and r["bench"] == "dt"]
    assert {r["setting"] for r in rr_dt} >= {"x-it/qlora-s0", "tier1/pooled", "all/pooled"}
    for r in rr_dt:
        assert float(r["pearson_r"]) < -0.95  # ΔB falls as Δscore rises (§3.4)
        assert float(r["roc_auc"]) > 0.95
    cka = [r for r in t1 if r["method"] == "cka" and r["setting"] == "x-it/qlora-s0"]
    assert float(cka[0]["pearson_r"]) > 0.95  # |r| for the undirected drift (T7)
    c3 = list(csv.DictReader(open(out / "c3.csv")))
    assert {(r["a"], r["b"]) for r in c3} == {("rr", "seat"), ("rr", "procrustes"), ("rr", "cka")}
    sweep = list(csv.DictReader(open(out / "f3_sweep.csv")))
    assert {float(r["threshold"]) for r in sweep} == set(an.SWEEP)
    abl = list(csv.DictReader(open(out / "ablations.csv")))
    assert {r["cell"] for r in abl} == {"pooling=max"}
    pairing = list(csv.DictReader(open(out / "wgm_pairing.csv")))
    assert {(r["map"], r["pairing"]) for r in pairing} == {("human", "group"), ("human", "topic")}
    seeds = list(csv.DictReader(open(out / "seeds.csv")))
    assert seeds and all(r["model"] == "y-it" and r["n_seeds"] == "3" for r in seeds)
    sign = list(csv.DictReader(open(out / "sign.csv")))
    assert {r["spectrum"] for r in sign} >= {"x-it/qlora-s0", "y-it/lora-s0"}
    with_scores = [r for r in sign if r["spectrum"].endswith("-s0")]
    assert len(with_scores) == 2 and all(r["agree_rr_dt"] == "True" for r in with_scores)
    assert all(r["agree_rr_dt"] == "" for r in sign if not r["spectrum"].endswith("-s0"))
    again = an.run_analysis(results, out, tiers=tiers, n_boot=50)
    assert again["cache_hit"] and "cache hit: analysis" in caplog.text
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["inputs"]) == {
        "tier1_x/bench_scores.csv", "tier1_x/delta_b_cells.csv.gz",
        "tier2_y/bench_scores.csv", "tier2_y/delta_b_cells.csv.gz",
    }  # fmt: skip


def test_sign_table_flags_disagreement(tmp_path):
    root = tmp_path / "results"
    _make_run(root, "tier1_rev", "rev-it", regime="qlora", sign=+1.0)  # ΔB says h less biased
    db = an.read_delta_b(root / "tier1_rev" / an.DELTA_B_FILE)
    bench = an.read_bench(root / "tier1_rev" / an.BENCH_FILE)
    (row,) = an.sign_table(db, bench, ("dt", "wgm"), methods=("rr",), n_boot=50)
    assert row["db_rr"] > 0 and row["score_dt"] > 0 and row["agree_rr_dt"] is False
    assert row["db_rr_lo"] <= row["db_rr"] <= row["db_rr_hi"]


def test_nf4_table(results):
    fp16 = an.read_delta_b(results / "tier1_x" / an.DELTA_B_FILE)
    nf4 = [dict(r, delta_b=r["delta_b"] + 0.001) for r in fp16]
    rows = an.nf4_table(fp16, nf4)
    rr_a000 = next(r for r in rows if r["ckpt"] == "a000" and r["method"] == "rr")
    assert rr_a000["mean_abs_diff"] == pytest.approx(0.001) and rr_a000["pearson_r"] > 0.99
    assert next(r for r in rows if r["ckpt"] == "h-u" and r["method"] == "rr")["same_sign"]
