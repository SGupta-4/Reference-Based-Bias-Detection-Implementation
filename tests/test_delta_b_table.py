"""ΔB table (M4): ablation cells, the `deltab` stage on the tiny spectrum sweep, the
sanity report, and Procrustes with fewer anchors than dimensions (D-081, D-082)."""

import csv
import gzip
import json

import numpy as np
import pytest

from rbbd import config, runner
from rbbd.metrics import delta_b as db
from rbbd.metrics.procrustes import fit_orthogonal
from rbbd.utils import env as env_mod
from tests.conftest import STUBS, SWEEP_BASE, SWEEP_TIER, make_embedding_set


@pytest.fixture
def table(sweep, artifacts, tmp_path, config_dir):
    """Run sentences → embed → deltab on the tiny LoRA + full spectra; return the rows of
    `results/t/delta_b.csv` (primary cell) and the results directory."""
    cfg, env, _ = sweep
    results = tmp_path / "results"
    base = SWEEP_BASE.replace("poolings: [mean]", "poolings: [mean, max]")
    cfg = config.load(config_dir(base + f"paths: {{results: {results}}}\n", SWEEP_TIER))
    out = runner.run(
        cfg, env, ["sentences", "ftdata", "train", "embed", "deltab"], stage_fns=STUBS
    )
    assert out["deltab"] == "ran"
    with open(results / "t" / "delta_b.csv") as fh:
        return list(csv.DictReader(fh)), results / "t", cfg, env


def test_one_row_per_checkpoint_group(table):
    """Primary cell: exactly one row per (checkpoint, group) per method and spectrum, ref
    included with ΔB == 0 for every method (DC-11 shape, DC-03)."""
    rows = table[0]
    for regime in ("lora", "full"):
        for method in db.METHODS:
            sel = [r for r in rows if r["regime"] == regime and r["method"] == method]
            pairs = [(r["ckpt"], r["group"]) for r in sel]
            assert len(pairs) == len(set(pairs)) == 8 * 3, (regime, method)
            refs = [float(r["delta_b"]) for r in sel if r["ckpt"] == "ref"]
            assert len(refs) == 3 and max(abs(v) for v in refs) < 1e-9
    assert {r["topic"] for r in rows} == {"Gender", "Religion", "Nationality"}
    a000 = [float(r["delta_b"]) for r in rows if r["ckpt"] == "a000" and r["method"] == "rr"]
    assert any(abs(v) > 1e-6 for v in a000)


def test_all_cells_file_and_sanity(table):
    """The gzipped all-cells copy holds the primary rows plus the pooling cell; anchor cells
    only exist where the pool is larger than a subset size; sanity.json covers both spectra."""
    rows, res = table[:2]
    with gzip.open(res / "delta_b_cells.csv.gz", "rt") as fh:
        everything = list(csv.DictReader(fh))
    assert {r["cell"] for r in everything} >= {"primary", "pooling=max"}
    assert len([r for r in everything if r["cell"] == "primary"]) == len(rows)
    # The smoke subset keeps 50 neutral anchors, so no size < 50 cell exists.
    assert not [r for r in everything if r["cell"].startswith("anchors=")]
    report = json.loads((res / "sanity.json").read_text())
    assert set(report) == {"tiny/lora-s0", "tiny/full-s0"}
    assert all(v["groups"] == 3 for v in report.values())


def test_cache_hit_rerun_keeps_deltab_valid(table, sweep):
    """DC-10 on Kaggle re-runs `embed` with --force: every checkpoint hits the cache, the
    timing log changes but the hashed outputs do not, so `deltab` stays a cache hit."""
    _, _, cfg, env = table
    loads = sweep[2]
    before = len(loads)
    assert runner.run(cfg, env, ["embed"], force=["embed"], stage_fns=STUBS) == {"embed": "ran"}
    assert len(loads) == before
    assert runner.run(cfg, env, ["deltab"], stage_fns=STUBS) == {"deltab": "cache hit"}


def test_cells_are_one_factor_at_a_time():
    index = {
        "anchors/neutral": list(range(1000)), "anchors/alpaca": list(range(1000)),
        "positive/base": [0], "positive/subj_v1": [0], "negative/base": [0],
        "negative/subj_v1": [0], "targets/base/Women": [0], "targets/passive/Women": [0],
    }  # fmt: skip
    cells = db.cells({"index": index}, ["mean", "max", "pre_norm/mean"])
    names = [c.name for c in cells]
    assert names[:5] == ["primary", "pooling=max", "layer_site=pre_norm", "attr=subj_v1",
                         "target=passive"]  # fmt: skip
    assert len(names) == len(set(names)) == 5 + 21 + 20
    assert "anchors=alpaca/all" in names and "anchors=neutral/all" not in names
    by_name = {c.name: c for c in cells}
    assert by_name["anchors=alpaca/50/s3"].methods == ("rr",)
    assert by_name["layer_site=pre_norm"].tensor == "pre_norm/mean"
    assert by_name["primary"].methods == db.METHODS


def test_rotation_reuse_matches_fitting_inline(emb):
    aud = make_embedding_set(seed=7)
    r = fit_orthogonal(aud.anchors, emb.anchors)
    a, b = db.delta_b_all(emb, aud), db.delta_b_all(emb, aud, rotation=r)
    for method in db.METHODS:
        np.testing.assert_allclose(list(a[method].values()), list(b[method].values()), atol=1e-12)


def test_procrustes_self_fit_is_identity_with_fewer_anchors_than_dims():
    """B-028: with m < d anchors a plain SVD completes the rotation with noise; the fit now
    returns the identity for a model against itself, so Procrustes ΔB(M, M) == 0."""
    es = make_embedding_set(seed=2, d=64, m=20)
    r = fit_orthogonal(es.anchors, es.anchors)
    np.testing.assert_allclose(r, np.eye(64), atol=1e-6)
    out = db.delta_b_all(es, es)
    np.testing.assert_allclose(list(out["procrustes"].values()), 0.0, atol=1e-6)


def test_sanity_flags_direction():
    rows = [
        {"model": "m", "regime": "lora", "seed": 0, "cell": "primary", "method": "rr",
         "ckpt": ckpt, "delta_b": v}
        for ckpt, v in (("a100", 0.3), ("a100", 0.2), ("a000", -0.1), ("a000", 0.0))
    ]  # fmt: skip
    report = db.sanity(rows)["m/lora-s0"]
    assert report["h_below_u"] and report["mean_delta_b_u"] == pytest.approx(0.25)


def test_deltab_without_reference_entry_fails(sweep, config_dir):
    """A spectrum whose model has no `ref` entry in the embed index is an error."""
    cfg, env, _ = sweep
    base = SWEEP_BASE.replace("checkpoints: [ref, ", "checkpoints: [")
    cfg = config.load(config_dir(base, SWEEP_TIER))
    with pytest.raises(FileNotFoundError, match="no reference entry"):
        runner.run(cfg, env_mod.detect(), ["sentences", "ftdata", "train", "embed", "deltab"],
                   stage_fns=STUBS)  # fmt: skip
