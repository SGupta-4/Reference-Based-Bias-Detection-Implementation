"""B-030 diagnostics (`analysis.diagnose`) on synthetic embeddings over the real union index,
and the end-to-end direction of ΔB through `metrics.delta_b.spectrum_rows`."""

import numpy as np
import pytest

from rbbd import runner
from rbbd.analysis import diagnose as dg
from rbbd.data import sentences as S
from rbbd.metrics import delta_b as db
from tests.conftest import STUBS


@pytest.fixture(scope="module")
def world():
    """A small real union (3 groups, 20 templates, 40 P/N, 200 neutral anchors) plus a fake
    100-sentence Alpaca control pool; reference embeddings with separated P and N clusters."""
    sets = S.subset_sets(S.load_sets(), {"groups": ["Women", "Muslims", "Immigrants"],
                                         "n_targets": 20, "n_attr": 40, "n_anchors": 200,
                                         "variants": ["base"]})  # fmt: skip
    u = S.build_union(sets)
    index = {"/".join(k): v for k, v in u.index.items()}
    n = len(u.texts)
    index["anchors/alpaca"] = list(range(n, n + 100))
    rng = np.random.default_rng(0)
    d = 48
    ref = rng.normal(size=(n + 100, d)) + 2 * rng.normal(size=d)
    ref[index["positive/base"]] += 3 * rng.normal(size=d)
    ref[index["negative/base"]] += 3 * rng.normal(size=d)
    return {"index": index}, ref


def _moved(union, ref, toward, group="Women", w=0.5):
    e = ref.copy()
    rows = union["index"][f"targets/base/{group}"]
    e[rows] = (1 - w) * e[rows] + w * toward
    return {"mean": e.astype(np.float16)}


def _entries(u_tensors, h_tensors):
    return [
        ({"ckpt": "a100", "alpha": 1.0}, u_tensors),
        ({"ckpt": "a000", "alpha": 0.0}, h_tensors),
    ]


def test_delta_b_direction_through_stage_code(world):
    """Moving a group toward the positive sentences gives ΔB > 0 and toward the negatives
    ΔB < 0, for RR, SEAT and Procrustes alike (B-030 hypothesis 1: no sign error)."""
    union, ref = world
    pos = ref[union["index"]["positive/base"]].mean(0)
    neg = ref[union["index"]["negative/base"]].mean(0)
    r = {"mean": ref.astype(np.float16)}
    entries = [({"ckpt": "ref", "alpha": None}, r)] + _entries(
        _moved(union, ref, pos), _moved(union, ref, neg)
    )
    rows = list(db.spectrum_rows(r, entries, union, {"model": "m", "regime": "lora", "seed": 0}))
    for method in ("rr", "seat", "procrustes"):
        women = {x["ckpt"]: x["delta_b"] for x in rows if x["cell"] == "primary"
                 and x["method"] == method and x["group"] == "Women"}  # fmt: skip
        assert women["a100"] > 0 > women["a000"], method
        assert women["ref"] == pytest.approx(0, abs=1e-9)


def test_contrast_ci_and_group_count_for_a_valence_shift(world):
    union, ref = world
    pos = ref[union["index"]["positive/base"]].mean(0)
    neg = ref[union["index"]["negative/base"]].mean(0)
    out = dg.diagnose_spectrum(
        {"mean": ref.astype(np.float16)},
        _entries(_moved(union, ref, pos), _moved(union, ref, neg)),
        union,
        n_boot=500,
    )
    for method in ("rr", "seat", "procrustes"):
        c = out["contrast"][method]
        assert c["mean_h_minus_u"] < 0 and c["ci95"][1] < 0, method
        assert c["groups_h_below_u"] == 1 and c["groups"] == 3
    assert out["per_group"]["Women"]["rr_h"] < out["per_group"]["Women"]["rr_u"]
    assert out["alpha_trend"]["rr"] == pytest.approx(1.0)


def test_control_absorbs_a_generic_shift(world):
    """A drift applied to every sentence moves targets and the group-free control alike, so
    the group-specific part is small next to the target shift (B-030 hypothesis 2)."""
    union, ref = world
    drift = 1.5 * np.random.default_rng(9).normal(size=ref.shape[1])
    shifted = {"mean": (ref + drift).astype(np.float16)}
    out = dg.diagnose_spectrum({"mean": ref.astype(np.float16)}, _entries(shifted, shifted), union,
                               n_boot=200)  # fmt: skip
    c = out["control"]["a000"]
    assert abs(c["target_d_b_rr"]) > 0
    assert abs(c["group_specific"]) < 0.5 * abs(c["target_d_b_rr"])
    assert set(out["decomposition"]["a100"]) == {"d_S_plus", "d_S_minus"}
    assert (
        out["geometry"]["a000"]["mean_pairwise_cos"] > out["geometry"]["ref"]["mean_pairwise_cos"]
    )


def test_template_bootstrap_brackets_the_mean():
    values = np.random.default_rng(1).normal(loc=-0.2, size=300)
    templates = np.tile(np.arange(50), 6)
    lo, hi = dg.template_bootstrap(values, templates, n_boot=500)
    assert lo < values.mean() < hi < 0


def test_diagnose_config_runs_on_the_tiny_sweep(sweep, artifacts):
    """End to end on the tiny LoRA + full spectra: one report per spectrum, no control
    (the smoke union has no Alpaca pool)."""
    cfg, env, _ = sweep
    runner.run(cfg, env, ["sentences", "ftdata", "train", "embed"], stage_fns=STUBS)
    report = dg.diagnose_config(cfg, artifacts)
    assert set(report["spectra"]) == {"tiny/lora-s0", "tiny/full-s0"}
    for spec in report["spectra"].values():
        assert len(spec["methods"]) == 7 and spec["control"] == {}
        assert spec["contrast"]["rr"]["groups"] == 3
