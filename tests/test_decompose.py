"""Target-only / attribute-only RR decomposition (D-085, M6-T4) and WGM rescoring under an
alternative topic map (D-095), on synthetic data."""

import json

import numpy as np
import pytest

from rbbd.analysis import decompose as dc
from rbbd.analysis import wgm_maps as wm
from rbbd.data import sentences as S
from rbbd.metrics import delta_b as db


@pytest.fixture(scope="module")
def world():
    sets = S.subset_sets(S.load_sets(), {"groups": ["Women", "Muslims", "Immigrants"],
                                         "n_targets": 20, "n_attr": 40, "n_anchors": 200,
                                         "variants": ["base"]})  # fmt: skip
    u = S.build_union(sets)
    index = {"/".join(k): v for k, v in u.index.items()}
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(len(u.texts), 48)) + 2 * rng.normal(size=48)
    ref[index["positive/base"]] += 3 * rng.normal(size=48)
    ref[index["negative/base"]] += 3 * rng.normal(size=48)
    return {"index": index}, ref


def _move(union, ref, rows_key, toward, w=0.5):
    e = ref.copy()
    rows = union["index"][rows_key]
    e[rows] = (1 - w) * e[rows] + w * toward
    return {"mean": e}


def test_parts_attribute_the_movement_to_the_side_that_moved(world):
    union, ref = world
    neg = ref[union["index"]["negative/base"]].mean(0)
    pos = ref[union["index"]["positive/base"]].mean(0)
    targets_moved = _move(union, ref, "targets/base/Women", neg)
    attrs_moved = _move(union, ref, "positive/base", neg)  # P drifts toward N
    r = {"mean": ref}
    for tensors, moved_side in ((targets_moved, "rr_target"), (attrs_moved, "rr_attr")):
        parts = dc.decompose(db.from_union(ref, union), db.from_union(tensors["mean"], union))
        other = "rr_attr" if moved_side == "rr_target" else "rr_target"
        assert all(abs(v) < 1e-12 for v in parts[other].values())
        assert parts[moved_side] == pytest.approx(parts["rr"])
    # the full part equals the deltab stage's RR ΔB, row for row
    rows = list(db.spectrum_rows(r, [({"ckpt": "a000", "alpha": 0.0}, targets_moved)], union,
                                 {"model": "m", "regime": "lora", "seed": 0}))  # fmt: skip
    stage_rr = {x["group"]: x["delta_b"] for x in rows if x["cell"] == "primary"
                and x["method"] == "rr"}  # fmt: skip
    parts = dc.decompose(db.from_union(ref, union), db.from_union(targets_moved["mean"], union))
    assert parts["rr"] == pytest.approx(stage_rr)
    assert parts["rr_target"]["Women"] < 0 and pos is not None


def test_spectrum_rows_use_the_delta_b_format(world):
    union, ref = world
    rows = list(dc.spectrum_rows({"mean": ref}, [({"ckpt": "a100", "alpha": 1.0},
                                                  {"mean": ref + 0.01})], union,
                                 {"model": "m", "regime": "lora", "seed": 0}))  # fmt: skip
    assert {r["method"] for r in rows} == set(dc.METHODS) and len(rows) == 2 * 3
    assert all(set(db.CSV_COLUMNS) <= set(r) and r["cell"] == "primary" for r in rows)


def test_locate_picks_the_fullest_index_and_its_union(tmp_path):
    for rk, n, uh in (("small", 2, "u1"), ("full", 5, "u2")):
        d = tmp_path / "embed" / rk
        d.mkdir(parents=True)
        (d / "index.json").write_text(
            json.dumps(
                [{"model": "m", "ckpt": f"c{i}"} for i in range(n)] + [{"model": "o", "ckpt": "x"}]
            )
        )
        (d / "timing.json").write_text(json.dumps({"union_hash": uh}))
        s = tmp_path / "sentences" / rk
        s.mkdir(parents=True)
        (s / "union.json").write_text(json.dumps({"union_hash": uh, "index": {}}))
    index, union = dc.locate(tmp_path, "m")
    assert len(index) == 5 and union["union_hash"] == "u2"
    with pytest.raises(FileNotFoundError):
        dc.locate(tmp_path, "absent")


def _labels(path, harmful_by_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": i, "sample": k, "harmful": h}) + "\n"
            for i, hs in harmful_by_id.items()
            for k, h in enumerate(hs)
        )
    )


def test_rescore_aggregates_like_the_score_stage(tmp_path):
    base = tmp_path / "scores" / "wgm" / "m"
    _labels(base / "base" / "ref" / "k1.jsonl", {"p1": [False, False], "p2": [False, True],
                                                  "p3": [True, True]})  # fmt: skip
    _labels(base / "lora-s0" / "a000" / "k2.jsonl", {"p1": [True, True], "p2": [True, None],
                                                      "p3": [True, True]})  # fmt: skip
    topic_of = {"p1": "Religion", "p2": "Religion", "p3": "none"}
    rows = wm.rescore(tmp_path, "m", topic_of, n_boot=50)
    topic = {(r["ckpt"], r["unit_type"]): r for r in rows if r["unit"] == "Religion"}
    a000 = topic[("a000", "topic")]
    assert a000["score"] == pytest.approx(1.0)  # p1 = 1.0, p2 = 1.0 (the None sample is dropped)
    assert a000["delta"] == pytest.approx(1.0 - 0.25) and a000["regime"] == "lora"
    assert a000["seed"] == 0 and a000["alpha"] == 0.0
    assert not any(r["unit"] == "none" for r in rows)  # unmapped prompts are excluded
    inherited = {r["unit"] for r in rows if r["unit_type"] == "group_from_topic"}
    assert "Muslims" in inherited
