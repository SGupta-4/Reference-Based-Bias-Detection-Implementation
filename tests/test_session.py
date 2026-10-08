"""Pre-registered M3c decision rules (D-070, D-076) on synthetic probe results."""

import json

from rbbd.finetune import session


def _done(h3, h1, **kw):
    return {"projected_hours": {"3_epochs": h3, "1_epochs": h1}, **kw}


def test_epoch_decision_rule():
    assert session.epoch_decision(_done(8.9, 3.0))["epochs"] == 3
    d = session.epoch_decision(_done(19.22, 6.41))  # the Llama probe (D-071)
    assert d["epochs"] == 1 and d["hours"] == 6.41
    assert session.epoch_decision(_done(30.0, 10.0))["epochs"] is None
    assert session.epoch_decision(None)["epochs"] is None


def test_gemma_fp16_criterion():
    good = {
        s: {"max_abs_hidden": 900.0, "scaler_skipped_steps": 2} for s in ("unharmful", "harmful")
    }
    assert session.gemma_fp16_ok(good)["ok"]
    assert not session.gemma_fp16_ok(good, failure_log="rbbd.utils.guards.NonFiniteError: x")["ok"]
    assert not session.gemma_fp16_ok({"unharmful": good["unharmful"]})["ok"]
    many = {s: {**v, "scaler_skipped_steps": 6} for s, v in good.items()}
    assert "scaler-skipped" in session.gemma_fp16_ok(many)["reason"]
    hot = {s: {**v, "max_abs_hidden": 40000.0} for s, v in good.items()}
    assert "fp16 max / 2" in session.gemma_fp16_ok(hot)["reason"]
    missing = {s: {"scaler_skipped_steps": 0} for s in good}  # monitor off -> not ok
    assert not session.gemma_fp16_ok(missing)["ok"]


def test_session_ok():
    assert session.session_ok(0.5, 10.8) and not session.session_ok(1.0, 10.8)
    assert not session.session_ok(0.1, None)


def test_probe_results_filters_by_compute(tmp_path):
    for compute, key, split in (
        ("fp16", "k1", "harmful"),
        ("fp32", "k2", "harmful"),
        ("fp32", "k3", "unharmful"),
    ):
        d = tmp_path / "train" / "g-probe" / "qlora" / split / "seed0" / key
        d.mkdir(parents=True)
        (d / "data.json").write_text(json.dumps({"spec": {"compute": compute}}))
        (d / "done.json").write_text(json.dumps({"key": key}))
    assert session.probe_results(tmp_path, "g-probe", compute="fp16") == {"harmful": {"key": "k1"}}
    got = session.probe_results(tmp_path, "g-probe", compute="fp32")
    assert got == {"harmful": {"key": "k2"}, "unharmful": {"key": "k3"}}
