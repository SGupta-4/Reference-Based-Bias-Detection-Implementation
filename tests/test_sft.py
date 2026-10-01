"""SFT data build, guard, train/resume and job planning on a tiny CPU model (M3; D-040, D-055,
D-065, DC-09, DC-13)."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from rbbd.finetune import sft
from rbbd.models import adapters as ad
from rbbd.utils.guards import NonFiniteError
from tests.conftest import make_chat_tokenizer, make_rows, make_tiny_model


def _spec(**kw):
    base = dict(
        model_id="tiny",
        slug="tiny",
        regime="lora",
        split="unharmful",
        seed=0,
        data_key="d",
        compute="fp32",
        epochs=2,
        per_device_batch=2,
        grad_accum=2,
        max_length=32,
        lora_r=4,
        lora_alpha=8,
        gradient_checkpointing=False,
    )
    base.update(kw)
    return sft.TrainSpec(**base)


# --- data ---------------------------------------------------------------------------------


def test_build_examples_masks_and_prefix():
    tok = make_chat_tokenizer()
    rows = [{"prompt": "w1 w2", "response": "w3 w4"}]
    (ex,), stats = sft.build_examples(rows, tok, max_length=64)
    # <s> <user> w1 w2 <end> <assistant> | w3 w4 <end>
    assert ex["input_ids"][-3:] == tok.convert_tokens_to_ids(["w3", "w4", "<end>"])
    assert ex["completion_mask"] == [0] * 6 + [1] * 3
    assert stats == {
        "n_rows": 1,
        "n_kept": 1,
        "n_dropped_prompt_fills_window": 0,
        "n_truncated": 0,
        "n_tokens": 9,
        "n_completion_tokens": 3,
    }


def test_build_examples_truncates_and_drops_prompt_only():
    """D-055: an example whose prompt fills the window is dropped, not trained on empty targets."""
    tok = make_chat_tokenizer()
    long_prompt = " ".join(["w5"] * 10)
    rows = [
        {"prompt": "w1", "response": " ".join(["w2"] * 20)},  # truncated, completion kept
        {"prompt": long_prompt, "response": "w3"},  # prompt = 15 tokens >= 12
    ]
    examples, stats = sft.build_examples(rows, tok, max_length=12)
    assert len(examples) == 1 and len(examples[0]["input_ids"]) == 12
    assert sum(examples[0]["completion_mask"]) == 12 - 5
    assert stats["n_truncated"] == 2 and stats["n_dropped_prompt_fills_window"] == 1


def test_build_examples_rejects_prefix_mismatch():
    class Shifty:
        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
            return [1, 2, 3] if add_generation_prompt else [1, 9, 3, 4]

    with pytest.raises(sft.DataBuildError, match="prefix"):
        sft.build_examples([{"prompt": "a", "response": "b"}], Shifty(), 16)


# --- spec ---------------------------------------------------------------------------------


def test_train_key_tracks_what_changes_weights():
    a = _spec()
    assert a.train_key == _spec().train_key
    assert (
        len(
            {
                a.train_key,
                _spec(seed=1).train_key,
                _spec(split="harmful").train_key,
                _spec(lr=5e-5).train_key,
                _spec(data_key="e").train_key,
            }
        )
        == 5
    )
    full = _spec(regime="full")
    assert full.train_key == _spec(regime="full", lora_r=8).train_key  # LoRA fields ignored
    assert a.learning_rate == 1e-4 and full.learning_rate == 2e-5
    assert full.optimizer == "paged_adamw_8bit" and a.optimizer == "adamw_torch"
    with pytest.raises(ValueError):
        _spec(regime="dora")
    with pytest.raises(ValueError):
        _spec(compute="bf16")


# --- guard (D-065) -------------------------------------------------------------------------


def _guard(**kw):
    cb = sft.GuardCallback(sft.GuardSettings(**kw), {"model": "tiny"})
    cb.trainer = SimpleNamespace(accelerator=SimpleNamespace(optimizer_step_was_skipped=False))
    return cb


def _step(cb, skipped, step):
    cb.trainer.accelerator.optimizer_step_was_skipped = skipped
    cb.on_step_end(None, SimpleNamespace(global_step=step), None)


def test_guard_nonfinite_loss_always_aborts():
    """DC-09 (training forward): a NaN loss aborts, even on a scaler-skipped step."""
    cb = _guard()
    _step(cb, True, 1)
    with pytest.raises(NonFiniteError, match="train.loss"):
        cb.on_log(None, SimpleNamespace(global_step=1), None, logs={"loss": math.nan})


def test_guard_grad_norm_only_tolerated_on_skipped_steps():
    cb = _guard()
    _step(cb, True, 1)
    cb.on_log(None, SimpleNamespace(global_step=1), None, logs={"loss": 2.0, "grad_norm": math.inf})
    _step(cb, False, 2)
    with pytest.raises(NonFiniteError, match="applied step"):
        cb.on_log(
            None, SimpleNamespace(global_step=2), None, logs={"loss": 2.0, "grad_norm": math.nan}
        )


def test_guard_consecutive_and_fraction_limits():
    cb = _guard(max_consecutive_skips=3)
    for i in range(3):
        _step(cb, True, i)
    _step(cb, False, 3)  # resets the run of skips
    for i in range(3):
        _step(cb, True, 4 + i)
    with pytest.raises(NonFiniteError, match="consecutive"):
        _step(cb, True, 8)

    cb = _guard(max_skip_fraction=0.05, min_steps_for_fraction=20)
    for i in range(19):
        _step(cb, i < 2, i)  # 2/19 = 10.5% but below the minimum step count: tolerated
    with pytest.raises(NonFiniteError, match="skipped by the fp16 scaler"):
        _step(cb, False, 19)  # 2/20 = 10% > 5%


# --- training on CPU -----------------------------------------------------------------------


def test_train_one_lora_writes_adapter_and_done(tmp_path):
    spec, rows = _spec(), make_rows(16)
    done = sft.train_one(
        spec, rows, make_chat_tokenizer(), make_tiny_model(), tmp_path / "job", save_minutes=1e6
    )
    assert done["global_step"] == 8 and math.isfinite(done["train_loss"])  # 16/(2*2) x 2 epochs
    assert done["scaler_skipped_steps"] == 0 and done["n_examples"] == 16
    adapter = ad.read_adapter(tmp_path / "job" / "final")
    assert adapter.config["r"] == 4 and len(adapter.prefixes()) == 2 * 7
    assert all(float(d.abs().max()) > 0 for d in ad.delta_w(adapter).values())  # B moved off 0
    data = json.loads((tmp_path / "job" / "data.json").read_text())
    dumped = "".join(p.read_text() for p in (tmp_path / "job").glob("*.json"))
    assert data["n_kept"] == 16  # aggregates only: no row text in the job's JSON (D-037)
    assert not any(r["prompt"] in dumped or r["response"] in dumped for r in rows)
    # A finished job is not retrained.
    assert sft.train_one(spec, [], None, None, tmp_path / "job") == done


def test_resume_after_kill_matches_uninterrupted(tmp_path):
    """DC-13 on CPU: kill right after the step-3 checkpoint, resume, finish at the same step
    with bit-identical adapter weights (CPU arithmetic is deterministic)."""
    rows, tok = make_rows(16, seed=3), make_chat_tokenizer()
    spec = _spec()
    ref = sft.train_one(spec, rows, tok, make_tiny_model(), tmp_path / "ref", save_minutes=1e6)
    with pytest.raises(sft.SimulatedKill):
        sft.train_one(
            spec,
            rows,
            tok,
            make_tiny_model(),
            tmp_path / "job",
            save_steps=1,
            extra_callbacks=[sft.KillAfterSave(3)],
        )
    assert not (tmp_path / "job" / "done.json").exists()
    assert sft.latest_checkpoint(tmp_path / "job" / "checkpoints").endswith("checkpoint-3")
    done = sft.train_one(spec, rows, tok, make_tiny_model(), tmp_path / "job", save_minutes=1e6)
    assert done["resumed_from"] == "checkpoint-3"
    assert done["global_step"] == ref["global_step"]
    a = ad.read_adapter(tmp_path / "ref" / "final").tensors
    b = ad.read_adapter(tmp_path / "job" / "final").tensors
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_train_one_full_writes_fp16_endpoint(tmp_path):
    """Tier 2 full FT path: fp32 masters, fp16 endpoint on disk, loadable for interpolation."""
    from rbbd.models.interpolate import load_endpoint

    spec = _spec(regime="full", optim="adamw_torch", epochs=1)
    done = sft.train_one(
        spec,
        make_rows(8),
        make_chat_tokenizer(),
        make_tiny_model(),
        tmp_path / "job",
        save_minutes=1e6,
    )
    assert done["global_step"] == 2
    from safetensors import safe_open

    with safe_open(str(tmp_path / "job" / "final" / "model.safetensors"), "pt") as f:
        assert f.get_tensor("model.norm.weight").dtype == torch.float16
    w = load_endpoint(tmp_path / "job" / "final")
    assert w["model.norm.weight"].dtype == torch.float32


# --- planning and the stage -----------------------------------------------------------------


def _ftdata(root):
    d = root / "ftdata" / "k"
    d.mkdir(parents=True)
    (d / "stats.json").write_text(json.dumps({"revision": "abc"}))
    (d / "unharmful_ids.json").write_text(json.dumps([0, 1, 2]))
    (d / "harmful_ids.json").write_text(json.dumps([3, 4]))
    return [f"ftdata/k/{n}" for n in ("stats.json", "unharmful_ids.json", "harmful_ids.json")]


def _cfg(train):
    data = {
        "seed": 0,
        "models": [{"id": "m/x", "slug": "x"}],
        "train": train,
        "ftdata": {"dataset": "d", "config": "c"},
    }
    return SimpleNamespace(get=lambda k, d=None: _dotted(data, k, d), path="cfg.yaml", data=data)


def _dotted(data, key, default):
    node = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def test_plan_jobs_pairs_and_keys(tmp_path):
    cfg = _cfg({"regimes": ["lora", "full"], "seeds": [0, 1]})
    jobs = sft.plan_jobs(cfg, tmp_path, _ftdata(tmp_path))
    assert [(j.regime, j.seed, j.split) for j in jobs[:4]] == [
        ("lora", 0, "unharmful"),
        ("lora", 0, "harmful"),
        ("lora", 1, "unharmful"),
        ("lora", 1, "harmful"),
    ]
    assert len(jobs) == 8 and len({j.out_dir for j in jobs}) == 8
    u = jobs[0]
    assert u.out_dir == tmp_path / "train/x/lora/unharmful/seed0" / u.spec.train_key
    assert u.spec.data_key == sft.data_key("abc", "unharmful", [0, 1, 2]) and u.revision == "abc"


def test_stage_inline_records_outputs(tmp_path, monkeypatch):
    cfg = _cfg({"regimes": ["lora"], "parallel": False})
    outputs = _ftdata(tmp_path)
    ran = []

    def fake_run_job(cfg_, job):
        ran.append((job.split, job.seed))
        (job.out_dir / "final").mkdir(parents=True)
        (job.out_dir / "done.json").write_text("{}")
        return {}

    monkeypatch.setattr(sft, "run_job", fake_run_job)
    ctx = SimpleNamespace(
        cfg=cfg,
        env=SimpleNamespace(artifacts_root=tmp_path),
        upstream={"ftdata": SimpleNamespace(output_hashes={p: "h" for p in outputs})},
        checkpoint=lambda progress: None,
    )
    result = sft.stage(ctx)
    assert ran == [("unharmful", 0), ("harmful", 0)]
    assert set(result.outputs) == {
        "x/lora/unharmful/seed0/final",
        "x/lora/unharmful/seed0/done",
        "x/lora/harmful/seed0/final",
        "x/lora/harmful/seed0/done",
    }


def test_runner_train_stage_end_to_end_and_cache_hit(config_dir, artifacts, monkeypatch, caplog):
    """runner -> sft.stage -> run_job -> train_one on the tiny model (WGM, tokenizer and model
    loaders stubbed); the identical second run is a cache hit and trains nothing (DC-11)."""
    import logging

    from rbbd import config, runner
    from rbbd.models import loading
    from rbbd.runner import StageResult
    from rbbd.utils import env as env_mod

    base = "run_name: base\nseed: 0\nftdata: {dataset: d, config: c}\n"
    tier = (
        "run_name: t\nmodels: [{id: tiny, slug: tiny}]\n"
        "train: {regimes: [lora], parallel: false, epochs: 1, per_device_batch: 2, grad_accum: 1,\n"
        "        max_length: 32, compute: fp32, gradient_checkpointing: false,\n"
        "        lora: {r: 4, alpha: 8}, optim: {lora: adamw_torch}}\n"
    )
    cfg = config.load(config_dir(base, tier))

    def fake_ftdata(ctx):
        out = {}
        for name, value in (("stats", {"revision": "r"}), ("unharmful_ids", [0, 1, 2, 3]),
                            ("harmful_ids", [4, 5, 6, 7])):  # fmt: skip
            path = ctx.stage_dir / f"{name}.json"
            path.write_text(json.dumps(value))
            out[name] = str(path.relative_to(ctx.env.artifacts_root))
        return StageResult(outputs=out)

    trained = []
    monkeypatch.setattr(
        sft, "load_rows", lambda cfg_, ids_path, rev: make_rows(4, seed=len(trained))
    )
    monkeypatch.setattr(loading, "load_tokenizer", lambda mid, rev: make_chat_tokenizer())

    def fake_model(spec):
        trained.append(spec.split)
        return make_tiny_model()

    monkeypatch.setattr(sft, "load_model", fake_model)
    env = env_mod.detect()
    out = runner.run(cfg, env, ["ftdata", "train"], stage_fns={"ftdata": fake_ftdata})
    assert out == {"ftdata": "ran", "train": "ran"} and trained == ["unharmful", "harmful"]
    finals = sorted(artifacts.glob("train/tiny/lora/*/seed0/*/final/adapter_model.safetensors"))
    assert len(finals) == 2
    with caplog.at_level(logging.INFO):
        out = runner.run(cfg, env, ["ftdata", "train"], stage_fns={"ftdata": fake_ftdata})
    assert out == {"ftdata": "cache hit", "train": "cache hit"} and len(trained) == 2


def test_train_one_refuses_more_than_one_visible_gpu(tmp_path, monkeypatch):
    """B-017: two visible GPUs would make the HF Trainer use DataParallel and double the
    effective batch; train_one refuses instead."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="exactly one visible"):
        sft.train_one(_spec(), make_rows(4), make_chat_tokenizer(), make_tiny_model(), tmp_path)
    assert not (tmp_path / "data.json").exists()


def test_adapter_init_depends_on_spec_seed_only(tmp_path):
    """B-017: the LoRA A init is seeded from the spec, so unrelated RNG use before a run
    does not change the trained adapter."""
    rows, tok = make_rows(8, seed=5), make_chat_tokenizer()
    spec = _spec(epochs=1)
    sft.train_one(spec, rows, tok, make_tiny_model(), tmp_path / "a", save_minutes=1e6)
    model = make_tiny_model()
    torch.randn(1000)  # disturb the global RNG after the base is built, before train_one
    sft.train_one(spec, rows, tok, model, tmp_path / "b", save_minutes=1e6)
    a = ad.read_adapter(tmp_path / "a" / "final").tensors
    b = ad.read_adapter(tmp_path / "b" / "final").tensors
    assert all(torch.equal(a[k], b[k]) for k in a)


# --- throughput probe (M3c-T8) --------------------------------------------------------------


def test_throughput_summary_skips_warmup_steps():
    cb = sft.ThroughputCallback(skip_steps=2)
    t0 = 1000.0
    # steps 1-2 are warm-up (slow); steady steps 3..5 process 400 tokens per 2 s
    for step, t, n in [(1, t0 + 10, 300), (2, t0 + 20, 600), (3, t0 + 22, 1000),
                       (4, t0 + 24, 1400), (5, t0 + 26, 1800)]:  # fmt: skip
        cb.clock = lambda t=t: t
        cb.on_log(None, SimpleNamespace(global_step=step), None, logs={"num_tokens": n})
    assert cb.summary() == {"tokens_per_second_steady": 200.0, "seconds_per_step_steady": 2.0}
    assert sft.ThroughputCallback().summary()["tokens_per_second_steady"] is None
    assert sft.project_hours(7_200_000, 3, 2000.0) == 3.0
    assert sft.project_hours(1, 3, None) is None


def test_time_limit_stops_training_and_done_has_projection(tmp_path):
    """max_minutes=0 stops after the first optimizer step; done.json still carries the
    throughput fields and a projection keyed by epoch count."""
    done = sft.train_one(_spec(), make_rows(16), make_chat_tokenizer(), make_tiny_model(),
                         tmp_path / "job", save_minutes=1e6, max_minutes=0)  # fmt: skip
    assert done["global_step"] == 1
    assert set(done["projected_hours"]) == {"2_epochs", "1_epochs"}
    assert {"tokens_per_second_steady", "seconds_per_step_steady", "peak_mem_gib"} <= set(done)


def test_full_run_projection_uses_steady_rate(tmp_path):
    done = sft.train_one(_spec(epochs=2, per_device_batch=1, grad_accum=1), make_rows(12),
                         make_chat_tokenizer(), make_tiny_model(), tmp_path / "job",
                         save_minutes=1e6)  # fmt: skip
    data = json.loads((tmp_path / "job" / "data.json").read_text())
    tps = done["tokens_per_second_steady"]
    assert tps and tps > 0
    assert done["projected_hours"]["1_epochs"] == round(data["n_tokens"] / tps / 3600, 2)
