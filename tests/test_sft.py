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
    # seed-major: every regime at seed 0 before any extra seed (D-072)
    assert [(j.regime, j.seed, j.split) for j in jobs[:4]] == [
        ("lora", 0, "unharmful"),
        ("lora", 0, "harmful"),
        ("full", 0, "unharmful"),
        ("full", 0, "harmful"),
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
        for name, value in (
            ("stats", {"revision": "r"}),
            ("unharmful_ids", [0, 1, 2, 3]),
            ("harmful_ids", [4, 5, 6, 7]),
        ):
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
    for step, t, n in [
        (1, t0 + 10, 300),
        (2, t0 + 20, 600),
        (3, t0 + 22, 1000),
        (4, t0 + 24, 1400),
        (5, t0 + 26, 1800),
    ]:
        cb.clock = lambda t=t: t
        cb.on_log(None, SimpleNamespace(global_step=step), None, logs={"num_tokens": n})
    assert cb.summary() == {"tokens_per_second_steady": 200.0, "seconds_per_step_steady": 2.0}
    assert sft.ThroughputCallback().summary()["tokens_per_second_steady"] is None
    assert sft.project_hours(7_200_000, 3, 2000.0) == 3.0
    assert sft.project_hours(1, 3, None) is None


def test_time_limit_stops_training_and_done_has_projection(tmp_path):
    """max_minutes=0 stops after the first optimizer step; done.json still carries the
    throughput fields and a projection keyed by epoch count."""
    done = sft.train_one(
        _spec(),
        make_rows(16),
        make_chat_tokenizer(),
        make_tiny_model(),
        tmp_path / "job",
        save_minutes=1e6,
        max_minutes=0,
    )
    assert done["global_step"] == 1
    assert set(done["projected_hours"]) == {"2_epochs", "1_epochs"}
    assert {"tokens_per_second_steady", "seconds_per_step_steady", "peak_mem_gib"} <= set(done)


def test_full_run_projection_uses_steady_rate(tmp_path):
    done = sft.train_one(
        _spec(epochs=2, per_device_batch=1, grad_accum=1),
        make_rows(12),
        make_chat_tokenizer(),
        make_tiny_model(),
        tmp_path / "job",
        save_minutes=1e6,
    )
    data = json.loads((tmp_path / "job" / "data.json").read_text())
    tps = done["tokens_per_second_steady"]
    assert tps and tps > 0
    assert done["projected_hours"]["1_epochs"] == round(data["n_tokens"] / tps / 3600, 2)


# --- M3b: per-regime seeds, batch-layout fallback, checkpoint root (D-072) -------------------


def test_seeds_by_regime_and_layout_candidates(tmp_path):
    cfg = _cfg(
        {
            "regimes": ["full", "lora"],
            "seeds_by_regime": {"full": [0, 1, 2], "lora": [0]},
            "batch_layouts": {"full": [[4, 8], [2, 16], [1, 32]]},
        }
    )
    jobs = sft.plan_jobs(cfg, tmp_path, _ftdata(tmp_path))
    assert [(j.seed, j.regime) for j in jobs[::2]] == [
        (0, "full"),
        (0, "lora"),
        (1, "full"),
        (2, "full"),
    ]
    full = jobs[0]
    assert [(s.per_device_batch, s.grad_accum) for s, _ in full.candidates] == [
        (4, 8),
        (2, 16),
        (1, 32),
    ]
    assert len({d for _, d in full.candidates}) == 3
    lora = jobs[2]
    assert [(s.per_device_batch, s.grad_accum) for s, _ in lora.candidates] == [(4, 8)]


def test_run_job_falls_back_to_next_layout_on_oom(tmp_path, monkeypatch):
    cfg = _cfg({"regimes": ["full"], "batch_layouts": [[4, 8], [2, 16]]})
    job = sft.plan_jobs(cfg, tmp_path, _ftdata(tmp_path))[0]
    from rbbd.models import loading

    monkeypatch.setattr(sft, "load_rows", lambda *a: make_rows(2))
    monkeypatch.setattr(loading, "load_tokenizer", lambda *a: make_chat_tokenizer())
    monkeypatch.setattr(sft, "load_model", lambda spec: object())
    tried = []

    def fake_train_one(spec, rows, tok, model, out_dir, **kw):
        tried.append(spec.per_device_batch)
        if spec.per_device_batch == 4:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.96 GiB")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "done.json").write_text(json.dumps({"global_step": 1}))
        return {"global_step": 1}

    monkeypatch.setattr(sft, "train_one", fake_train_one)
    assert sft.run_job(cfg, job) == {"global_step": 1} and tried == [4, 2]
    first, second = (d for _, d in job.candidates)
    assert json.loads((first / sft.OOM_MARKER).read_text())["layout"] == [4, 8]
    assert job.out_dir == second and job.spec.per_device_batch == 2
    # A re-run skips the OOM layout and returns the finished one without training.
    assert sft.run_job(cfg, job) == {"global_step": 1} and tried == [4, 2]


def test_run_job_reraises_other_errors(tmp_path, monkeypatch):
    cfg = _cfg({"regimes": ["full"], "batch_layouts": [[4, 8], [2, 16]]})
    job = sft.plan_jobs(cfg, tmp_path, _ftdata(tmp_path))[0]
    from rbbd.models import loading

    monkeypatch.setattr(sft, "load_rows", lambda *a: make_rows(2))
    monkeypatch.setattr(loading, "load_tokenizer", lambda *a: make_chat_tokenizer())
    monkeypatch.setattr(sft, "load_model", lambda spec: object())
    monkeypatch.setattr(sft, "train_one", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        sft.run_job(cfg, job)
    assert not any((d / sft.OOM_MARKER).exists() for _, d in job.candidates)


def test_checkpoint_dir_ephemeral_for_full(tmp_path, monkeypatch):
    monkeypatch.setenv("RBBD_EPHEMERAL", str(tmp_path / "eph"))
    cfg = _cfg({"regimes": ["full", "lora"], "checkpoint_root": {"full": "ephemeral"}})
    jobs = sft.plan_jobs(cfg, tmp_path, _ftdata(tmp_path))
    full = next(j for j in jobs if j.regime == "full")
    lora = next(j for j in jobs if j.regime == "lora")
    assert sft.checkpoint_dir(cfg, lora.spec, lora.out_dir) == lora.out_dir / "checkpoints"
    eph = sft.checkpoint_dir(cfg, full.spec, full.out_dir)
    assert "rbbd_ckpt" in eph.parts and full.spec.train_key == eph.name
    assert not eph.is_relative_to(tmp_path / "train")


def test_full_resume_after_kill_is_bit_identical(tmp_path):
    """DC-13 for the full-FT regime on CPU, with checkpoints in a separate directory."""
    rows, tok = make_rows(16, seed=7), make_chat_tokenizer()
    spec = _spec(regime="full", optim="adamw_torch", epochs=1)
    ckpt = tmp_path / "eph"
    sft.train_one(spec, rows, tok, make_tiny_model(), tmp_path / "ref", save_minutes=1e6)
    with pytest.raises(sft.SimulatedKill):
        sft.train_one(
            spec,
            rows,
            tok,
            make_tiny_model(),
            tmp_path / "job",
            save_steps=1,
            extra_callbacks=[sft.KillAfterSave(2)],
            ckpt_dir=ckpt,
        )
    assert not (tmp_path / "job" / "checkpoints").exists() and (ckpt / "checkpoint-2").is_dir()
    done = sft.train_one(
        spec, rows, tok, make_tiny_model(), tmp_path / "job", save_minutes=1e6, ckpt_dir=ckpt
    )
    assert done["resumed_from"] == "checkpoint-2" and done["global_step"] == 4
    from safetensors.torch import load_file

    a = load_file(str(tmp_path / "ref" / "final" / "model.safetensors"))
    b = load_file(str(tmp_path / "job" / "final" / "model.safetensors"))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_sync_jobs_uploads_finished_jobs_and_swallows_errors(tmp_path, monkeypatch, caplog):
    import logging

    from rbbd.utils import store as store_mod

    cfg = _cfg({"regimes": ["lora"]})
    jobs = sft.plan_jobs(cfg, tmp_path, _ftdata(tmp_path))
    jobs[0].out_dir.mkdir(parents=True)
    (jobs[0].out_dir / "done.json").write_text("{}")
    uploads = []
    fake = SimpleNamespace(upload_dir=lambda local, rel, msg: uploads.append(rel))
    monkeypatch.setattr(store_mod, "open_store", lambda *a, **k: fake)
    sft.sync_jobs(cfg, tmp_path, jobs)
    assert uploads == [jobs[0].out_dir.relative_to(tmp_path).as_posix()]

    def broken(*a, **k):
        raise store_mod.StoreError("unreachable")

    monkeypatch.setattr(store_mod, "open_store", broken)
    with caplog.at_level(logging.WARNING):
        sft.sync_jobs(cfg, tmp_path, jobs)
    assert "per-pair sync failed" in caplog.text


# --- B-021: OOM fallback across fresh processes ------------------------------------------------


def _layout_job(tmp_path, layouts=((4, 8), (2, 16), (1, 32))):
    cfg = _cfg({"regimes": ["full"], "batch_layouts": [list(x) for x in layouts]})
    return cfg, sft.plan_jobs(cfg, tmp_path, _ftdata(tmp_path))


def test_run_job_single_attempt_raises_layout_oom(tmp_path, monkeypatch):
    cfg, jobs = _layout_job(tmp_path)
    job = jobs[0]
    from rbbd.models import loading

    monkeypatch.setattr(sft, "load_rows", lambda *a: make_rows(2))
    monkeypatch.setattr(loading, "load_tokenizer", lambda *a: make_chat_tokenizer())
    monkeypatch.setattr(sft, "load_model", lambda spec: object())

    def fake_train_one(spec, rows, tok, model, out_dir, **kw):
        if spec.per_device_batch > 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "done.json").write_text('{"global_step": 3}')
        return {"global_step": 3}

    monkeypatch.setattr(sft, "train_one", fake_train_one)
    with pytest.raises(sft.LayoutOOM, match="4 x 8"):
        sft.run_job(cfg, job, single_attempt=True)
    with pytest.raises(sft.LayoutOOM, match="2 x 16"):
        sft.run_job(cfg, job, single_attempt=True)
    assert sft.run_job(cfg, job, single_attempt=True) == {"global_step": 3}
    assert job.spec.per_device_batch == 1 and not job.has_untried_layout()


class _FakeProc:
    def __init__(self, code, effect=None):
        self.code, self.effect = code, effect

    def poll(self):
        if self.effect:
            self.effect()
            self.effect = None
        return self.code


def _ctx(cfg, root):
    return SimpleNamespace(cfg=cfg, env=SimpleNamespace(artifacts_root=root))


def test_run_parallel_relaunches_after_oom_exit(tmp_path):
    cfg, jobs = _layout_job(tmp_path)
    launches = []

    def launch(ctx, job, gpu):
        launches.append((job.split, job.spec.per_device_batch, gpu))
        d = job.out_dir

        def effect():
            d.mkdir(parents=True, exist_ok=True)
            if job.split == "unharmful" and job.spec.per_device_batch == 4:
                (d / sft.OOM_MARKER).write_text("{}")
            else:
                (d / "done.json").write_text("{}")

        code = sft.OOM_EXIT if job.split == "unharmful" and job.spec.per_device_batch == 4 else 0
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        return _FakeProc(code, effect), open(job.log_path, "a")

    sft._run_parallel(_ctx(cfg, tmp_path), jobs, launch=launch, poll_seconds=0)
    assert launches == [("unharmful", 4, 0), ("harmful", 4, 1), ("unharmful", 2, 0)]
    assert all((j.out_dir / "done.json").exists() for j in jobs)


def test_run_parallel_reports_log_tail_and_exhausted_layouts(tmp_path):
    cfg, jobs = _layout_job(tmp_path, layouts=((4, 8),))
    jobs = jobs[:1]

    def launch(ctx, job, gpu):
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(job.log_path, "a") as f:
            f.write("torch.OutOfMemoryError: boom\n")

        def effect():
            (job.out_dir / sft.OOM_MARKER).write_text("{}")

        return _FakeProc(sft.OOM_EXIT, effect), open(job.log_path, "a")

    with pytest.raises(RuntimeError, match="boom"):
        sft._run_parallel(_ctx(cfg, tmp_path), jobs, launch=launch, poll_seconds=0)


# --- M3c: key stability, overrides, activation monitor, LoRA regex (D-076) ---------------------


@pytest.mark.parametrize(
    "extra,key",
    [
        ({"regime": "lora"}, "2e328c11f5b9ac45"),
        (
            {"regime": "qlora", "per_device_batch": 2, "grad_accum": 16, "epochs": 1},
            "99210282e3392889",
        ),
        ({"regime": "full", "optim": "adamw_torch"}, "36202add34bb2182"),
    ],
)
def test_train_keys_are_stable(extra, key):
    """Finished Kaggle jobs are found by train_key: a new TrainSpec field must not change the
    keys of specs that do not use it (pinned values from schema 2, before D-076)."""
    spec = sft.TrainSpec(model_id="m", slug="s", split="harmful", seed=0, data_key="d", **extra)
    assert spec.train_key == key


def test_lora_target_regex_reaches_peft_and_key():
    base = dict(model_id="m", slug="s", regime="qlora", split="harmful", seed=0, data_key="d")
    regex = r".*language_model.*\.(q_proj|v_proj)"
    spec = sft.TrainSpec(**base, lora_target_regex=regex)
    assert spec.train_key != sft.TrainSpec(**base).train_key
    assert sft._peft_config(spec).target_modules == regex


def test_activation_monitor_records_and_raises(tmp_path):
    """D-076: the monitor records max |hidden| during training and fails on a non-finite value."""
    done = sft.train_one(
        _spec(epochs=1),
        make_rows(8),
        make_chat_tokenizer(),
        make_tiny_model(),
        tmp_path / "ok",
        save_minutes=1e6,
        activation_monitor=True,
    )
    assert done["max_abs_hidden"] > 0 and done["monitored_forwards"] >= done["global_step"]

    model = make_tiny_model()

    def poison(_m, _i, output):
        out = output[0] if isinstance(output, tuple) else output
        out[0, 0, 0] = float("nan")
        return output

    model.model.layers[-1].register_forward_hook(poison)
    with pytest.raises(NonFiniteError, match="train.hidden"):
        sft.train_one(
            _spec(epochs=1),
            make_rows(8),
            make_chat_tokenizer(),
            model,
            tmp_path / "bad",
            save_minutes=1e6,
            activation_monitor=True,
        )
    assert not (tmp_path / "bad" / "done.json").exists()


def test_gemma_lora_regex_targets_language_model_only():
    """D-076: Gemma-3's LoRA stays off the SigLIP vision tower and the projector; the probe
    and the training config use the identical regex."""
    import re

    from rbbd import config
    from tests.conftest import REPO_ROOT

    regexes = {
        config.load(REPO_ROOT / "configs" / f)["models"][0]["lora_target_regex"]
        for f in ("tier1_gemma3-4b.yaml", "m3c_probe_gemma3-4b.yaml")
    }
    assert len(regexes) == 1
    rx = regexes.pop()
    for proj in sft.LORA_TARGETS:
        kind = "mlp" if proj in ("gate_proj", "up_proj", "down_proj") else "self_attn"
        assert re.fullmatch(rx, f"model.language_model.layers.7.{kind}.{proj}")
    for name in ("model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
                 "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1",
                 "model.multi_modal_projector.mm_input_projection", "lm_head"):  # fmt: skip
        assert not re.fullmatch(rx, name)


def test_launch_passes_overrides_to_train_one(tmp_path, monkeypatch):
    """Runtime decisions (`--set`, D-076) reach every train-one subprocess."""
    import subprocess

    cfg = _cfg({"regimes": ["qlora"]})
    cfg.overrides = ("train.epochs=1", "train.compute=fp32")
    job = sft.plan_jobs(cfg, tmp_path, _ftdata(tmp_path))[0]
    seen = {}

    class P:
        def __init__(self, cmd, **kw):
            seen["cmd"], seen["env"] = cmd, kw["env"]

    monkeypatch.setattr(subprocess, "Popen", P)
    _, logf = sft._launch(SimpleNamespace(cfg=cfg), job, 1)
    logf.close()
    cmd = seen["cmd"]
    assert cmd[cmd.index("--set") + 1] == "train.epochs=1" and "train.compute=fp32" in cmd
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "1"
