"""Stage manifests and runner skip/resume (D-030)."""

import logging

import pytest

from rbbd import config, runner
from rbbd.utils import env as env_mod
from rbbd.utils import manifest as mf


@pytest.fixture
def cfg(config_dir):
    return config.load(config_dir("run_name: base\nseed: 0\n", "run_name: t\n"))


def _writer(calls):
    """Stage body that writes one output file and counts invocations."""

    def fn(ctx):
        calls.append(ctx.resume)
        out = ctx.stage_dir / "out.txt"
        out.write_text("payload")
        return runner.StageResult(outputs={"out": str(out.relative_to(ctx.env.artifacts_root))})

    return fn


def test_valid_manifest_skips_stage(cfg, artifacts, caplog):
    """A complete manifest with a matching config hash makes the runner skip the stage."""
    env = env_mod.detect()
    calls = []
    fns = {"sentences": _writer(calls)}
    assert runner.run(cfg, env, ["sentences"], stage_fns=fns) == {"sentences": "ran"}
    with caplog.at_level(logging.INFO, logger="rbbd"):
        assert runner.run(cfg, env, ["sentences"], stage_fns=fns) == {"sentences": "cache hit"}
    assert calls == [False]
    assert "cache hit: stage sentences" in caplog.text

    # An output edited after the fact invalidates the manifest; --force re-runs regardless.
    (next(artifacts.rglob("out.txt"))).write_text("tampered")
    assert runner.run(cfg, env, ["sentences"], stage_fns=fns) == {"sentences": "ran"}
    assert runner.run(cfg, env, ["sentences"], stage_fns=fns, force=["sentences"]) == {
        "sentences": "ran"
    }
    assert len(calls) == 3


def test_incomplete_manifest_resumes(cfg, artifacts):
    """complete=false triggers resume with the saved progress, not skip."""
    env = env_mod.detect()
    seen = []

    def flaky(ctx):
        seen.append((ctx.resume, dict(ctx.progress)))
        if not ctx.resume:
            ctx.checkpoint({"shards_done": 3})
            raise RuntimeError("session killed")
        out = ctx.stage_dir / "done.txt"
        out.write_text("ok")
        return runner.StageResult(outputs={"done": str(out.relative_to(artifacts))})

    with pytest.raises(RuntimeError, match="session killed"):
        runner.run(cfg, env, ["sentences"], stage_fns={"sentences": flaky})
    keys = runner.stage_run_keys(cfg)
    partial = mf.read(mf.manifest_path(artifacts, "sentences", keys["sentences"]))
    assert partial.complete is False and partial.progress == {"shards_done": 3}
    assert "session killed" in partial.error

    assert runner.run(cfg, env, ["sentences"], stage_fns={"sentences": flaky}) == {
        "sentences": "resumed"
    }
    assert seen == [(False, {}), (True, {"shards_done": 3})]
    final = mf.read(mf.manifest_path(artifacts, "sentences", keys["sentences"]))
    assert final.complete and final.error is None and final.output_hashes


def test_missing_upstream_raises(cfg, artifacts):
    """A stage whose dependency has no complete manifest refuses to run."""
    with pytest.raises(runner.RunnerError, match="needs a complete 'ftdata'"):
        runner.run(cfg, env_mod.detect(), ["train"])


def test_stub_names_its_milestone(cfg, artifacts):
    """Unimplemented stages fail loudly with their milestone and leave an incomplete manifest."""
    done = [s for s in runner.STAGES if s not in ("analyze", "report")]
    fns = {s: _writer([]) for s in done} | {"analyze": runner.DEFAULT_STAGE_FNS["analyze"]}
    with pytest.raises(NotImplementedError, match="implemented in M6"):
        runner.run(cfg, env_mod.detect(), [*done, "analyze"], stage_fns=fns)
    rows = {r["stage"]: r["status"] for r in runner.stage_status(cfg, env_mod.detect())}
    assert rows["score"] == "valid" and rows["analyze"] == "incomplete"
    assert rows["report"] == "missing"


def test_sentences_stage_runs_and_caches(cfg, artifacts, caplog):
    """The real `sentences` stage writes union.json; an identical second run is a cache hit."""
    import json
    import logging

    env = env_mod.detect()
    assert runner.run(cfg, env, ["sentences"]) == {"sentences": "ran"}
    union = json.loads(next(artifacts.rglob("union.json")).read_text())
    assert union["stats"]["anchor_group_sizes"] == {"min": 41, "max": 42}
    assert len(union["index"]["targets/base/Women"]) == 50
    with caplog.at_level(logging.INFO, logger="rbbd"):
        assert runner.run(cfg, env, ["sentences"]) == {"sentences": "cache hit"}
    assert "cache hit: stage sentences" in caplog.text


def test_upstream_change_invalidates_downstream(cfg, artifacts):
    """Re-running an upstream stage with different bytes makes the downstream manifest stale."""
    env = env_mod.detect()
    content = {"v": "one"}

    def upstream(ctx):
        out = ctx.stage_dir / "rows.txt"
        out.write_text(content["v"])
        return runner.StageResult(outputs={"rows": str(out.relative_to(artifacts))})

    down_calls = []
    fns = {"ftdata": upstream, "train": _writer(down_calls)}
    runner.run(cfg, env, ["ftdata", "train"], stage_fns=fns)
    assert runner.run(cfg, env, ["train"], stage_fns=fns) == {"train": "cache hit"}
    content["v"] = "two"
    runner.run(cfg, env, ["ftdata"], stage_fns=fns, force=["ftdata"])
    assert runner.run(cfg, env, ["train"], stage_fns=fns) == {"train": "ran"}
