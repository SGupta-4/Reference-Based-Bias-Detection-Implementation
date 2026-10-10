"""Stage graph, manifest-based skip/resume and stage timing (D-030, FLOW.md › Entrypoint).

`run()` walks the requested stages in dependency order. For each stage it derives
a `run_key` from the config hash, the stage name and the upstream run keys, then:

- valid complete manifest  -> log `cache hit` and skip;
- incomplete manifest      -> call the stage with `ctx.resume=True` and its saved `progress`;
- otherwise                -> call the stage fresh.

A manifest with `complete=False` is written *before* the stage body runs, so a
session killed at the 12 h limit leaves a resumable record (B-007). Stage bodies
are plain functions `fn(ctx) -> StageResult`; until their milestone lands they
are stubs that raise `NotImplementedError` naming the milestone.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rbbd.config import Config
from rbbd.utils import manifest as mf
from rbbd.utils.cache import make_key
from rbbd.utils.env import Env
from rbbd.utils.logging import get_logger

log = get_logger(__name__)

STAGES: tuple[str, ...] = (
    "sentences",
    "ftdata",
    "train",
    "embed",
    "deltab",
    "generate",
    "score",
    "analyze",
    "report",
)
DEPENDS: dict[str, tuple[str, ...]] = {
    "sentences": (),
    "ftdata": (),
    "train": ("ftdata",),
    "embed": ("sentences", "train"),
    "deltab": ("embed",),
    "generate": ("train",),
    "score": ("generate",),
    "analyze": ("deltab", "score"),
    "report": ("analyze",),
}
MILESTONE: dict[str, str] = {
    "sentences": "M1",
    "ftdata": "M1/M3",
    "train": "M3",
    "embed": "M2",
    "deltab": "M4",
    "generate": "M5",
    "score": "M5",
    "analyze": "M6",
    "report": "M7",
}


class RunnerError(RuntimeError):
    """Unknown stage, or a dependency without a valid manifest."""


@dataclass
class StageResult:
    """What a stage returns: named outputs as paths relative to the artifact root."""

    outputs: dict[str, str] = field(default_factory=dict)


@dataclass
class StageContext:
    """Everything a stage body may use. `stage_dir` = `<root>/<stage>/<run_key>/`."""

    stage: str
    cfg: Config
    env: Env
    run_key: str
    stage_dir: Path
    upstream: dict[str, mf.Manifest]
    resume: bool
    progress: dict[str, Any]
    only_ckpt: str | None
    _manifest: mf.Manifest
    _manifest_path: Path

    def checkpoint(self, progress: Mapping[str, Any]) -> None:
        """Persist partial progress (e.g. finished shards) into the incomplete manifest."""
        self.progress = dict(progress)
        self._manifest.progress = self.progress
        mf.write(self._manifest_path, self._manifest)


StageFn = Callable[[StageContext], StageResult]


def _stub(stage: str) -> StageFn:
    def fn(ctx: StageContext) -> StageResult:
        raise NotImplementedError(f"stage '{stage}' is implemented in {MILESTONE[stage]} (PLAN §7)")

    return fn


# Implemented stage bodies, imported only when the stage runs so CPU-only stages never
# pull in the GPU stack. Stages not listed here are stubs that name their milestone.
STAGE_IMPLS: dict[str, str] = {
    "sentences": "rbbd.data.sentences:stage",
    "ftdata": "rbbd.data.ft_data:stage",
    "train": "rbbd.finetune.sft:stage",
    "embed": "rbbd.embed.extract:stage",
    "deltab": "rbbd.metrics.delta_b:stage",
    "generate": "rbbd.bench.generate:stage",
    "score": "rbbd.bench.score:stage",
}

# Stages that sweep checkpoints. Under `--only-ckpt` they cover one checkpoint, so their
# manifest is left incomplete (D-081): a later full run resumes it instead of mistaking
# a one-checkpoint result for the whole sweep.
SWEEP_STAGES = frozenset({"embed", "deltab", "generate", "score"})


def _lazy(stage: str) -> StageFn:
    target = STAGE_IMPLS.get(stage)
    if target is None:
        return _stub(stage)

    def fn(ctx: StageContext) -> StageResult:
        import importlib

        module, name = target.split(":")
        return getattr(importlib.import_module(module), name)(ctx)

    return fn


DEFAULT_STAGE_FNS: dict[str, StageFn] = {s: _lazy(s) for s in STAGES}


def topo_order(requested: Iterable[str]) -> list[str]:
    """Requested stages in pipeline order; raises on unknown names."""
    requested = list(requested)
    unknown = sorted(set(requested) - set(STAGES))
    if unknown:
        raise RunnerError(f"unknown stages: {unknown}; known: {list(STAGES)}")
    return [s for s in STAGES if s in requested]


def stage_deps(stage: str, cfg: Config) -> tuple[str, ...]:
    """Upstream stages of `stage` under `cfg`.

    `embed` needs `train` only when α-checkpoints are configured; extracting the
    reference model alone (M2, and the reference half of every later run) depends on
    `sentences` only (D-057).
    """
    if stage == "embed" and list(cfg.get("embed.checkpoints", ["ref"])) == ["ref"]:
        return ("sentences",)
    return DEPENDS[stage]


def stage_run_keys(cfg: Config) -> dict[str, str]:
    """Run key for every stage: content key of (config hash, stage, upstream run keys)."""
    keys: dict[str, str] = {}
    for stage in STAGES:
        keys[stage] = make_key(
            {
                "config_hash": cfg.hash,
                "stage": stage,
                "upstream": {d: keys[d] for d in stage_deps(stage, cfg)},
            }
        )
    return keys


def _input_hashes(upstream: Mapping[str, mf.Manifest]) -> dict[str, str]:
    # One digest per upstream stage over its recorded outputs, so any re-run
    # upstream that changes bytes invalidates this stage too.
    return {name: make_key(m.output_hashes) for name, m in sorted(upstream.items())}


def stage_status(cfg: Config, env: Env) -> list[dict[str, str]]:
    """Per-stage manifest status for `cli status`: valid / incomplete / missing / stale(reason)."""
    keys = stage_run_keys(cfg)
    rows = []
    for stage in STAGES:
        path = mf.manifest_path(env.artifacts_root, stage, keys[stage])
        m = mf.read(path)
        if m is None:
            status = "missing"
        elif not m.complete:
            status = "incomplete"
        else:
            upstream = {
                d: mf.read(mf.manifest_path(env.artifacts_root, d, keys[d]))
                for d in stage_deps(stage, cfg)
            }
            if any(u is None for u in upstream.values()):
                status = "stale (upstream missing)"
            else:
                ok, reason = mf.validate(
                    m,
                    config_hash=cfg.hash,
                    input_hashes=_input_hashes(upstream),
                    artifacts_root=env.artifacts_root,
                )
                status = "valid" if ok else f"stale ({reason})"
        rows.append({"stage": stage, "run_key": keys[stage], "status": status})
    return rows


def run(
    cfg: Config,
    env: Env,
    stages: Iterable[str] | None = None,
    *,
    force: Iterable[str] = (),
    stage_fns: Mapping[str, StageFn] | None = None,
    only_ckpt: str | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Run stages; returns {stage: "cache hit" | "ran" | "resumed" | "partial" | "planned"}."""
    fns = {**DEFAULT_STAGE_FNS, **(stage_fns or {})}
    order = topo_order(stages if stages is not None else STAGES)
    forced = set(topo_order(force))
    keys = stage_run_keys(cfg)
    root = env.artifacts_root
    outcome: dict[str, str] = {}
    sha = mf.git_sha()

    for stage in order:
        upstream: dict[str, mf.Manifest] = {}
        for dep in stage_deps(stage, cfg):
            dep_m = mf.read(mf.manifest_path(root, dep, keys[dep]))
            if dep_m is None or not dep_m.complete:
                if dry_run:
                    continue
                raise RunnerError(
                    f"stage '{stage}' needs a complete '{dep}' manifest; run '{dep}' first"
                )
            upstream[dep] = dep_m
        inputs = _input_hashes(upstream)
        path = mf.manifest_path(root, stage, keys[stage])
        existing = mf.read(path)
        valid, reason = mf.validate(
            existing, config_hash=cfg.hash, input_hashes=inputs, artifacts_root=root
        )

        if valid and stage not in forced:
            log.info("cache hit: stage %s (run_key %s)", stage, keys[stage])
            outcome[stage] = "cache hit"
            continue
        resume = (
            existing is not None
            and not existing.complete
            and existing.config_hash == cfg.hash
            and existing.input_hashes == inputs
            and stage not in forced
        )
        if dry_run:
            log.info(
                "dry run: stage %s would %s (%s)", stage, "resume" if resume else "run", reason
            )
            outcome[stage] = "planned"
            continue

        started = datetime.now(timezone.utc).isoformat()
        current = mf.Manifest(
            stage=stage,
            run_key=keys[stage],
            config_hash=cfg.hash,
            git_sha=sha,
            input_hashes=inputs,
            started=started,
            env_summary=env.summary(),
            complete=False,
            progress=dict(existing.progress) if resume and existing else {},
        )
        mf.write(path, current)
        stage_dir = root / stage / keys[stage]
        stage_dir.mkdir(parents=True, exist_ok=True)
        ctx = StageContext(
            stage=stage,
            cfg=cfg,
            env=env,
            run_key=keys[stage],
            stage_dir=stage_dir,
            upstream=upstream,
            resume=resume,
            progress=dict(current.progress),
            only_ckpt=only_ckpt,
            _manifest=current,
            _manifest_path=path,
        )
        log.info(
            "%s stage %s (run_key %s)", "resuming" if resume else "running", stage, keys[stage]
        )
        t0 = time.time()
        try:
            result = fns[stage](ctx)
        except BaseException as exc:
            current.error = f"{type(exc).__name__}: {exc}"
            current.seconds = round(time.time() - t0, 3)
            mf.write(path, current)
            raise
        current.output_hashes = {rel: mf.hash_path(root / rel) for rel in result.outputs.values()}
        current.finished = datetime.now(timezone.utc).isoformat()
        current.seconds = round(time.time() - t0, 3)
        partial = only_ckpt is not None and stage in SWEEP_STAGES
        current.complete = not partial
        current.error = None
        mf.write(path, current)
        if partial:
            log.info("stage %s covered only %s in %.1f s; manifest left incomplete",
                     stage, only_ckpt, current.seconds)  # fmt: skip
            outcome[stage] = "partial"
            continue
        log.info("stage %s done in %.1f s", stage, current.seconds)
        outcome[stage] = "resumed" if resume else "ran"
    return outcome
