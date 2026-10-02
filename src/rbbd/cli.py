"""Single CLI entrypoint: `python -m rbbd.cli {probe,run,status,sync,vllm-case}`.

Call path: FLOW.md › Entrypoint.

- `probe`     M0: env.json (hardware, access), optional vLLM probe and store round trip.
- `run`       run pipeline stages through `runner.run`.
- `status`    print manifest validity per stage.
- `sync`      upload `<artifacts>/<path>` to the same path in the private store (D-041).
- `vllm-case` internal: one vLLM probe case in a fresh process (called by `probe --vllm`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rbbd import config as config_mod
from rbbd import runner
from rbbd.utils import env as env_mod
from rbbd.utils.logging import get_logger
from rbbd.utils.seeds import seed_all

log = get_logger("rbbd.cli")

DEFAULT_BASE = Path("configs/base.yaml")


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rbbd", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="write env.json (hardware, gated access)")
    p.add_argument("--config", type=Path, default=DEFAULT_BASE)
    p.add_argument("--no-require-gpu", action="store_true", help="do not fail without 2x sm75 GPUs")
    p.add_argument("--skip-access", action="store_true", help="skip HF auth_check calls")
    p.add_argument("--vllm", action="store_true", help="run the vLLM probe cases from the config")
    p.add_argument(
        "--check-store", action="store_true", help="upload/download round trip to the store"
    )

    r = sub.add_parser("run", help="run pipeline stages")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value (YAML-parsed), e.g. train.epochs=1",
    )
    r.add_argument("--stages", help="comma-separated subset (default: all)")
    r.add_argument("--force", help="comma-separated stages to re-run despite a valid manifest")
    r.add_argument("--only-ckpt", help="restrict checkpoint-sweeping stages to one slug, e.g. a050")
    r.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("status", help="manifest status per stage")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value (YAML-parsed), e.g. train.epochs=1",
    )

    y = sub.add_parser("sync", help="upload an artifact subfolder to the private store")
    y.add_argument("--config", type=Path, default=DEFAULT_BASE)
    y.add_argument(
        "--path", required=True, help="path relative to the artifact root, e.g. env/<session>"
    )

    rs = sub.add_parser("restore", help="download an artifact subfolder from the private store")
    rs.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    rs.add_argument(
        "--path", required=True, help="path relative to the artifact root, e.g. train/<slug>"
    )

    c = sub.add_parser("compare-embeddings", help="compare two configs' cached ref embeddings")
    c.add_argument("--a", type=Path, required=True, help="config whose embeddings are tested")
    c.add_argument("--b", type=Path, required=True, help="config used as the reference")

    t = sub.add_parser("train-one", help="train one endpoint (launched by the train stage)")
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--model", required=True, help="model slug from the config")
    t.add_argument("--regime", required=True, choices=["lora", "qlora", "full"])
    t.add_argument("--split", required=True, choices=["unharmful", "harmful"])
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    v = sub.add_parser("vllm-case", help=argparse.SUPPRESS)
    v.add_argument("--json", required=True)
    v.add_argument("--out", type=Path, required=True)
    return parser


def _open_store(cfg: config_mod.Config):
    from rbbd.utils.store import open_store

    return open_store(cfg.get("store.repo_id"), cfg.get("store.repo_type", "auto"))


def _cmd_probe(args: argparse.Namespace) -> int:
    cfg = config_mod.load(args.config)
    env = env_mod.detect(cfg.get("paths.artifacts"))
    session = env_mod.session_id()
    out_dir = env.artifacts_root / "env" / session
    repos = [] if args.skip_access else cfg.get("probe.access_repos", [])
    try:
        info = env_mod.probe(out_dir, access_repos=repos, require_gpu=not args.no_require_gpu)
    except env_mod.HardwareError as exc:
        log.error("hardware requirements failed: %s (env.json written to %s)", exc, out_dir)
        return 2
    if args.vllm:
        results = env_mod.probe_vllm(cfg.get("probe.vllm_cases", []), out_dir)
        info["vllm"] = [
            {"name": r["case"]["name"], "ok": r["ok"], "refused": r.get("refused")} for r in results
        ]
    if args.check_store:
        from rbbd.utils.store import roundtrip

        info["store"] = roundtrip(_open_store(cfg), env.artifacts_root, session)
    env_mod.write_json(out_dir / "env.json", info)
    print(
        json.dumps(
            {
                "env_json": str(out_dir / "env.json"),
                "requirements_ok": info["requirements"]["ok"],
                "access_all_ok": info["access_all_ok"],
                "store": info.get("store", {}).get("ok"),
                "vllm": info.get("vllm"),
            },
            indent=2,
        )
    )
    failed = info["access_all_ok"] is False or (args.check_store and not info["store"]["ok"])
    return 1 if failed else 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = config_mod.load(args.config, args.set)
    env = env_mod.detect(cfg.get("paths.artifacts"))
    env_mod.session_id()
    seed_all(int(cfg["seed"]))
    log.info("config %s hash %s; artifacts %s", args.config, cfg.hash, env.artifacts_root)
    outcome = runner.run(
        cfg,
        env,
        _csv(args.stages),
        force=_csv(args.force) or (),
        only_ckpt=args.only_ckpt,
        dry_run=args.dry_run,
    )
    print(json.dumps({"config_hash": cfg.hash, "stages": outcome}, indent=2))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = config_mod.load(args.config, args.set)
    env = env_mod.detect(cfg.get("paths.artifacts"))
    for row in runner.stage_status(cfg, env):
        print(f"{row['stage']:<10} {row['run_key']}  {row['status']}")
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    cfg = config_mod.load(args.config)
    env = env_mod.detect(cfg.get("paths.artifacts"))
    local = env.artifacts_root / args.path
    commit = _open_store(cfg).upload_dir(local, args.path, f"sync {args.path}")
    print(json.dumps({"path": args.path, "commit": commit}))
    return 0


def _cmd_restore(args: argparse.Namespace) -> int:
    cfg = config_mod.load(args.config)
    env = env_mod.detect(cfg.get("paths.artifacts"))
    local = _open_store(cfg).download_dir(args.path, env.artifacts_root)
    n = sum(1 for p in local.rglob("*") if p.is_file()) if local.exists() else 0
    print(json.dumps({"path": args.path, "files": n}))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from rbbd.embed.extract import cached_ref_entry, compare_entries

    cfg_a, cfg_b = config_mod.load(args.a), config_mod.load(args.b)
    env = env_mod.detect(cfg_a.get("paths.artifacts"))
    slug_a, slug_b = cfg_a["models"][0]["slug"], cfg_b["models"][0]["slug"]
    a, union_a = cached_ref_entry(cfg_a, env, slug_a)
    b, union_b = cached_ref_entry(cfg_b, env, slug_b)
    if union_a["union_hash"] != union_b["union_hash"]:
        raise SystemExit("the two configs embedded different sentence unions")
    result = {"a": str(args.a), "b": str(args.b), **compare_entries(a, b, union_b)}
    out = env.artifacts_root / "env" / env_mod.session_id() / f"compare_{cfg_a['run_name']}.json"
    env_mod.write_json(out, result)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_train_one(args: argparse.Namespace) -> int:
    # One endpoint, in its own process with one visible GPU (D-003). The job is
    # re-derived from the config and the ftdata manifest, so the parent passes names only.
    from rbbd.finetune import sft
    from rbbd.utils import manifest as mf

    cfg = config_mod.load(args.config, args.set)
    env = env_mod.detect(cfg.get("paths.artifacts"))
    seed_all(args.seed)
    keys = runner.stage_run_keys(cfg)
    ft = mf.read(mf.manifest_path(env.artifacts_root, "ftdata", keys["ftdata"]))
    if ft is None or not ft.complete:
        raise SystemExit("train-one needs a complete ftdata manifest; run the ftdata stage first")
    jobs = sft.plan_jobs(cfg, env.artifacts_root, list(ft.output_hashes))
    match = [j for j in jobs if (j.spec.slug, j.regime, j.split, j.seed)
             == (args.model, args.regime, args.split, args.seed)]  # fmt: skip
    if len(match) != 1:
        raise SystemExit(
            f"no unique job for {args.model}/{args.regime}/{args.split}/seed{args.seed}"
        )
    try:
        done = sft.run_job(cfg, match[0], single_attempt=True)
    except sft.LayoutOOM as exc:
        # The parent relaunches a fresh process for the next layout (B-021).
        print(f"layout OOM: {exc}")
        return sft.OOM_EXIT
    print(json.dumps(done, indent=2))
    return 0


def _cmd_vllm_case(args: argparse.Namespace) -> int:
    case = json.loads(args.json)
    result = env_mod.vllm_case(case, args.out.parent)
    env_mod.write_json(args.out, result)
    return 0 if result["ok"] else 3


def main(argv: list[str] | None = None) -> int:
    """Parse `argv` and dispatch; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    handlers = {
        "probe": _cmd_probe,
        "run": _cmd_run,
        "status": _cmd_status,
        "sync": _cmd_sync,
        "compare-embeddings": _cmd_compare,
        "train-one": _cmd_train_one,
        "restore": _cmd_restore,
        "vllm-case": _cmd_vllm_case,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
