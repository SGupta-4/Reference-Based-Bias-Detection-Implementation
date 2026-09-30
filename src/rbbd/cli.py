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
    r.add_argument("--stages", help="comma-separated subset (default: all)")
    r.add_argument("--force", help="comma-separated stages to re-run despite a valid manifest")
    r.add_argument("--only-ckpt", help="restrict checkpoint-sweeping stages to one slug, e.g. a050")
    r.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("status", help="manifest status per stage")
    s.add_argument("--config", type=Path, required=True)

    y = sub.add_parser("sync", help="upload an artifact subfolder to the private store")
    y.add_argument("--config", type=Path, default=DEFAULT_BASE)
    y.add_argument(
        "--path", required=True, help="path relative to the artifact root, e.g. env/<session>"
    )

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
    cfg = config_mod.load(args.config)
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
    cfg = config_mod.load(args.config)
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
        "vllm-case": _cmd_vllm_case,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
