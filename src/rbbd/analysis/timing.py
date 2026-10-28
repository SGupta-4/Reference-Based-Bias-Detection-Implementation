"""Wall-clock per stage for the R6 cost table (T3 analogue; D-102). CPU only.

Reads what the GPU stages already logged, for one config's model:
- embed: `stats.seconds` of every embedding sidecar `embeddings/<slug>/.../<embed_key>.json`
  the embed index lists (the sweep that computed it, not a later cache hit; model load
  excluded). Restore with `cli restore --path embeddings/<slug> --include "*.json"`.
- deltab: `deltab/<run_key>/timing.json` (`spectra` seconds of this model).
- generate: `generate/<run_key>/timing.json` (seconds per checkpoint × benchmark, summed
  over every process that worked on it, D-086 / DC-14).
- score: `manifests/score/<run_key>.json` `seconds` of the last score run (WildGuard over
  every checkpoint; DT/ToxiGen labels reused by score_key, D-096), so it is a lower bound
  for the classifiers' share.

Each stage directory is taken at this config's run key when it holds the model, otherwise
the one with the most entries for it (config hashes changed between milestones, D-097).
Output: `results/<run>/timing.json` (numbers only, DC-18), read by `analysis.report`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rbbd.utils.logging import get_logger

log = get_logger(__name__)


def _load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _pick(root: Path, stage: str, run_key: str | None, count: Callable[[Any], int],
          name: str = "timing.json") -> tuple[Path | None, Any]:  # fmt: skip
    """(path, content) of `<stage>/<key>/<name>`: this run key if it covers the model,
    else the file with the largest `count`."""
    best: tuple[int, Path | None, Any] = (0, None, None)
    paths = sorted(Path(root).glob(f"{stage}/*/{name}"))
    if run_key is not None:
        own = Path(root) / stage / run_key / name
        paths = [own, *[p for p in paths if p != own]] if own.exists() else paths
    for path in paths:
        content = _load(path)
        n = count(content) if content is not None else 0
        if n and run_key is not None and path.parent.name == run_key:
            return path, content
        if n > best[0]:
            best = (n, path, content)
    return best[1], best[2]


def collect(root: Path, slug: str, run_keys: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Timing summary for model `slug` under the artifact root (see the module docstring).

    Returns {"model", "embed": {"seconds_total", "n"}, "deltab": {...}, "generate":
    {bench: {...}}, "score": {...}, "sources": {part: path}}; `n` = checkpoints covered,
    so `seconds_total / n` is minutes-per-checkpoint material. Missing parts are absent."""
    root = Path(root)
    keys = dict(run_keys or {})
    out: dict[str, Any] = {"model": slug, "sources": {}}

    def rows_of(content):
        return [r for r in content if r.get("model") == slug] if isinstance(content, list) else []

    path, index = _pick(root, "embed", keys.get("embed"), lambda c: len(rows_of(c)), "index.json")
    if path is not None:
        secs, n = 0.0, 0
        for r in rows_of(index):
            side = _load((root / r["path"]).with_suffix(".json"))
            stats = (side or {}).get("stats") or {}
            if "seconds" in stats:
                secs, n = secs + float(stats["seconds"]), n + 1
        if n:
            out["embed"] = {"seconds_total": round(secs, 1), "n": n}
            out["sources"]["embed"] = str(path.relative_to(root))
        else:
            log.warning("no embedding sidecars with stats for %s (restore with --include)", slug)

    def spectra_of(content):
        sp = (content or {}).get("spectra", {}) if isinstance(content, dict) else {}
        return {k: v for k, v in sp.items() if k.startswith(f"{slug}/")}

    path, tim = _pick(root, "deltab", keys.get("deltab"), lambda c: len(spectra_of(c)))
    if path is not None:
        sp = spectra_of(tim)
        audited = sum(1 for r in rows_of(index or []) if r.get("ckpt") != "ref") or 7 * len(sp)
        out["deltab"] = {"seconds_total": round(sum(sp.values()), 1), "n": audited}
        out["sources"]["deltab"] = str(path.relative_to(root))

    path, gen = _pick(root, "generate", keys.get("generate"), lambda c: len(rows_of(c)))
    if path is not None:
        per: dict[str, dict[str, Any]] = {}
        for r in rows_of(gen):
            b = per.setdefault(r["bench"], {"seconds_total": 0.0, "n": 0})
            b["seconds_total"] = round(b["seconds_total"] + float(r.get("seconds") or 0.0), 1)
            b["n"] += 1
        out["generate"] = per
        out["sources"]["generate"] = str(path.relative_to(root))
        n_ckpts = len({(r["spectrum"], r["ckpt"]) for r in rows_of(gen)})
        man = _load(root / "manifests" / "score" / f"{keys.get('score')}.json") if keys else None
        if man and man.get("seconds"):
            out["score"] = {"seconds_total": float(man["seconds"]), "n": n_ckpts,
                            "note": "last score run"}  # fmt: skip
            out["sources"]["score"] = f"manifests/score/{keys['score']}.json"
    return out
