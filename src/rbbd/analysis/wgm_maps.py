"""WGM scores under an alternative topic map (D-095 sensitivity: the Qwen2.5-7B map vs the
human map). CPU only, from the per-sample WildGuard labels the `score` stage wrote.

Reads `scores/wgm/<slug>/<spectrum>/<ckpt>/<score_key>.jsonl` (prompt SHA-256, sample,
harmful; no text) for one model and aggregates them per topic exactly as the `score`
stage does (`bench.score.item_values` / `units_of` / `aggregate`, paired bootstrap of Δ
vs ref over the topic's prompts), so the two maps differ only in `topic_of`. Output rows
use `bench.score.CSV_COLUMNS` and go to `results/<run>/bench_scores_wgm_qwen.csv`.
Restore `scores/wgm/<slug>` first.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rbbd.utils.logging import get_logger

log = get_logger(__name__)


def label_files(root: Path, slug: str) -> dict[tuple[str, str], Path]:
    """{(spectrum dir, ckpt): label file}; the newest file wins if a checkpoint has several
    (an older `schema_version.score`), and that is logged."""
    out: dict[tuple[str, str], Path] = {}
    base = Path(root) / "scores" / "wgm" / slug
    for path in sorted(base.glob("*/*/*.jsonl")):
        key = (path.parent.parent.name, path.parent.name)
        if key in out:
            newer = max(out[key], path, key=lambda p: p.stat().st_mtime)
            log.warning("several WGM label files for %s/%s; using %s", *key, newer.name)
            path = newer
        out[key] = path
    if not out:
        raise FileNotFoundError(f"no WGM label files under {base} (restore scores/wgm/{slug})")
    return out


def rescore(
    root: Path, slug: str, topic_of: Mapping[str, str], n_boot: int = 1000, seed: int = 0
) -> list[dict[str, Any]]:
    """`bench_scores.csv` rows for WGM under `topic_of` (prompt sha → topic or "none")."""
    from rbbd.bench import score as sc
    from rbbd.models.spectrum import alpha_of

    files = label_files(root, slug)
    labels = {k: sc._read_labels(p) for k, p in files.items()}
    if ("base", "ref") not in labels:
        raise FileNotFoundError(f"no reference WGM labels for {slug}")
    ids = {x["id"] for ls in labels.values() for x in ls}
    units = sc.units_of("wgm", {i: {} for i in ids}, topic_of)
    rows: list[dict[str, Any]] = []
    for spectrum in sorted({s for s, _ in labels if s != "base"}):
        regime, seed_ = spectrum.rsplit("-s", 1)
        keys = [("base", "ref")] + sorted(k for k in labels if k[0] == spectrum)
        per_ckpt = {(slug, s, c): sc.item_values(labels[(s, c)], "harmful") for s, c in keys}
        n_samp = {(slug, s, c): len(labels[(s, c)]) for s, c in keys}
        meta = {
            (slug, s, c): {"model": slug, "regime": regime, "seed": int(seed_), "ckpt": c,
                           "alpha": alpha_of(c)}
            for s, c in keys
        }  # fmt: skip
        rows += sc.aggregate(per_ckpt, n_samp, "wgm", units, meta, n_boot, seed)
    return rows
