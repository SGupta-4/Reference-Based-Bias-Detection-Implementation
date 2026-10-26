"""Target-only vs attribute-only RR ΔB (M6-T4, the B-030 follow-up of D-085). CPU only,
from cached embeddings.

RR coordinates are relative to the shared anchor sentences, so a sentence's relative
representation r(x) = [cos(e(x), e(a_i))] is comparable across the reference and an
audited checkpoint (§3.3). That lets ΔB be split by which side moved:

- full:        B(r_aud(T), r_aud(P), r_aud(N)) − B(r_ref(T), r_ref(P), r_ref(N))  (= RR ΔB)
- target-only: B(r_aud(T), r_ref(P), r_ref(N)) − B_ref   ("rr_target")
- attr-only:   B(r_ref(T), r_aud(P), r_aud(N)) − B_ref   ("rr_attr")

Rows use the `metrics.delta_b` CSV format (cell "primary", mean pooling, post-norm,
neutral anchors) with methods `rr_target` / `rr_attr`, so `analysis.analyze` joins them
with the benchmarks like any other ΔB method.

The embed index and sentence union are located by content, not by the runner's run keys
(which follow the config hash, D-097): `locate` picks the `embed/*/index.json` with the
most entries for the model and the `sentences/*/union.json` whose hash that embed run
recorded in its `timing.json`. Restore `embed/`, `sentences/` and `embeddings/<slug>`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rbbd.data.groups import BY_NAME
from rbbd.metrics import rr
from rbbd.metrics.delta_b import EmbeddingSet, from_union

METHODS = ("rr_target", "rr_attr")


def _rel(es: EmbeddingSet) -> dict[str, np.ndarray]:
    """Relative representations [n, m] of targets, positives and negatives of `es`."""
    return {k: rr.relative(getattr(es, k), es.anchors)
            for k in ("targets", "positives", "negatives")}  # fmt: skip


def decompose(ref: EmbeddingSet, aud: EmbeddingSet) -> dict[str, dict[str, float]]:
    """{"rr", "rr_target", "rr_attr"} → {group: ΔB}. Embedding sets [*, d] per checkpoint."""
    r0, r1 = _rel(ref), _rel(aud)
    groups = ref.target_groups

    def b(t, p, n):
        return rr.group_means(rr.row_bias_rel(t, p, n), groups)

    base = b(r0["targets"], r0["positives"], r0["negatives"])
    parts = {
        "rr": b(r1["targets"], r1["positives"], r1["negatives"]),
        "rr_target": b(r1["targets"], r0["positives"], r0["negatives"]),
        "rr_attr": b(r0["targets"], r1["positives"], r1["negatives"]),
    }
    return {m: {g: v - base[g] for g, v in vals.items()} for m, vals in parts.items()}


def spectrum_rows(
    ref: Mapping[str, np.ndarray],
    entries: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
    union: Mapping[str, Any],
    spectrum: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """`metrics.delta_b` CSV rows (methods `rr_target`, `rr_attr`) for one spectrum.
    `entries` = (index row with `ckpt`, `alpha`; tensors) per audited checkpoint."""
    ref_set = from_union(ref["mean"], dict(union))
    for row, tensors in entries:
        parts = decompose(ref_set, from_union(tensors["mean"], dict(union)))
        for method in METHODS:
            for group, value in parts[method].items():
                yield {
                    **spectrum, "ckpt": row["ckpt"], "alpha": row["alpha"], "group": group,
                    "topic": BY_NAME[group].topic, "method": method, "cell": "primary",
                    "anchor_source": "neutral", "n_anchors": -1, "anchor_seed": -1,
                    "attr_variant": "base", "target_variant": "base", "pooling": "mean",
                    "layer_site": "post_norm", "B_ref": None, "B_aud": None, "delta_b": value,
                }  # fmt: skip


def locate(root: Path, slug: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(embed index rows of `slug`, the sentence union they were extracted over)."""
    root = Path(root)
    best: tuple[int, Path] | None = None
    for path in sorted(root.glob("embed/*/index.json")):
        n = sum(r["model"] == slug for r in json.loads(path.read_text()))
        if n and (best is None or n > best[0]):
            best = (n, path)
    if best is None:
        raise FileNotFoundError(f"no embed index with entries for {slug} (restore embed/)")
    index = [r for r in json.loads(best[1].read_text()) if r["model"] == slug]
    timing = best[1].with_name("timing.json")
    want = json.loads(timing.read_text())["union_hash"] if timing.exists() else None
    for path in sorted(root.glob("sentences/*/union.json")):
        union = json.loads(path.read_text())
        if want is None or union.get("union_hash") == want:
            return index, union
    raise FileNotFoundError(f"no sentences/*/union.json with hash {want} (restore sentences/)")


def decompose_model(root: Path, slug: str) -> list[dict[str, Any]]:
    """Rows for every spectrum of `slug` found in the restored caches."""
    from rbbd.utils.cache import TensorCache

    root = Path(root)
    index, union = locate(root, slug)
    cache = TensorCache(root)

    def read(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
        path = root / row["path"]
        tensors, _ = cache.read(str(path.parent.relative_to(root)), path.stem)
        return tensors

    refs = [r for r in index if r["ckpt"] == "ref"]
    if not refs:
        raise FileNotFoundError(f"no reference entry for {slug}")
    ref = read(refs[0])
    rows: list[dict[str, Any]] = []
    for model, regime, seed in sorted({(r["model"], r["regime"], r["seed"])
                                       for r in index if r["ckpt"] != "ref"}):  # fmt: skip
        members = [
            r for r in index if (r["model"], r["regime"], r["seed"]) == (model, regime, seed)
        ]
        entries = [(r, read(r)) for r in members]
        rows += list(spectrum_rows(ref, entries, union,
                                   {"model": model, "regime": regime, "seed": seed}))  # fmt: skip
    return rows
