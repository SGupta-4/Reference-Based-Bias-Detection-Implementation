"""Representational Bias Shift ΔB = B_aud - B_ref per target group (§3.4, Eq. 7).

`EmbeddingSet` holds one checkpoint's embeddings for one ablation cell (pooling,
layer site, anchor source/size, attribute and target variant), all from the same
cache entry. `delta_b_all` returns every method's per-group value for one
(reference, audited) pair; the `deltab` stage (`stage`, M4) turns those into CSV rows
for every cell of `cells` (FLOW.md › Stage: deltab).
A negative ΔB means the group moved towards the negative attributes (increased bias).
"""

from __future__ import annotations

import csv
import gzip
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rbbd.metrics.cka import cka_drift_by_group
from rbbd.metrics.procrustes import bias_aligned_by_group, fit_orthogonal
from rbbd.metrics.rr import bias_rel_by_group
from rbbd.metrics.seat import bias_seat_by_group

METHODS = ("rr", "seat", "procrustes", "cka")


@dataclass(frozen=True)
class EmbeddingSet:
    """One checkpoint's embeddings: targets [n, d] with group labels [n], P [p, d], N [q, d],
    anchors [m, d]. All rows must share d."""

    targets: np.ndarray
    target_groups: tuple[str, ...]
    positives: np.ndarray
    negatives: np.ndarray
    anchors: np.ndarray

    def __post_init__(self) -> None:
        dims = {a.shape[1] for a in (self.targets, self.positives, self.negatives, self.anchors)}
        if len(dims) != 1:
            raise ValueError(f"embedding widths differ: {dims}")
        if len(self.target_groups) != self.targets.shape[0]:
            raise ValueError("target_groups must align with target rows")


def group_bias(es: EmbeddingSet, method: str) -> dict[str, float]:
    """B per group for `method` in {"rr", "seat"} (Eq. 3 or Eq. 6)."""
    if method == "rr":
        return bias_rel_by_group(
            es.targets, es.target_groups, es.positives, es.negatives, es.anchors
        )
    if method == "seat":
        return bias_seat_by_group(es.targets, es.target_groups, es.positives, es.negatives)
    raise ValueError(f"group_bias supports rr and seat, not {method!r}")


def delta(b_aud: dict[str, float], b_ref: dict[str, float]) -> dict[str, float]:
    """Eq. 7 per group: B_aud - B_ref (same group keys required)."""
    if b_aud.keys() != b_ref.keys():
        raise ValueError("reference and audited groups differ")
    return {g: b_aud[g] - b_ref[g] for g in b_ref}


def delta_b_all(
    ref: EmbeddingSet, aud: EmbeddingSet, rotation: np.ndarray | None = None
) -> dict[str, dict[str, float]]:
    """Per-group value of every method for one (reference, audited) pair.

    rr, seat: ΔB (Eq. 7). procrustes: B of rotated audited targets vs reference P/N
    minus B_ref (SEAT). cka: 1 - CKA drift (undirected, not a difference).
    `rotation` [d, d] is a precomputed `procrustes.fit_orthogonal(aud.anchors, ref.anchors)`
    (the deltab stage reuses one fit across cells that share the anchors); None fits here.
    """
    if ref.target_groups != aud.target_groups:
        raise ValueError("reference and audited target rows must be aligned")
    seat_ref = group_bias(ref, "seat")
    if rotation is None:
        aligned = bias_aligned_by_group(
            aud.targets, aud.target_groups, aud.anchors, ref.anchors, ref.positives, ref.negatives
        )
    else:
        aligned = bias_seat_by_group(
            aud.targets @ rotation, aud.target_groups, ref.positives, ref.negatives
        )
    return {
        "rr": delta(group_bias(aud, "rr"), group_bias(ref, "rr")),
        "seat": delta(group_bias(aud, "seat"), seat_ref),
        "procrustes": delta(aligned, seat_ref),
        "cka": cka_drift_by_group(ref.targets, aud.targets, ref.target_groups),
    }


def from_union(
    emb: np.ndarray,
    union: dict,
    *,
    target_variant: str = "base",
    attr_variant: str = "base",
    anchor_source: str = "neutral",
    anchor_subset: list[int] | None = None,
) -> EmbeddingSet:
    """Assemble an EmbeddingSet from one cached tensor [N, d] and the union index (D-020).

    `union` is the `sentences` stage payload; its index maps "targets/<variant>/<group>",
    "positive/<variant>", "negative/<variant>" and "anchors/<source>" to row lists.
    `anchor_subset` picks positions within the anchor pool (`data.anchors.subset_rows`).
    """
    index = union["index"]
    prefix = f"targets/{target_variant}/"
    groups = [k[len(prefix) :] for k in index if k.startswith(prefix)]
    rows = [i for g in groups for i in index[prefix + g]]
    labels = tuple(g for g in groups for _ in index[prefix + g])
    return EmbeddingSet(
        targets=emb[rows].astype(np.float64),
        target_groups=labels,
        positives=emb[index[f"positive/{attr_variant}"]].astype(np.float64),
        negatives=emb[index[f"negative/{attr_variant}"]].astype(np.float64),
        anchors=emb[_pick(index[f"anchors/{anchor_source}"], anchor_subset)].astype(np.float64),
    )


def _pick(rows: list[int], subset: list[int] | None) -> list[int]:
    return rows if subset is None else [rows[i] for i in subset]


# --------------------------------------------------------------------------- deltab stage

CSV_COLUMNS = (
    "model", "regime", "seed", "ckpt", "alpha", "group", "topic", "method", "cell",
    "anchor_source", "n_anchors", "anchor_seed", "attr_variant", "target_variant",
    "pooling", "layer_site", "B_ref", "B_aud", "delta_b",
)  # fmt: skip


@dataclass(frozen=True)
class Cell:
    """One ablation cell: which cached tensor, sentence variants and anchors to score.

    The primary cell is (mean, post_norm, neutral anchors, all 1,000, base, base). Every
    other cell changes exactly one factor (PLAN M6-T4: T5a/b, T6, F4, D-017).
    `n_anchors`/`anchor_seed` = -1 means the whole pool, no subsampling.
    """

    name: str
    pooling: str = "mean"
    layer_site: str = "post_norm"
    anchor_source: str = "neutral"
    n_anchors: int = -1
    anchor_seed: int = -1
    attr_variant: str = "base"
    target_variant: str = "base"

    @property
    def tensor(self) -> str:
        """Name of the tensor in the cache entry ("mean", "pre_norm/mean", ...)."""
        if self.layer_site == "post_norm":
            return self.pooling
        return f"{self.layer_site}/{self.pooling}"

    @property
    def methods(self) -> tuple[str, ...]:
        """Anchor-pool cells score RR only (F4 is an RR ablation; SEAT and CKA do not use
        anchors, and Procrustes is fitted on the 1,000 neutral anchors only, D-028)."""
        primary_anchors = self.anchor_source == "neutral" and self.n_anchors == -1
        return METHODS if primary_anchors else ("rr",)


def cells(union: Mapping[str, Any], tensors: Sequence[str]) -> list[Cell]:
    """Primary cell plus one-factor-at-a-time variations available in this union/cache."""
    from rbbd.data.anchors import ANCHOR_SEEDS, ANCHOR_SIZES

    index = union["index"]
    out = [Cell("primary")]
    poolings = [t for t in tensors if "/" not in t]
    out += [Cell(f"pooling={p}", pooling=p) for p in poolings if p != "mean"]
    out += [Cell(f"layer_site={t.split('/')[0]}", layer_site=t.split("/")[0])
            for t in tensors if t.endswith("/mean")]  # fmt: skip
    attrs = sorted(k.split("/", 1)[1] for k in index if k.startswith("positive/"))
    out += [Cell(f"attr={v}", attr_variant=v) for v in attrs if v != "base"]
    targets = sorted({k.split("/")[1] for k in index if k.startswith("targets/")})
    out += [Cell(f"target={v}", target_variant=v) for v in targets if v != "base"]
    for key in sorted(k for k in index if k.startswith("anchors/")):
        src, n_pool = key.split("/", 1)[1], len(index[key])
        if src != "neutral":
            out.append(Cell(f"anchors={src}/all", anchor_source=src))
        for size in (s for s in ANCHOR_SIZES if s < n_pool):
            out += [Cell(f"anchors={src}/{size}/s{seed}", anchor_source=src, n_anchors=size,
                         anchor_seed=seed) for seed in ANCHOR_SEEDS]  # fmt: skip
    return out


def _embedding_set(emb: np.ndarray, union: Mapping[str, Any], cell: Cell) -> EmbeddingSet:
    from rbbd.data.anchors import subset_rows

    subset = None
    if cell.n_anchors != -1:
        n_pool = len(union["index"][f"anchors/{cell.anchor_source}"])
        subset = subset_rows(n_pool, cell.n_anchors, cell.anchor_seed)
    return from_union(emb, dict(union), target_variant=cell.target_variant,
                      attr_variant=cell.attr_variant, anchor_source=cell.anchor_source,
                      anchor_subset=subset)  # fmt: skip


def spectrum_rows(
    ref: Mapping[str, np.ndarray],
    entries: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
    union: Mapping[str, Any],
    spectrum: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """CSV rows for one spectrum: every (checkpoint, cell, method, group).

    `ref` = the reference entry's tensors; `entries` = (index row, tensors) per checkpoint,
    the reference itself included (its ΔB is 0 by construction: DC-03, DC-11). One
    Procrustes rotation is fitted per (checkpoint, tensor) on the neutral anchors and
    reused by every cell of that tensor; the reference uses the identity.
    """
    from rbbd.data.groups import BY_NAME

    all_cells = cells(union, sorted(ref))
    for row, aud_tensors in entries:
        rotations: dict[str, np.ndarray] = {}
        for cell in all_cells:
            if cell.tensor not in ref:
                continue
            ref_set = _embedding_set(ref[cell.tensor], union, cell)
            aud_set = _embedding_set(aud_tensors[cell.tensor], union, cell)
            if cell.methods == ("rr",):
                values = {"rr": delta(group_bias(aud_set, "rr"), group_bias(ref_set, "rr"))}
                b_ref, b_aud = {"rr": group_bias(ref_set, "rr")}, {"rr": group_bias(aud_set, "rr")}
            else:
                if cell.tensor not in rotations:
                    rotations[cell.tensor] = (
                        np.eye(ref_set.anchors.shape[1]) if row["ckpt"] == "ref"
                        else fit_orthogonal(aud_set.anchors, ref_set.anchors)
                    )  # fmt: skip
                values = delta_b_all(ref_set, aud_set, rotation=rotations[cell.tensor])
                seat_ref = group_bias(ref_set, "seat")
                b_ref = {"rr": group_bias(ref_set, "rr"), "seat": seat_ref,
                         "procrustes": seat_ref, "cka": {}}  # fmt: skip
                b_aud = {
                    "rr": {g: b_ref["rr"][g] + v for g, v in values["rr"].items()},
                    "seat": {g: seat_ref[g] + v for g, v in values["seat"].items()},
                    "procrustes": {g: seat_ref[g] + v for g, v in values["procrustes"].items()},
                    "cka": {},
                }
            cell_fields = {k: v for k, v in asdict(cell).items() if k != "name"}
            for method, per_group in values.items():
                for group, value in per_group.items():
                    yield {
                        **spectrum, "ckpt": row["ckpt"], "alpha": row["alpha"],
                        "group": group, "topic": BY_NAME[group].topic, "method": method,
                        "cell": cell.name, **cell_fields,
                        "B_ref": b_ref[method].get(group), "B_aud": b_aud[method].get(group),
                        "delta_b": value,
                    }  # fmt: skip


def sanity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """M4-T5: per spectrum, mean RR ΔB over groups at a100 (u) and a000 (h), primary cell.

    App. E.5 reports means of 0.291 (u) vs −0.051 (h) for Llama: the harmful endpoint
    should sit lower (negative ΔB = towards negative attributes). A signal, not a gate.
    """
    out: dict[str, Any] = {}
    for r in rows:
        if r["cell"] != "primary" or r["method"] != "rr" or r["ckpt"] not in ("a100", "a000"):
            continue
        key = f"{r['model']}/{r['regime']}-s{r['seed']}"
        out.setdefault(key, {"a100": [], "a000": []})[r["ckpt"]].append(r["delta_b"])
    report = {}
    for key, v in out.items():
        u, h = float(np.mean(v["a100"])), float(np.mean(v["a000"]))
        report[key] = {"mean_delta_b_u": u, "mean_delta_b_h": h, "h_below_u": h < u,
                       "groups": len(v["a100"])}  # fmt: skip
    return report


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    path.parent.mkdir(parents=True, exist_ok=True)
    with opener(path, "wt", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r[k] is None else r[k]) for k in CSV_COLUMNS})


def results_dir(cfg: Any, artifacts_root: Path) -> Path:
    """`results/<run_name>/` in the repository checkout (aggregates only, DC-18), or under
    the artifact root when the package is not running from a checkout."""
    configured = cfg.get("paths.results")
    if configured:
        return Path(configured) / cfg["run_name"]
    repo = Path(__file__).resolve().parents[3]
    base = repo / "results" if (repo / "pyproject.toml").exists() else artifacts_root / "results"
    return base / cfg["run_name"]


def stage(ctx: Any) -> Any:
    """`deltab` stage: ΔB for every (spectrum, checkpoint, cell, method, group) (§3.4, M4).

    Reads the `embed` stage's `index.json` and the sentence union, groups the entries by
    spectrum (each with the model's shared reference entry), and writes:
    - `deltab/<run_key>/delta_b.csv.gz`: every row (columns `CSV_COLUMNS`);
    - `deltab/<run_key>/sanity.json` (M4-T5) and `timing.json` (seconds, DC-12);
    - copies for git: `results/<run>/delta_b.csv` (primary cell only),
      `results/<run>/delta_b_cells.csv.gz` (all cells) and `results/<run>/sanity.json`.
    Embeddings and ΔB values are aggregates over sentences, never text (DC-18).
    """
    from rbbd import runner
    from rbbd.runner import StageResult
    from rbbd.utils import manifest as mf
    from rbbd.utils.cache import TensorCache

    t0 = time.time()
    root = Path(ctx.env.artifacts_root)
    embed = ctx.upstream["embed"]
    index = json.loads((root / next(p for p in embed.output_hashes
                                    if p.endswith("index.json"))).read_text())  # fmt: skip
    keys = runner.stage_run_keys(ctx.cfg)
    sen = mf.read(mf.manifest_path(root, "sentences", keys["sentences"]))
    if sen is None or not sen.complete:
        raise FileNotFoundError("no complete sentences manifest for this config")
    union = json.loads((root / next(p for p in sen.output_hashes
                                    if p.endswith("union.json"))).read_text())  # fmt: skip
    cache = TensorCache(root)

    def read(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
        path = root / row["path"]
        tensors, _ = cache.read(str(path.parent.relative_to(root)), path.stem)
        return tensors

    rows: list[dict[str, Any]] = []
    timing: dict[str, Any] = {"spectra": {}}
    refs = {r["model"]: r for r in index if r["ckpt"] == "ref"}
    spectra = sorted({(r["model"], r["regime"], r["seed"]) for r in index if r["ckpt"] != "ref"},
                     key=lambda k: (k[0], k[1], k[2]))  # fmt: skip
    for model, regime, seed in spectra:
        if model not in refs:
            raise FileNotFoundError(f"no reference entry for {model} in the embed index")
        ts = time.time()
        ref = read(refs[model])
        members = [
            r for r in index if (r["model"], r["regime"], r["seed"]) == (model, regime, seed)
        ]
        entries = [({**refs[model], "alpha": None}, ref)] + [(r, read(r)) for r in members]
        spectrum = {"model": model, "regime": regime, "seed": seed}
        rows += list(spectrum_rows(ref, entries, union, spectrum))
        timing["spectra"][f"{model}/{regime}-s{seed}"] = round(time.time() - ts, 1)
    report = sanity(rows)
    out_csv = ctx.stage_dir / "delta_b.csv.gz"
    _write_csv(out_csv, rows)
    (ctx.stage_dir / "sanity.json").write_text(json.dumps(report, indent=2))
    timing["seconds"] = round(time.time() - t0, 1)
    timing["rows"] = len(rows)
    (ctx.stage_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    res = results_dir(ctx.cfg, root)
    _write_csv(res / "delta_b.csv", [r for r in rows if r["cell"] == "primary"])
    _write_csv(res / "delta_b_cells.csv.gz", rows)
    (res / "sanity.json").write_text(json.dumps(report, indent=2))
    rel = {name: str((ctx.stage_dir / name).relative_to(root))
           for name in ("delta_b.csv.gz", "sanity.json")}  # fmt: skip
    return StageResult(outputs=rel)  # timing.json is a log, not an output
