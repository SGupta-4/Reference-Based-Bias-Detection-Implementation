"""The merge spectrum: α list, checkpoint slugs, endpoint lookup and α activation on one
loaded base (App. C; D-004, D-007, D-008, D-066, D-081).

α is the weight on the unharmful endpoint: `a100` is the unharmful adapter or weights,
`a000` the harmful ones, `ref` the base model with no fine-tuning.

Called by the `embed` stage (M4) and later by `generate` (M5):
- `spectra(cfg, root)` finds every finished (model, regime, seed) u/h pair through
  `finetune.sft.plan_jobs` only, never by searching folders (B-020);
- `Activator` holds one loaded base model per model entry and switches it to any
  checkpoint in place:
  - LoRA / QLoRA: the trained adapter at α ∈ {1, 0}, the exact rank-2r concatenation in
    between (`models.adapters`), loaded through PEFT onto the un-quantised inference base
    (D-004) from a temporary directory on ephemeral disk that is deleted right after;
  - full FT: `models.interpolate.apply_alpha` writes (1−α)·W_h + α·W_u into the weights.
  The reference is always extracted first, from the plain base, so it never sees an
  adapter or interpolated weights.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALPHAS: tuple[float, ...] = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0)
REF = "ref"


def slug_for(alpha: float) -> str:
    """Checkpoint slug of α: 1.0 -> "a100", 0.5 -> "a050", 0.0 -> "a000"."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return f"a{round(alpha * 100):03d}"


def alpha_of(slug: str) -> float | None:
    """Inverse of `slug_for`; None for "ref"."""
    if slug == REF:
        return None
    if len(slug) != 4 or slug[0] != "a" or not slug[1:].isdigit():
        raise ValueError(f"not a checkpoint slug: {slug!r}")
    return int(slug[1:]) / 100.0


CHECKPOINTS: tuple[str, ...] = (REF, *(slug_for(a) for a in ALPHAS))


@dataclass(frozen=True)
class Spectrum:
    """One finished u/h endpoint pair: `finals[split]` is the job's `final/` directory
    (PEFT adapter or fp16 full weights) and `train_keys[split]` its content key."""

    model: dict[str, Any] = field(hash=False)
    regime: str
    seed: int
    finals: dict[str, Path] = field(hash=False)
    train_keys: dict[str, str] = field(hash=False)

    @property
    def name(self) -> str:
        """`<regime>-s<seed>`, the namespace level under the model slug."""
        return f"{self.regime}-s{self.seed}"


def spectra(cfg: Any, artifacts_root: Path, ftdata_outputs: Sequence[str]) -> list[Spectrum]:
    """Every (model, regime, seed) whose two endpoints have `done.json`, in plan order.

    Order: per model, LoRA-type regimes before full FT (the `Activator` restores the plain
    base by unloading PEFT, while full FT overwrites the weights), seed-major within each.
    Raises if a planned pair is unfinished: a sweep over half a spectrum is never valid.
    """
    from rbbd.finetune.sft import plan_jobs

    pairs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for job in plan_jobs(cfg, artifacts_root, ftdata_outputs):
        key = (job.model_entry["slug"], job.regime, job.seed)
        if not (job.out_dir / "done.json").exists():
            slug = job.model_entry["slug"]
            raise FileNotFoundError(
                f"endpoint not trained: {job.out_dir} (restore train/{slug} first)"
            )
        pair = pairs.setdefault(key, {"model": job.model_entry, "finals": {}, "train_keys": {}})
        pair["finals"][job.split] = job.out_dir / "final"
        pair["train_keys"][job.split] = job.spec.train_key
    out = [
        Spectrum(p["model"], regime, seed, p["finals"], p["train_keys"])
        for (_, regime, seed), p in pairs.items()
    ]
    order = {m["slug"]: i for i, m in enumerate(cfg.get("models", []))}
    return sorted(out, key=lambda s: (order[s.model["slug"]], s.regime == "full", s.seed))


def checkpoint_fields(sp: Spectrum, slug: str, endpoint_hashes: dict[str, str]) -> dict[str, Any]:
    """The `checkpoint` part of an embedding cache key (D-019) for `slug` of `sp`.

    Everything else in the key (model load spec, layer, extraction settings, union hash,
    schema version) is shared with the reference entry (D-004; `embed.extract`
    asserts it). `endpoint_hashes` = `manifest.hash_path` of each `final/` directory.
    """
    alpha = alpha_of(slug)
    if alpha is None:
        return {"slug": REF}
    if sp.regime == "full":
        merge = "interp-fp32"  # D-008
    else:
        merge = "endpoint" if alpha in (0.0, 1.0) else "concat-rank2r"  # D-066, D-007
    return {
        "slug": slug,
        "alpha": alpha,
        "regime": sp.regime,
        "seed": sp.seed,
        "merge": merge,
        "endpoints": {
            split: {"train_key": sp.train_keys[split], "hash": endpoint_hashes[split]}
            for split in ("unharmful", "harmful")
        },
    }


class Activator:
    """One loaded base model, switched in place between the checkpoints of its spectra.

    `load` returns (model, tokenizer) of the plain inference base (the same `LoadSpec` as
    the reference, D-004); it is called lazily, on the first checkpoint that misses the
    cache, so a fully cached sweep never loads a model (DC-10). `get(sp, slug)` returns
    the (model, tokenizer) whose base decoder now computes that checkpoint.
    """

    def __init__(self, load: Callable[[], tuple[Any, Any]], scratch: Path | None = None) -> None:
        self._load = load
        self._scratch = scratch
        self.model: Any = None
        self.tokenizer: Any = None
        self.state = "unloaded"  # unloaded | plain | peft:<name>:<slug> | full:<name>
        self._endpoints: dict[str, Any] = {}
        self._adapters: dict[str, Any] = {}

    def _base(self) -> None:
        if self.model is None:
            self.model, self.tokenizer = self._load()
            self.state = "plain"

    def plain(self) -> tuple[Any, Any]:
        """The base with no adapter and its original weights (the reference)."""
        self._base()
        if self.state.startswith("full"):
            raise RuntimeError("reference requested after full-FT weights were applied")
        if self.state.startswith("peft"):
            self.model = self.model.unload()
            self.state = "plain"
        return self.model, self.tokenizer

    def get(self, sp: Spectrum, slug: str) -> tuple[Any, Any]:
        """Switch to checkpoint `slug` of `sp` and return (model, tokenizer)."""
        alpha = alpha_of(slug)
        if alpha is None:
            return self.plain()
        self._base()
        if sp.regime == "full":
            return self._full(sp, alpha)
        return self._lora(sp, slug, alpha)

    def _lora(self, sp: Spectrum, slug: str, alpha: float) -> tuple[Any, Any]:
        from peft import PeftModel

        from rbbd.models import adapters as ad

        if self.state.startswith("full"):
            raise RuntimeError("LoRA spectra must run before full-FT spectra (see `spectra`)")
        if sp.name not in self._adapters:
            self._adapters = {sp.name: (ad.read_adapter(sp.finals["unharmful"]),
                                        ad.read_adapter(sp.finals["harmful"]))}  # fmt: skip
        u, h = self._adapters[sp.name]
        tmp = Path(tempfile.mkdtemp(prefix=f"rbbd_{slug}_", dir=self._scratch))
        try:
            path = ad.write_adapter(ad.adapter_for_alpha(u, h, alpha), tmp / slug)
            name = f"{sp.name}-{slug}".replace(".", "_")
            if self.state == "plain":
                self.model = PeftModel.from_pretrained(self.model, str(path), adapter_name=name)
            else:
                previous = self.state.split(":", 1)[1]
                self.model.load_adapter(str(path), adapter_name=name)
                self.model.set_adapter(name)
                self.model.delete_adapter(previous)
            self.model.eval()
            self.state = f"peft:{name}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return self.model, self.tokenizer

    def _full(self, sp: Spectrum, alpha: float) -> tuple[Any, Any]:
        from rbbd.models import interpolate as ip

        if self.state.startswith("peft"):
            self.model = self.model.unload()
        if sp.name not in self._endpoints:
            self._endpoints = {}  # one spectrum's fp32 endpoints in RAM at a time (D-008)
            w_u = ip.load_endpoint(sp.finals["unharmful"])
            w_h = ip.load_endpoint(sp.finals["harmful"])
            ip.check_endpoints(w_h, w_u)
            self._endpoints[sp.name] = (w_h, w_u)
        w_h, w_u = self._endpoints[sp.name]
        ip.apply_alpha(self.model, w_h, w_u, alpha)
        self.state = f"full:{sp.name}"
        return self.model, self.tokenizer

    def close(self) -> None:
        """Drop the model and endpoints so the next model entry can load."""
        self.model = self.tokenizer = None
        self._endpoints, self._adapters = {}, {}
        self.state = "unloaded"
