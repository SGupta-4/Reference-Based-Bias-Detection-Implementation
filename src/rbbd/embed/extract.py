"""Batched, length-sorted, guarded final-layer extraction into the embedding cache
(§4.1; D-004, D-006, D-017–D-020).

Call path (FLOW.md › Stage: embed): `stage` -> `extract_cached` -> (cache hit: read, no
model load, no forward) or (`load` -> `encode` -> `TensorCache.write`).

`encode` assumes a right-padding tokenizer (`models.loading.prepare_tokenizer`) and a
causal LM whose inner decoder is reachable with `models.loading.base_decoder`. Batches
are formed longest-first under a token budget, so padding waste stays small (E1) and an
out-of-memory error shows up on the first batch rather than the last. Every batch's
hidden states and pooled vectors pass the NaN/Inf guard before anything is cached (D-006).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rbbd.data.hashing import text_hash
from rbbd.embed.pooling import POOLINGS, pool_all
from rbbd.models.loading import LoadSpec, base_decoder, input_device, num_layers
from rbbd.utils.cache import TensorCache, make_key
from rbbd.utils.guards import assert_finite
from rbbd.utils.logging import get_logger

log = get_logger(__name__)

LAYER_SITES = ("post_norm", "pre_norm")


@dataclass(frozen=True)
class ExtractSettings:
    """Extraction settings; all of them enter the cache key."""

    poolings: tuple[str, ...] = POOLINGS
    layer_sites: tuple[str, ...] = ("post_norm",)
    max_length: int = 64
    token_budget: int = 16384
    max_batch: int = 256

    def key_fields(self) -> dict[str, Any]:
        """Settings plus the fixed tokenizer policy of D-018."""
        return {
            **asdict(self),
            "tokenizer": {
                "add_special_tokens": True,
                "padding_side": "right",
                "chat_template": False,
            },
        }

    @staticmethod
    def from_config(cfg: Mapping[str, Any]) -> ExtractSettings:
        """Build from the `embed` config section, ignoring keys that are not settings."""
        names = {f for f in ExtractSettings.__dataclass_fields__}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in names}
        return ExtractSettings(**kw)


@dataclass
class ExtractStats:
    """What one `encode` call cost; written to the stage's timing.json (DC-12)."""

    n_texts: int = 0
    n_batches: int = 0
    forward_calls: int = 0
    real_tokens: int = 0
    padded_tokens: int = 0
    seconds: float = 0.0
    peak_mem_gib: dict[str, float] = field(default_factory=dict)

    @property
    def padding_waste(self) -> float:
        """Share of computed token positions that were padding."""
        return 1.0 - self.real_tokens / self.padded_tokens if self.padded_tokens else 0.0


def output_names(settings: ExtractSettings) -> list[str]:
    """Tensor names in the cache file: "mean" for post-norm, "pre_norm/mean" for pre-norm."""
    return [
        k if site == "post_norm" else f"{site}/{k}"
        for site in settings.layer_sites
        for k in settings.poolings
    ]


def token_lengths(tokenizer: Any, texts: Sequence[str], max_length: int) -> list[int]:
    """Token count of each text with special tokens, truncated at `max_length`."""
    enc = tokenizer(list(texts), add_special_tokens=True, truncation=True, max_length=max_length)
    return [len(ids) for ids in enc["input_ids"]]


def make_batches(lengths: Sequence[int], token_budget: int, max_batch: int) -> list[list[int]]:
    """Index batches, longest first, with batch_size x longest_length <= token_budget."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        longest = lengths[current[0]] if current else lengths[i]
        if current and ((len(current) + 1) * longest > token_budget or len(current) >= max_batch):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def encode(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    settings: ExtractSettings,
    *,
    context: Mapping[str, Any] | None = None,
    stats: ExtractStats | None = None,
) -> dict[str, np.ndarray]:
    """Pooled final-layer embeddings of `texts` -> {name: [N, d] float16}, rows in input order.

    `context` (model slug, checkpoint) is attached to guard errors. A pre-norm hook on
    the last decoder layer is registered only when "pre_norm" is requested (D-017).
    """
    import torch

    context = dict(context or {})
    stats = stats if stats is not None else ExtractStats()
    start = time.time()
    decoder = base_decoder(model)
    device = input_device(model)
    lengths = token_lengths(tokenizer, texts, settings.max_length)
    batches = make_batches(lengths, settings.token_budget, settings.max_batch)
    names = output_names(settings)
    out: dict[str, np.ndarray] = {}
    captured: dict[str, Any] = {}
    handle = None
    if "pre_norm" in settings.layer_sites:

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured["pre_norm"] = output[0] if isinstance(output, tuple) else output

        handle = decoder.layers[-1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for bi, idx in enumerate(batches):
                enc = tokenizer(
                    [texts[i] for i in idx],
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=settings.max_length,
                    return_tensors="pt",
                )
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                result = decoder(input_ids=ids, attention_mask=mask)
                stats.forward_calls += 1
                stats.real_tokens += int(mask.sum())
                stats.padded_tokens += int(mask.numel())
                sites = {
                    "post_norm": result.last_hidden_state,
                    "pre_norm": captured.get("pre_norm"),
                }
                for site in settings.layer_sites:
                    hidden = sites[site]
                    assert_finite(hidden, f"embed.{site}", batch=bi, **context)
                    pooled = pool_all(hidden, mask.to(hidden.device), settings.poolings)
                    for kind, vec in pooled.items():
                        name = kind if site == "post_norm" else f"{site}/{kind}"
                        half = vec.to(torch.float16)
                        # fp32 models can produce values beyond the fp16 range of the cache.
                        assert_finite(half, f"embed.{name}.fp16", batch=bi, **context)
                        if name not in out:
                            out[name] = np.zeros((len(texts), half.shape[1]), dtype=np.float16)
                        out[name][idx] = half.cpu().numpy()
    finally:
        if handle is not None:
            handle.remove()
    stats.n_texts = len(texts)
    stats.n_batches = len(batches)
    stats.seconds = round(time.time() - start, 3)
    if torch.cuda.is_available():
        stats.peak_mem_gib = {
            f"cuda:{d}": round(torch.cuda.max_memory_allocated(d) / 2**30, 2)
            for d in range(torch.cuda.device_count())
        }
    return {n: out[n] for n in names}


def extract_cached(
    cache: TensorCache,
    namespace: str,
    key_fields: Mapping[str, Any],
    texts: Sequence[str],
    settings: ExtractSettings,
    load: Callable[[], tuple[Any, Any]],
    *,
    context: Mapping[str, Any] | None = None,
    stats: ExtractStats | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
    """Return (tensors, sidecar, hit). On a cache hit `load` is never called (DC-10)."""
    key = make_key(key_fields)
    if cache.lookup(namespace, key) is not None:
        tensors, meta = cache.read(namespace, key)
        return tensors, meta, True
    model, tokenizer = load()
    stats = stats if stats is not None else ExtractStats()
    tensors = encode(model, tokenizer, texts, settings, context=context, stats=stats)
    sidecar = {
        "fields": dict(key_fields),
        "text_hashes": [text_hash(t) for t in texts],
        "stats": {**asdict(stats), "padding_waste": round(stats.padding_waste, 4)},
    }
    path = cache.write(namespace, key, tensors, sidecar)
    log.info("cached %d embeddings at %s", len(texts), path)
    return tensors, sidecar, False


def resolve_revision(model_id: str, revision: str | None) -> str:
    """Pin a model to a commit sha (configured, else the Hub's current one) for the cache key."""
    if revision:
        return revision
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id).sha


def _free(model: Any) -> None:
    import gc

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for d in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(d)
    except ImportError:
        pass


def stage(ctx: Any) -> Any:
    """`embed` stage (M2: reference checkpoint only; α-checkpoints arrive in M4).

    Reads the sentence union from the upstream `sentences` manifest, extracts every
    configured model once, and writes `embed/<run_key>/timing.json`. Outputs are the
    cache files, so a deleted or altered cache entry invalidates the manifest.
    """
    from rbbd.models.loading import download, load_for_inference
    from rbbd.runner import StageResult

    root = Path(ctx.env.artifacts_root)
    union_rel = next(p for p in ctx.upstream["sentences"].output_hashes if p.endswith("union.json"))
    union = json.loads((root / union_rel).read_text())
    texts = union["texts"]
    emb_cfg = ctx.cfg.get("embed", {})
    checkpoints = emb_cfg.get("checkpoints", ["ref"])
    if list(checkpoints) != ["ref"]:
        raise NotImplementedError("α-checkpoint extraction is implemented in M4 (PLAN §7)")
    settings = ExtractSettings.from_config(emb_cfg)
    cache = TensorCache(root)
    outputs: dict[str, str] = {}
    timing: dict[str, Any] = {
        "n_texts": len(texts),
        "union_hash": union["union_hash"],
        "models": {},
    }
    for m in ctx.cfg["models"]:
        spec = LoadSpec(
            model_id=m["id"],
            revision=resolve_revision(m["id"], m.get("revision")),
            precision=m.get("precision", "fp16"),
            placement=m.get("placement", "balanced"),
        )
        key_fields = {
            "model": spec.key_fields(),
            "checkpoint": {"slug": "ref"},
            "layer": int(m["layer"]),
            "extract": settings.key_fields(),
            "union_hash": union["union_hash"],
            "schema_version": int(ctx.cfg.get("schema_version.embed", 1)),
        }
        namespace = f"embeddings/{m['slug']}/base/ref"
        loaded: dict[str, Any] = {}

        def load(
            spec: LoadSpec = spec, m: Mapping[str, Any] = m, loaded: dict[str, Any] = loaded
        ) -> tuple[Any, Any]:
            t0 = time.time()
            download(spec)
            loaded["download_seconds"] = round(time.time() - t0, 1)
            t0 = time.time()
            model, tok = load_for_inference(spec)
            loaded["load_seconds"] = round(time.time() - t0, 1)
            if num_layers(model) != int(m["layer"]):
                raise ValueError(
                    f"{m['id']}: config layer {m['layer']} != {num_layers(model)} decoder layers"
                )
            loaded["model"] = model
            return model, tok

        stats = ExtractStats()
        _, sidecar, hit = extract_cached(
            cache, namespace, key_fields, texts, settings, load,
            context={"model": m["slug"], "ckpt": "ref"}, stats=stats,
        )  # fmt: skip
        key = make_key(key_fields)
        path = cache.path_for(namespace, key)
        outputs[f"{m['slug']}/ref"] = str(path.relative_to(root))
        timing["models"][m["slug"]] = {
            "cache_hit": hit,
            "key": key,
            "download_seconds": loaded.get("download_seconds"),
            "load_seconds": loaded.get("load_seconds"),
            **(sidecar["stats"] if not hit else {"stats": sidecar.get("stats")}),
        }
        if "model" in loaded:
            _free(loaded.pop("model"))
    path = ctx.stage_dir / "timing.json"
    path.write_text(json.dumps(timing, indent=2))
    outputs["timing"] = str(path.relative_to(root))
    return StageResult(outputs=outputs)


def cached_ref_entry(cfg: Any, env: Any, slug: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read the reference embeddings and the sentence union recorded by a config's manifests."""
    from rbbd import runner
    from rbbd.utils import manifest as mf

    keys = runner.stage_run_keys(cfg)
    root = Path(env.artifacts_root)
    emb = mf.read(mf.manifest_path(root, "embed", keys["embed"]))
    sen = mf.read(mf.manifest_path(root, "sentences", keys["sentences"]))
    if emb is None or not emb.complete or sen is None:
        raise FileNotFoundError(f"no complete embed/sentences manifest for {cfg.path}")
    rel = next(p for p in emb.output_hashes if f"/{slug}/base/ref/" in p)
    path = root / rel
    tensors, _ = TensorCache(root).read(str(path.parent.relative_to(root)), path.stem)
    union_rel = next(p for p in sen.output_hashes if p.endswith("union.json"))
    return tensors, json.loads((root / union_rel).read_text())


def compare_entries(
    a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], union: Mapping[str, Any]
) -> dict[str, Any]:
    """How far two extractions of the same sentences differ (M2-T4 Gemma fp16 vs fp32, D-005).

    Per pooling: row-wise cosine (mean, p01, min). For mean pooling: per-group B under RR
    and SEAT on the base sets, the largest absolute difference, and the spread of B across
    groups in `b` for scale.
    """
    from rbbd.metrics.delta_b import from_union, group_bias

    out: dict[str, Any] = {"cosine": {}}
    for name in sorted(set(a) & set(b)):
        x = a[name].astype(np.float64)
        y = b[name].astype(np.float64)
        cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        out["cosine"][name] = {
            "mean": float(cos.mean()), "p01": float(np.percentile(cos, 1)), "min": float(cos.min())
        }  # fmt: skip
    for method in ("rr", "seat"):
        ba = group_bias(from_union(a["mean"], union), method)
        bb = group_bias(from_union(b["mean"], union), method)
        diffs = [abs(ba[g] - bb[g]) for g in bb]
        out[f"bias_{method}"] = {
            "max_abs_diff": float(max(diffs)),
            "group_spread_b": float(max(bb.values()) - min(bb.values())),
        }
    return out
