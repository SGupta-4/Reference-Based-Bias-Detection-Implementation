"""Benchmark generation: one process per checkpoint, merged weights on ephemeral disk, one
vLLM engine for all benchmarks, resumable JSONL shards (D-050, D-086; PLAN M5-T2).

Call path (FLOW.md › Stage: generate):
`stage` → per (model, spectrum or ref, checkpoint) a `cli generate-one` process
(`_launch`), two at a time on the two GPUs for TP=1 models → `run_checkpoint`:
1. `gen_jobs`: one job per benchmark with its `gen_key` and output directory; jobs whose
   `done.json` exists are skipped, and if all are, nothing is loaded.
2. `materialize`: `ref` → the downloaded base; LoRA/QLoRA → W₀ + ΔW(α) per target module
   in fp32 (exact rank-2r combination, D-007/D-066), cast to the inference dtype and
   saved to `ephemeral_dir()`; full FT → `interpolate.materialize` (D-008). Never under
   the artifact root (asserted).
3. `make_engine` (vLLM, TP per model, dtype per D-004/D-005) → `generate_bench` per
   benchmark: chat-template prompts (`render`), per-item seeds shared by every checkpoint
   (paired sampling noise, D-026), shards of `shard_size` items written atomically,
   existing shards skipped (DC-13), `done.json` at the end.
4. The merged copy is deleted (D-050), whatever happened.
A session deadline (`$RBBD_DEADLINE_UNIX`) is checked between shards: the process stops
cleanly with exit 76 and the next session resumes (as training does, D-079).

Generations are harmful-model outputs: they live under `generations/` in the artifact
root and go to the private store only (D-037); nothing here prints generated text.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rbbd.utils.cache import make_key
from rbbd.utils.logging import get_logger

log = get_logger(__name__)

DEADLINE_ENV = "RBBD_DEADLINE_UNIX"
PAUSE_EXIT = 76


@dataclass(frozen=True)
class Sampling:
    """Sampling settings of one benchmark; all enter the gen_key."""

    n: int
    temperature: float
    top_p: float
    max_new_tokens: int


# App. C / D-022–D-025 / D-086 defaults; `bench.<name>.n` etc. in the config override.
DEFAULT_SAMPLING = {
    "dt": Sampling(n=3, temperature=1.0, top_p=1.0, max_new_tokens=150),
    "toxigen": Sampling(n=5, temperature=0.7, top_p=1.0, max_new_tokens=128),
    "wgm": Sampling(n=5, temperature=0.7, top_p=1.0, max_new_tokens=256),
}
BENCHES = tuple(DEFAULT_SAMPLING)


class GenerationPaused(RuntimeError):
    """Stopped at the session deadline between shards; finished shards are kept."""


def sampling_for(cfg: Any, bench: str) -> Sampling:
    """The benchmark's sampling settings, with config overrides (`bench.<name>.*`)."""
    base = asdict(DEFAULT_SAMPLING[bench])
    over = cfg.get(f"bench.{bench}", {}) or {}
    return Sampling(**{k: type(v)(over.get(k, v)) for k, v in base.items()})


def item_seed(base_seed: int, item_id: str) -> int:
    """Per-item sampling seed: the same for every checkpoint, so Δscores are paired."""
    return int(hashlib.sha256(f"{base_seed}:{item_id}".encode()).hexdigest()[:8], 16)


def subset_hash(items: Iterable[Mapping[str, Any]]) -> str:
    """Hash of the item ids (order-free): which prompts a gen_key covers (D-026)."""
    return hashlib.sha256("\n".join(sorted(i["id"] for i in items)).encode()).hexdigest()[:16]


def render(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> tuple[str, str]:
    """(prompt text, mode). Chat template with a generation prompt; a template that
    refuses a system turn gets it merged into the first user turn (mode "merged-system")."""
    try:
        text = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        return text, "chat"
    except Exception:  # noqa: BLE001 - jinja raises TemplateError for unsupported roles
        if not messages or messages[0]["role"] != "system":
            raise
        system, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
        rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
        text = tokenizer.apply_chat_template(rest, tokenize=False, add_generation_prompt=True)
        return text, "merged-system"


def gen_key_fields(
    model: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    bench: str,
    items_hash: str,
    sampling: Sampling,
    engine: Mapping[str, Any],
    bench_meta: Mapping[str, Any],
    schema_version: int,
) -> dict[str, Any]:
    """Every field that determines one benchmark's generations (ARCHITECTURE § cache keys)."""
    return {
        "model": dict(model),
        "checkpoint": dict(checkpoint),
        "bench": bench,
        "items": items_hash,
        "bench_source": dict(bench_meta),
        "sampling": asdict(sampling),
        "engine": dict(engine),
        "schema_version": schema_version,
    }


def shard_path(out_dir: Path, index: int) -> Path:
    return Path(out_dir) / f"part-{index:04d}.jsonl"


def _deadline() -> float | None:
    raw = os.environ.get(DEADLINE_ENV)
    return float(raw) if raw else None


def generate_bench(
    generate_fn: Callable[[list[str], list[Mapping[str, Any]]], list[list[dict[str, Any]]]],
    tokenizer: Any,
    items: Sequence[Mapping[str, Any]],
    sampling: Sampling,
    out_dir: Path,
    *,
    shard_size: int = 256,
    base_seed: int = 0,
    max_prompt_tokens: int | None = None,
    key_fields: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Generate `sampling.n` samples per item into JSONL shards under `out_dir`.

    `generate_fn(prompts, params)` returns, per prompt, `n` dicts {"text", "finish_reason",
    "n_tokens"} (the vLLM adapter is `vllm_generate_fn`; tests pass a stub). Each shard is
    written to a temporary file and renamed, so a killed run leaves only whole shards;
    existing shards are skipped (DC-13). Items whose rendered prompt exceeds
    `max_prompt_tokens` are skipped and listed in `done.json` (never truncated).
    Returns the `done.json` payload.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done.json"
    if done_path.exists():
        return json.loads(done_path.read_text())
    deadline = _deadline()
    t0 = clock()
    shards = [items[i : i + shard_size] for i in range(0, len(items), shard_size)]
    skipped: list[str] = []
    modes: dict[str, int] = {}
    new_shards = 0
    for index, chunk in enumerate(shards):
        path = shard_path(out_dir, index)
        if path.exists():
            continue
        if deadline is not None and clock() >= deadline:
            raise GenerationPaused(f"{out_dir.name}: paused before shard {index}/{len(shards)}")
        prompts, params, kept = [], [], []
        for item in chunk:
            text, mode = render(tokenizer, item["messages"])
            modes[mode] = modes.get(mode, 0) + 1
            if (
                max_prompt_tokens is not None
                and len(tokenizer(text)["input_ids"]) > max_prompt_tokens
            ):
                skipped.append(item["id"])
                continue
            prompts.append(text)
            params.append({**asdict(sampling), "seed": item_seed(base_seed, item["id"])})
            kept.append(item)
        outputs = generate_fn(prompts, params) if prompts else []
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for item, samples in zip(kept, outputs, strict=True):
                for k, s in enumerate(samples):
                    fh.write(
                        json.dumps({"id": item["id"], "sample": k, **s}, ensure_ascii=False) + "\n"
                    )
        os.replace(tmp, path)
        new_shards += 1
    # Items skipped in shards finished by an earlier process are recovered from the shards.
    present = {
        json.loads(line)["id"]
        for i in range(len(shards))
        for line in open(shard_path(out_dir, i), encoding="utf-8")
    }
    skipped_all = sorted({i["id"] for i in items} - present)
    done = {
        "key": make_key(key_fields) if key_fields is not None else None,
        "fields": dict(key_fields or {}),
        "n_items": len(items),
        "n_shards": len(shards),
        "shards_this_process": new_shards,
        "skipped_too_long": skipped_all,
        "render_modes": modes,
        "seconds_this_process": round(clock() - t0, 1),
    }
    done_path.write_text(json.dumps(done, indent=2))
    return done


def read_samples(out_dir: Path) -> list[dict[str, Any]]:
    """Every sample of a finished benchmark directory (private text; callers never print)."""
    done = json.loads((Path(out_dir) / "done.json").read_text())
    rows = []
    for i in range(done["n_shards"]):
        with open(shard_path(out_dir, i), encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    return rows


# --------------------------------------------------------------------------- checkpoints


def materialize(
    load_spec: Any,
    spectrum: Any | None,
    slug: str,
    out_dir: Path,
    artifacts_root: Path,
) -> tuple[str, bool]:
    """(path vLLM loads, whether it is a temporary copy to delete) for one checkpoint.

    `ref` is the downloaded base snapshot. LoRA/QLoRA: the base is loaded on CPU in the
    inference dtype; for every adapter module W ← cast(fp32(W) + ΔW(α)) with ΔW from
    `adapters.delta_w(adapter_for_alpha(u, h, α))` (exact; endpoints are the trained
    adapters). Full FT: `interpolate.materialize` (D-008). Gemma stays fp32 (D-005).
    """
    from rbbd.models import interpolate as ip
    from rbbd.models.loading import download, load_tokenizer, torch_dtype
    from rbbd.models.spectrum import alpha_of

    if spectrum is None:
        return download(load_spec), False
    import torch
    from transformers import AutoModelForCausalLM

    out = ip.assert_ephemeral(out_dir, artifacts_root)
    alpha = alpha_of(slug)
    download(load_spec)
    model = AutoModelForCausalLM.from_pretrained(
        load_spec.model_id,
        revision=load_spec.revision,
        dtype=torch_dtype(load_spec.precision),
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tok = load_tokenizer(load_spec.model_id, load_spec.revision)
    if spectrum.regime == "full":
        # One tensor at a time (B-033): two parallel processes each holding both fp32
        # endpoints of Llama-1B exceeded Kaggle's 31 GiB.
        w_u = ip.LazyEndpoint(spectrum.finals["unharmful"])
        w_h = ip.LazyEndpoint(spectrum.finals["harmful"])
        ip.check_endpoints(w_h, w_u)
        ip.materialize(model, tok, w_h, w_u, alpha, out, artifacts_root)
    else:
        from rbbd.models import adapters as ad

        u = ad.read_adapter(spectrum.finals["unharmful"])
        h = ad.read_adapter(spectrum.finals["harmful"])
        modules = dict(model.named_modules())
        with torch.no_grad():
            for prefix, dw in ad.delta_w(ad.adapter_for_alpha(u, h, alpha)).items():
                layer = modules[prefix.removeprefix("base_model.model.")]
                w = layer.weight
                w.copy_((w.float() + dw.to(w.device)).to(w.dtype))
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out), safe_serialization=True)
        tok.save_pretrained(str(out))
    del model
    return str(out), True


def engine_settings(cfg: Any, entry: Mapping[str, Any]) -> dict[str, Any]:
    """vLLM settings for a model entry (`bench.engine` defaults, `models[].bench_*` overrides)."""
    eng = dict(cfg.get("bench.engine", {}) or {})
    big = entry.get("placement") == "balanced"
    return {
        "tensor_parallel_size": int(entry.get("bench_tp", 2 if big else 1)),
        "dtype": {"fp16": "float16", "fp32": "float32"}[entry.get("precision", "fp16")],
        "max_model_len": int(eng.get("max_model_len", 4096)),
        "gpu_memory_utilization": float(eng.get("gpu_memory_utilization", 0.88)),
        "enforce_eager": bool(eng.get("enforce_eager", True)),
        "seed": int(cfg.get("seed", 0)),
    }


def make_engine(path: str, settings: Mapping[str, Any]) -> Any:
    """A vLLM `LLM` on `path` (V0 engine on sm75, D-048). Gemma-3's vision tower is
    disabled for text-only prompts."""
    from vllm import LLM

    kwargs = dict(settings)
    if "gemma-3" in path.lower() or "gemma3" in path.lower():
        kwargs["limit_mm_per_prompt"] = {"image": 0}
    return LLM(model=path, trust_remote_code=False, **kwargs)


def vllm_generate_fn(llm: Any) -> Callable[[list[str], list[Mapping[str, Any]]], list[list[dict]]]:
    """Adapter from `generate_bench`'s interface to `LLM.generate` with per-prompt params."""
    from vllm import SamplingParams

    def fn(prompts: list[str], params: list[Mapping[str, Any]]) -> list[list[dict]]:
        sps = [
            SamplingParams(
                n=p["n"],
                temperature=p["temperature"],
                top_p=p["top_p"],
                max_tokens=p["max_new_tokens"],
                seed=p["seed"],
            )
            for p in params
        ]
        results = llm.generate(prompts, sampling_params=sps, use_tqdm=False)
        return [
            [
                {"text": o.text, "finish_reason": o.finish_reason, "n_tokens": len(o.token_ids)}
                for o in r.outputs
            ]
            for r in results
        ]

    return fn


# --------------------------------------------------------------------------- prompt sets


def prepare_items(cfg: Any, out_dir: Path, benches: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Build each benchmark's prompt set once per session (network: DT and ToxiGen files at
    their pinned commits, WildGuardMix at a resolved revision).

    Writes, per benchmark, `items_<bench>.json` (with prompt text) under `out_dir`, which
    must be ephemeral disk: prompts are never persisted or synced. Returns {bench: meta}
    where meta = {"source": {...}, "items": subset hash, "n_items", "tags": {id: tags}}; the
    tags (group, topic, system-prompt type; no text) are what scoring needs.
    """
    from rbbd.bench import decodingtrust as dt
    from rbbd.bench import toxigen as tg
    from rbbd.bench import wildguard as wg

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = {}
    for bench in benches:
        bcfg = cfg.get(f"bench.{bench}", {}) or {}
        if bench == "dt":
            items = dt.build_prompts(dt.fetch("data"), dt.fetch("system"))
            items = dt.subset(items, bcfg.get("groups"), bcfg.get("topics"))
            source = {"repo": dt.DT_REPO, "commit": dt.DT_COMMIT}
            tags = {
                i["id"]: {k: i[k] for k in ("sys_type", "group", "topic", "variant")} for i in items
            }
        elif bench == "toxigen":
            groups = bcfg.get("groups") or list(tg.GROUP_MAP)
            files = {g: tg.fetch(g) for g in groups}
            items = tg.build_prompts(
                files,
                n=int(bcfg.get("n_prompts", tg.N_PROMPTS)),
                seed=int(cfg.get("seed", 0)),
                groups=groups,
            )
            source = {"repo": tg.TOXIGEN_REPO, "commit": tg.TOXIGEN_COMMIT}
            tags = {i["id"]: {"group": i["group"]} for i in items}
        elif bench == "wgm":
            items, sha = wg.load_prompts(bcfg.get("revision"))
            if bcfg.get("max_prompts"):
                items = items[: int(bcfg["max_prompts"])]
            source = {"dataset": wg.DATASET, "config": wg.CONFIG, "revision": sha}
            tags = {i["id"]: {"adversarial": i.get("adversarial")} for i in items}
        else:
            raise ValueError(f"unknown benchmark {bench!r}")
        (out_dir / f"items_{bench}.json").write_text(json.dumps(items, ensure_ascii=False))
        metas[bench] = {
            "source": source,
            "items": subset_hash(items),
            "n_items": len(items),
            "tags": tags,
        }
    return metas


# --------------------------------------------------------------------------- planning


@dataclass(frozen=True)
class GenJob:
    """One (checkpoint, benchmark): where its shards go and what determines them."""

    model_slug: str
    spectrum: str  # "base" for the reference, else "<regime>-s<seed>"
    ckpt: str
    bench: str
    out_dir: Path
    key: str
    fields: dict

    def done(self) -> bool:
        return (self.out_dir / "done.json").exists()


def engine_fields(cfg: Any, entry: Mapping[str, Any]) -> dict[str, Any]:
    """The engine part of a gen_key: vLLM version and the settings that change outputs."""
    from importlib import metadata

    try:
        version = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        version = "absent"
    s = engine_settings(cfg, entry)
    return {
        "name": "vllm",
        "version": version,
        "dtype": s["dtype"],
        "tp": s["tensor_parallel_size"],
        "max_model_len": s["max_model_len"],
        "enforce_eager": s["enforce_eager"],
        "seed": s["seed"],
    }


def plan_jobs(
    cfg: Any,
    root: Path,
    metas: Mapping[str, Mapping[str, Any]],
    spectra_list: Sequence[Any],
    revisions: Mapping[str, str],
) -> list[GenJob]:
    """Every (checkpoint, benchmark) job: per model the reference once, then each spectrum's
    checkpoints from `bench.checkpoints` (default `embed.checkpoints`), in sweep order,
    for the training seeds in `bench.seeds` (null: all)."""
    from rbbd.models import spectrum as spm
    from rbbd.models.loading import LoadSpec
    from rbbd.utils import manifest as mf

    slugs = list(cfg.get("bench.checkpoints") or cfg.get("embed.checkpoints", spm.CHECKPOINTS))
    schema = int(cfg.get("schema_version.gen", 1))
    jobs: list[GenJob] = []
    for entry in cfg["models"]:
        spec = LoadSpec(
            entry["id"],
            revisions[entry["slug"]],
            entry.get("precision", "fp16"),
            entry.get("placement", "balanced"),
        )
        eng = engine_fields(cfg, entry)
        plan: list[tuple[Any, str, dict]] = (
            [(None, spm.REF, {"slug": spm.REF})] if spm.REF in slugs else []
        )
        seeds = cfg.get("bench.seeds")  # PLAN M5-T8: seed 0 only; seeds are a ΔB ablation (D-033)
        mine = [s for s in spectra_list if s.model["slug"] == entry["slug"]]
        for sp in (s for s in mine if seeds is None or s.seed in seeds):
            hashes = {k: mf.hash_path(p) for k, p in sp.finals.items()}
            plan += [(sp, s, spm.checkpoint_fields(sp, s, hashes)) for s in slugs if s != spm.REF]
        for sp, slug, ckpt_fields in plan:
            name = "base" if sp is None else sp.name
            for bench, meta in metas.items():
                fields = gen_key_fields(
                    spec.key_fields(),
                    ckpt_fields,
                    bench,
                    meta["items"],
                    sampling_for(cfg, bench),
                    eng,
                    meta["source"],
                    schema,
                )
                key = make_key(fields)
                out = root / "generations" / bench / entry["slug"] / name / slug / key
                jobs.append(GenJob(entry["slug"], name, slug, bench, out, key, fields))
    return jobs


# --------------------------------------------------------------------------- one checkpoint


def run_checkpoint(
    cfg: Any,
    root: Path,
    jobs: Sequence[GenJob],
    items_dir: Path,
    spectrum: Any | None,
    engine_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Generate every unfinished benchmark of one checkpoint (the `generate-one` process).

    Merges the checkpoint to ephemeral disk if needed, starts one engine, runs the benches
    in order and deletes the merged copy afterwards, also on failure (D-050).
    `engine_factory(path, settings) -> generate_fn` is vLLM by default; tests pass a stub.
    """
    from rbbd.models.loading import LoadSpec, load_tokenizer
    from rbbd.utils.env import ephemeral_dir

    pending = [j for j in jobs if not j.done()]
    if not pending:
        return {"skipped": True}
    entry = next(m for m in cfg["models"] if m["slug"] == pending[0].model_slug)
    model_fields = pending[0].fields["model"]
    spec = LoadSpec(
        model_fields["model_id"],
        model_fields["revision"],
        model_fields["precision"],
        model_fields["placement"],
    )
    settings = engine_settings(cfg, entry)
    t0 = time.time()
    tmp_dir = (
        ephemeral_dir() / "rbbd_merged" / f"{entry['slug']}-{pending[0].spectrum}-{pending[0].ckpt}"
    )
    path, temporary = materialize(spec, spectrum, pending[0].ckpt, tmp_dir, root)
    timing: dict[str, Any] = {
        "materialize_seconds": round(time.time() - t0, 1),
        "materialize_peak_rss_gib": _peak_rss_gib(),
        "benches": {},
    }
    try:
        t1 = time.time()
        if engine_factory is None:
            gen_fn = vllm_generate_fn(make_engine(path, settings))
        else:
            gen_fn = engine_factory(path, settings)
        timing["engine_seconds"] = round(time.time() - t1, 1)
        tok = load_tokenizer(spec.model_id, spec.revision)
        for job in pending:
            items = json.loads((Path(items_dir) / f"items_{job.bench}.json").read_text())
            sampling = sampling_for(cfg, job.bench)
            t2, status = time.time(), "stopped"
            try:
                generate_bench(
                    gen_fn,
                    tok,
                    items,
                    sampling,
                    job.out_dir,
                    shard_size=int(cfg.get("bench.shard_size", 256)),
                    base_seed=int(cfg.get("seed", 0)),
                    max_prompt_tokens=settings["max_model_len"] - sampling.max_new_tokens,
                    key_fields=job.fields,
                )
                status = "done"
            finally:
                # DC-14: one line per process and benchmark, also when it paused, so the
                # stage can sum seconds over sessions (`stage` → timing.json).
                timing["benches"][job.bench] = round(time.time() - t2, 1)
                _append_timing(job.out_dir, {
                    "bench_seconds": timing["benches"][job.bench],
                    "materialize_seconds": timing["materialize_seconds"],
                    "engine_seconds": timing.get("engine_seconds"),
                    "n_benches_this_process": len(pending),
                    "status": status,
                })  # fmt: skip
    finally:
        if temporary:
            shutil.rmtree(path, ignore_errors=True)
            timing["merged_removed"] = not Path(path).exists()
        timing["peak_rss_gib"] = _peak_rss_gib()
    return timing


def _peak_rss_gib() -> float:
    """This process's peak resident memory (Linux ru_maxrss is in KiB); B-033."""
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def _append_timing(out_dir: Path, row: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "timing.jsonl", "a") as fh:
        fh.write(json.dumps({**row, "unix": round(time.time())}) + "\n")


def read_timing(out_dir: Path) -> dict[str, Any]:
    """Seconds one (checkpoint, benchmark) took over every process that worked on it:
    its own generation time plus an equal share of each process's merge + engine start."""
    path = Path(out_dir) / "timing.jsonl"
    rows = [json.loads(line) for line in open(path)] if path.exists() else []
    share = sum(
        ((r["materialize_seconds"] or 0) + (r["engine_seconds"] or 0)) / r["n_benches_this_process"]
        for r in rows
    )
    gen_s = sum(r["bench_seconds"] for r in rows)
    return {
        "seconds": round(gen_s + share, 1),
        "generate_seconds": round(gen_s, 1),
        "overhead_seconds": round(share, 1),
        "processes": len(rows),
    }


# --------------------------------------------------------------------------- stage

# Replaced by CPU tests: `engine_factory` passed to `run_checkpoint` for in-process runs.
ENGINE_FACTORY: Callable[..., Any] | None = None


def _checkpoint_groups(jobs: Sequence[GenJob]) -> list[list[GenJob]]:
    """Jobs grouped by checkpoint (one generate-one process each), in plan order."""
    groups: dict[tuple[str, str, str], list[GenJob]] = {}
    for j in jobs:
        groups.setdefault((j.model_slug, j.spectrum, j.ckpt), []).append(j)
    return list(groups.values())


def load_context(cfg: Any, root: Path, items_dir: Path) -> tuple[dict, list[Any], list[GenJob]]:
    """(metas, spectra, jobs) as the parent planned them, read back from `items_dir`."""
    from rbbd.embed.extract import ftdata_outputs
    from rbbd.models import spectrum as spm

    saved = json.loads((Path(items_dir) / "metas.json").read_text())
    slugs = list(cfg.get("bench.checkpoints") or cfg.get("embed.checkpoints", spm.CHECKPOINTS))
    needs_train = any(s != spm.REF for s in slugs)
    spectra_list = spm.spectra(cfg, root, ftdata_outputs(cfg, root)) if needs_train else []
    jobs = plan_jobs(cfg, root, saved["benches"], spectra_list, saved["revisions"])
    return saved, spectra_list, jobs


def _launch(cfg: Any, items_dir: Path, group: Sequence[GenJob], gpus: str, log_path: Path):
    import subprocess
    import sys

    j = group[0]
    cmd = [
        sys.executable,
        "-m",
        "rbbd.cli",
        "generate-one",
        "--config",
        str(Path(cfg.path).resolve()),
        "--model",
        j.model_slug,
        "--spectrum",
        j.spectrum,
        "--ckpt",
        j.ckpt,
        "--items-dir",
        str(items_dir),
    ]
    for assignment in getattr(cfg, "overrides", ()):
        cmd += ["--set", assignment]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "a")  # noqa: SIM115 - closed by the caller
    log.info("generate %s/%s/%s on GPU %s", j.model_slug, j.spectrum, j.ckpt, gpus)
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf


def _sync(cfg: Any, root: Path, group: Sequence[GenJob]) -> None:
    """Upload one checkpoint's generation directories to the private store (D-037)."""
    from rbbd.utils.store import open_store

    try:
        store = open_store(cfg.get("store.repo_id"), cfg.get("store.repo_type", "auto"))
        for j in group:
            if j.out_dir.exists():
                rel = j.out_dir.relative_to(root).as_posix()
                store.upload_dir(j.out_dir, rel, f"generations {rel}")
    except Exception as exc:  # noqa: BLE001 - the end-of-session sync still applies
        log.warning("generation sync failed (%s: %s)", type(exc).__name__, exc)


def stage(ctx: Any) -> Any:
    """`generate` stage: every benchmark for every checkpoint (M5, D-050, D-086).

    Prompt sets are built once into ephemeral disk (`prepare_items`); each checkpoint
    with unfinished benchmarks runs in its own `generate-one` process: two at a time on
    cuda:0 / cuda:1 for TP = 1 models, one on both GPUs for TP = 2. A process that stops
    at the session deadline (exit 76) leaves the stage incomplete. Outputs are the
    `done.json` files plus `generate/<run_key>/{index.json, tags.json}` (no text).
    """
    from rbbd.embed.extract import resolve_revision
    from rbbd.models.spectrum import alpha_of
    from rbbd.runner import StageResult
    from rbbd.utils.env import ephemeral_dir

    cfg, root = ctx.cfg, Path(ctx.env.artifacts_root)
    benches = list(cfg.get("bench.benches", BENCHES))
    items_dir = ephemeral_dir() / "rbbd_items" / ctx.run_key
    if not (items_dir / "metas.json").exists():
        metas = prepare_items(cfg, items_dir, benches)
        revisions = {m["slug"]: resolve_revision(m["id"], m.get("revision")) for m in cfg["models"]}
        (items_dir / "metas.json").write_text(
            json.dumps({"benches": metas, "revisions": revisions})
        )
    saved, spectra_list, jobs = load_context(cfg, root, items_dir)
    by_name = {sp.name: sp for sp in spectra_list}
    groups = [g for g in _checkpoint_groups(jobs) if not all(j.done() for j in g)]
    paused: list[str] = []
    if not cfg.get("bench.parallel", True):
        for g in groups:
            run_checkpoint(cfg, root, g, items_dir, by_name.get(g[0].spectrum), ENGINE_FACTORY)
    else:
        logs = ctx.stage_dir / "logs"
        queue = list(groups)
        while queue:
            entry = next(m for m in cfg["models"] if m["slug"] == queue[0][0].model_slug)
            tp = engine_settings(cfg, entry)["tensor_parallel_size"]
            batch = [queue.pop(0)] if tp > 1 else [queue.pop(0) for _ in range(min(2, len(queue)))]
            gpus = ["0,1"] if tp > 1 else ["0", "1"]
            running = []
            for g, gpu in zip(batch, gpus, strict=False):
                lp = logs / f"{g[0].model_slug}_{g[0].spectrum}_{g[0].ckpt}.log"
                running.append((g, lp, *_launch(cfg, items_dir, g, gpu, lp)))
            failed = []
            for g, lp, proc, logf in running:
                code = proc.wait()
                logf.close()
                if code == PAUSE_EXIT:
                    paused.append(f"{g[0].model_slug}/{g[0].spectrum}/{g[0].ckpt}")
                elif code != 0:
                    tail = "\n".join(lp.read_text().splitlines()[-30:])
                    failed.append(f"{lp.name} (exit {code}):\n{tail}")
                if cfg.get("bench.sync_each", False):
                    _sync(cfg, root, g)
            if failed:
                raise RuntimeError("generate-one failed:\n" + "\n\n".join(failed))
            if paused:
                raise GenerationPaused(f"paused at the session deadline: {paused}")
    index = []
    for j in jobs:
        if not j.done():
            raise RuntimeError(f"generation incomplete after the stage: {j.out_dir}")
        sp = by_name.get(j.spectrum)
        index.append(
            {
                "model": j.model_slug,
                "spectrum": j.spectrum,
                "regime": sp.regime if sp else "base",
                "seed": sp.seed if sp else None,
                "ckpt": j.ckpt,
                "alpha": alpha_of(j.ckpt),
                "bench": j.bench,
                "key": j.key,
                "dir": j.out_dir.relative_to(root).as_posix(),
            }
        )
    (ctx.stage_dir / "index.json").write_text(json.dumps(index, indent=1))
    # DC-14: a log, not a hashed output (as embed's timing.json, D-081).
    timing = [{**{k: r[k] for k in ("model", "spectrum", "ckpt", "bench")},
               **read_timing(j.out_dir)} for r, j in zip(index, jobs, strict=True)]  # fmt: skip
    (ctx.stage_dir / "timing.json").write_text(json.dumps(timing, indent=1))
    tags = {
        b: {"source": m["source"], "items": m["items"], "tags": m["tags"]}
        for b, m in saved["benches"].items()
    }
    (ctx.stage_dir / "tags.json").write_text(json.dumps(tags))
    outputs = {
        f"{j.bench}/{j.model_slug}/{j.spectrum}/{j.ckpt}": str(
            (j.out_dir / "done.json").relative_to(root)
        )
        for j in jobs
    }
    outputs["index"] = str((ctx.stage_dir / "index.json").relative_to(root))
    outputs["tags"] = str((ctx.stage_dir / "tags.json").relative_to(root))
    return StageResult(outputs=outputs)
