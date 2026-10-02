"""Supervised fine-tuning of the two endpoints, u and h (App. C; D-003, D-009, D-040, D-055, D-065).

Call path (FLOW.md › Stage: train):

    runner -> stage(ctx)                    one job per (model, regime, seed, split)
           -> subprocess `cli train-one`    u on cuda:0 and h on cuda:1 in parallel (D-003)
           -> run_job(cfg, env, ...)        WGM rows by the ftdata ids, tokenizer, model by regime
           -> train_one(spec, rows, ...)    build_examples -> TRL SFTTrainer -> final/ + done.json

Data (D-040, D-055): each WildGuardMix row becomes a prompt-completion pair rendered with
the model's chat template. Tokenisation reproduces TRL 0.25.1's prompt-completion path
(prompt with generation prompt, then prompt + completion; loss on completion tokens
only). It is done here so that examples whose prompt alone fills `max_length` can be
counted and dropped instead of contributing an empty target (D-055 item 4). Longer
examples are truncated from the right (prompt head kept, D-040).

Guards (D-006 as refined by D-065): a non-finite loss always aborts. A non-finite
grad-norm is tolerated only on a step the fp16 gradient scaler skipped. Too many
consecutive skips, or too high a skip fraction in a long run, also abort.

Checkpoints (B-007): a trainer checkpoint is written every `save_minutes` of wall-clock
time, and `train_one` resumes from the newest one. Only the newest is kept
(`save_total_limit=1`). `final/` holds the PEFT adapter (LoRA/QLoRA, fp32 tensors) or
the fp16 full model (Tier 2 full FT, D-008).

Job directory: `<artifacts>/train/<slug>/<regime>/<split>/seed<k>/<train_key>/`.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rbbd.utils.cache import make_key
from rbbd.utils.guards import NonFiniteError, assert_finite_scalar
from rbbd.utils.logging import get_logger

log = get_logger(__name__)

REGIMES = ("lora", "qlora", "full")
SPLITS = ("unharmful", "harmful")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEFAULT_LR = {"lora": 1e-4, "qlora": 1e-4, "full": 2e-5}  # App. C
# App. C says AdamW. Full FT uses the 8-bit paged variant to fit fp32 masters on a T4 (D-009).
DEFAULT_OPTIM = {"lora": "adamw_torch", "qlora": "adamw_torch", "full": "paged_adamw_8bit"}
# 2: LoRA init seeded from the spec inside train_one (B-017), so same spec -> same weights.
SCHEMA_VERSION = 2


class DataBuildError(ValueError):
    """Chat-template tokenisation broke the prompt-prefix property TRL relies on."""


class SimulatedKill(RuntimeError):
    """Raised by `KillAfterSave` to imitate a session dying right after a checkpoint (DC-13)."""


@dataclass(frozen=True)
class TrainSpec:
    """Everything that determines one endpoint's weights; its `train_key` names the job dir.

    `data_key` hashes the dataset revision and the selected row ids. `compute` is the
    autocast precision ("fp16" on T4, or "fp32" for D-005's Gemma fallback).
    """

    model_id: str
    slug: str
    regime: str
    split: str
    seed: int
    data_key: str
    revision: str | None = None
    compute: str = "fp16"
    epochs: float = 3
    per_device_batch: int = 4
    grad_accum: int = 8
    lr: float | None = None
    warmup_ratio: float = 0.03
    lr_scheduler: str = "linear"
    weight_decay: float = 0.0  # [unspecified in paper]; TrainingArguments default
    max_grad_norm: float = 1.0  # [unspecified in paper]; TrainingArguments default
    max_length: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = LORA_TARGETS
    # Optional PEFT regex (full match) replacing `lora_targets`, e.g. to keep LoRA off
    # Gemma-3's vision tower (D-076). None = the module-name list above.
    lora_target_regex: str | None = None
    optim: str | None = None
    gradient_checkpointing: bool = True
    max_steps: int = -1  # tests only; -1 = run the epochs

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {self.regime!r}")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        if self.compute not in ("fp16", "fp32"):
            raise ValueError(f"compute must be fp16 or fp32, got {self.compute!r}")

    @property
    def learning_rate(self) -> float:
        return self.lr if self.lr is not None else DEFAULT_LR[self.regime]

    @property
    def optimizer(self) -> str:
        return self.optim or DEFAULT_OPTIM[self.regime]

    def key_fields(self) -> dict[str, Any]:
        """Fields hashed into `train_key` (ARCHITECTURE › cache keys)."""
        fields = asdict(self)
        fields["lora_targets"] = list(self.lora_targets)
        fields.update(lr=self.learning_rate, optim=self.optimizer, schema_version=SCHEMA_VERSION)
        if self.lora_target_regex is None:  # absent from keys made before D-076
            fields.pop("lora_target_regex", None)
        if self.regime == "full":  # LoRA fields do not affect a full fine-tune
            for k in ("lora_r", "lora_alpha", "lora_dropout", "lora_targets", "lora_target_regex"):
                fields.pop(k, None)
        return fields

    @property
    def train_key(self) -> str:
        return make_key(self.key_fields())


@dataclass(frozen=True)
class GuardSettings:
    """D-065 limits on fp16 gradient-scaler skips."""

    max_consecutive_skips: int = 25
    max_skip_fraction: float = 0.05
    min_steps_for_fraction: int = 100


def job_dir(artifacts_root: Path, spec: TrainSpec) -> Path:
    """`<root>/train/<slug>/<regime>/<split>/seed<k>/<train_key>/`."""
    return (
        Path(artifacts_root)
        / "train"
        / spec.slug
        / spec.regime
        / spec.split
        / f"seed{spec.seed}"
        / spec.train_key
    )


def data_key(revision: str | None, split: str, ids: Sequence[int]) -> str:
    """Content key of one training split: dataset revision + split + selected row ids."""
    return make_key({"revision": revision, "split": split, "ids": [int(i) for i in ids]})


def build_examples(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int
) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    """Tokenise prompt-completion pairs for SFT (D-040) and drop prompt-only ones (D-055).

    Returns (examples, stats). Each example is {"input_ids": [L], "completion_mask": [L]}
    with L <= max_length; completion_mask is 1 on assistant tokens. stats counts rows
    in, kept, dropped because the prompt fills the window, truncated, and tokens kept.
    Raises `DataBuildError` if the prompt tokens are not a prefix of the full sequence.
    """
    from rbbd.data.ft_data import to_messages

    examples: list[dict[str, list[int]]] = []
    stats = {
        "n_rows": len(rows),
        "n_kept": 0,
        "n_dropped_prompt_fills_window": 0,
        "n_truncated": 0,
        "n_tokens": 0,
        "n_completion_tokens": 0,
    }
    for i, row in enumerate(rows):
        msgs = to_messages(row)
        prompt_ids = list(
            tokenizer.apply_chat_template(msgs["prompt"], tokenize=True, add_generation_prompt=True)
        )
        full_ids = list(
            tokenizer.apply_chat_template(msgs["prompt"] + msgs["completion"], tokenize=True)
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise DataBuildError(
                f"row {i}: chat-template prompt is not a prefix of prompt+completion"
            )
        mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
        if len(full_ids) > max_length:
            stats["n_truncated"] += 1
            full_ids, mask = full_ids[:max_length], mask[:max_length]
        if sum(mask) == 0:
            stats["n_dropped_prompt_fills_window"] += 1
            continue
        examples.append({"input_ids": full_ids, "completion_mask": mask})
        stats["n_tokens"] += len(full_ids)
        stats["n_completion_tokens"] += sum(mask)
    stats["n_kept"] = len(examples)
    return examples, stats


def _callback_base() -> type:
    from transformers import TrainerCallback

    return TrainerCallback


class GuardCallback(_callback_base()):  # type: ignore[misc]
    """D-065 training guard, checked on every logged step (`logging_steps=1`).

    `on_step_end` reads whether the gradient scaler skipped the optimizer step (via the
    trainer's accelerator, set as `.trainer` after the trainer is built). `on_log`
    then checks the logged loss and grad-norm. `summary()` feeds `done.json`.
    """

    def __init__(self, settings: GuardSettings, context: Mapping[str, Any]) -> None:
        self.settings = settings
        self.context = dict(context)
        self.trainer: Any = None
        self.steps = 0
        self.skipped = 0
        self.consecutive = 0
        self._last_skipped = False

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        accel = getattr(self.trainer, "accelerator", None)
        self._last_skipped = bool(getattr(accel, "optimizer_step_was_skipped", False))
        self.steps += 1
        if self._last_skipped:
            self.skipped += 1
            self.consecutive += 1
        else:
            self.consecutive = 0
        self.check_counts(state.global_step)

    def check_counts(self, step: int) -> None:
        """Abort on a collapsed loss scale, or on too many skips once the run is long enough.

        The fraction rule waits for `min_steps_for_fraction` steps: early fp16 steps
        often skip while the scaler finds its scale, and a short smoke run would
        otherwise fail on that alone (D-065).
        """
        s = self.settings
        if self.consecutive > s.max_consecutive_skips:
            raise NonFiniteError(
                f"{self.consecutive} consecutive fp16 scaler-skipped steps at step {step} "
                f"(limit {s.max_consecutive_skips}; D-065) ({self.context})"
            )
        if self.steps >= s.min_steps_for_fraction:
            frac = self.skipped / self.steps
            if frac > s.max_skip_fraction:
                raise NonFiniteError(
                    f"{frac:.1%} of {self.steps} steps skipped by the fp16 scaler "
                    f"(limit {s.max_skip_fraction:.0%}; D-065) ({self.context})"
                )

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
        logs = logs or {}
        ctx = {**self.context, "step": state.global_step}
        if "loss" in logs:
            assert_finite_scalar(logs["loss"], "train.loss", **ctx)
        if "grad_norm" in logs and not math.isfinite(float(logs["grad_norm"])):
            if not self._last_skipped:
                raise NonFiniteError(
                    f"non-finite grad_norm {logs['grad_norm']} on an applied step ({ctx})"
                )

    def summary(self) -> dict[str, Any]:
        return {"optimizer_steps_seen": self.steps, "scaler_skipped_steps": self.skipped}


class WallClockSaveCallback(_callback_base()):  # type: ignore[misc]
    """Request a trainer checkpoint every `minutes` of wall-clock time (B-007)."""

    def __init__(self, minutes: float) -> None:
        self.seconds = minutes * 60.0
        self.last = time.time()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if time.time() - self.last >= self.seconds:
            control.should_save = True
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.last = time.time()


class ThroughputCallback(_callback_base()):  # type: ignore[misc]
    """Steady-state training throughput from TRL's cumulative `num_tokens` log (M3c-T8).

    `num_tokens` counts attention-mask (non-pad) tokens, as `data.json`'s `n_tokens`
    does, so a projection from the two is consistent. The first `skip_steps` logged
    steps are excluded: `group_by_length` puts the longest batch first, and CUDA/kernel
    warm-up inflates the first steps.
    """

    def __init__(self, skip_steps: int = 3) -> None:
        self.skip_steps = skip_steps
        self.clock = time.time  # replaceable in tests
        self.points: list[tuple[int, float, float]] = []  # (step, wall time, num_tokens)

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
        if logs and "num_tokens" in logs:
            self.points.append((int(state.global_step), self.clock(), float(logs["num_tokens"])))

    def summary(self) -> dict[str, Any]:
        steady = [p for p in self.points if p[0] > self.skip_steps]
        out: dict[str, Any] = {"tokens_per_second_steady": None, "seconds_per_step_steady": None}
        if len(steady) >= 2:
            (s0, t0, n0), (s1, t1, n1) = steady[0], steady[-1]
            if t1 > t0:
                out["tokens_per_second_steady"] = round((n1 - n0) / (t1 - t0), 1)
                out["seconds_per_step_steady"] = round((t1 - t0) / (s1 - s0), 2)
        return out


class ActivationMonitor:
    """Final-layer hidden-state monitor for fp16 training probes (D-005, D-006, D-076).

    A forward hook on the text decoder's last layer (`base_decoder(model).layers[-1]`,
    the pre-norm residual stream, where fp16 overflows: Gemma's final norm can hide a
    near-overflow) records the largest |hidden| seen and raises `NonFiniteError` on the
    first NaN/Inf. An overflow in any earlier layer propagates to this output as Inf/NaN.
    So fp16 trouble stops training at once instead of surfacing later as a NaN loss.
    `attach(model)` must run before the trainer wraps the model (PEFT keeps
    the same module objects). Cost: two reductions per forward.
    """

    def __init__(self, context: Mapping[str, Any]) -> None:
        self.context = dict(context)
        self.max_abs = 0.0
        self.calls = 0
        self._handle: Any = None

    def attach(self, model: Any) -> None:
        from rbbd.models.loading import base_decoder

        self._handle = base_decoder(model).layers[-1].register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        from rbbd.utils.guards import assert_finite

        hidden = (
            output[0] if isinstance(output, tuple) else getattr(output, "last_hidden_state", output)
        )
        self.calls += 1
        assert_finite(hidden.detach(), "train.hidden", call=self.calls, **self.context)
        self.max_abs = max(self.max_abs, float(hidden.detach().abs().max()))

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def summary(self) -> dict[str, Any]:
        return {"max_abs_hidden": round(self.max_abs, 3), "monitored_forwards": self.calls}


class TimeLimitCallback(_callback_base()):  # type: ignore[misc]
    """Stop training cleanly after `minutes` of wall-clock time (throughput probes only)."""

    def __init__(self, minutes: float) -> None:
        self.deadline = time.time() + minutes * 60.0

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if time.time() >= self.deadline:
            control.should_training_stop = True
        return control


def project_hours(n_tokens: int, epochs: float, tokens_per_second: float | None) -> float | None:
    """Projected training hours for `epochs` passes over `n_tokens` at a measured rate."""
    if not tokens_per_second:
        return None
    return round(n_tokens * epochs / tokens_per_second / 3600.0, 2)


class KillAfterSave(_callback_base()):  # type: ignore[misc]
    """Test hook (DC-13): raise `SimulatedKill` right after the checkpoint at `step`."""

    def __init__(self, step: int) -> None:
        self.step = step

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step >= self.step:
            raise SimulatedKill(f"simulated kill after checkpoint {state.global_step}")


def _require_single_device() -> None:
    """Refuse to train with more than one visible GPU (D-003, B-017).

    With `device_map={"": 0}` and two visible GPUs, the HF Trainer wraps the model in
    DataParallel and doubles the effective batch, silently changing App. C's
    hyperparameters. The train stage gives each job one GPU via CUDA_VISIBLE_DEVICES.
    """
    import torch

    n = torch.cuda.device_count()
    if n > 1:
        raise RuntimeError(
            f"train_one sees {n} GPUs; run it with exactly one visible "
            "(CUDA_VISIBLE_DEVICES=<i>), as the train stage does (D-003)"
        )


def latest_checkpoint(ckpt_dir: Path) -> str | None:
    """Newest `checkpoint-N` under `ckpt_dir`, or None."""
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(ckpt_dir)) if Path(ckpt_dir).is_dir() else None


def _peft_config(spec: TrainSpec) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=spec.lora_target_regex or list(spec.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )


def sft_config(spec: TrainSpec, ckpt_dir: Path, save_steps: int | None = None) -> Any:
    """TRL `SFTConfig` with App. C hyperparameters (D-040: completion-only loss, by length).

    Without `save_steps`, checkpoints come only from `WallClockSaveCallback`.
    """
    import torch
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=spec.epochs,
        max_steps=spec.max_steps,
        per_device_train_batch_size=spec.per_device_batch,
        gradient_accumulation_steps=spec.grad_accum,
        learning_rate=spec.learning_rate,
        lr_scheduler_type=spec.lr_scheduler,
        warmup_ratio=spec.warmup_ratio,
        weight_decay=spec.weight_decay,
        max_grad_norm=spec.max_grad_norm,
        optim=spec.optimizer,
        fp16=spec.compute == "fp16",
        bf16=False,  # T4 has no bf16 (D-002)
        gradient_checkpointing=spec.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=spec.max_length,
        completion_only_loss=True,  # pre-tokenised data has no "prompt" column to infer it from
        group_by_length=True,
        seed=spec.seed,
        data_seed=spec.seed,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps or 10**9,
        save_total_limit=1,
        report_to=[],
        dataloader_num_workers=0,
        use_cpu=not torch.cuda.is_available(),
        disable_tqdm=True,
    )


def train_one(
    spec: TrainSpec,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    model: Any,
    out_dir: Path,
    *,
    save_minutes: float = 20.0,
    save_steps: int | None = None,
    guard: GuardSettings | None = None,
    extra_callbacks: Sequence[Any] = (),
    max_minutes: float | None = None,
    ckpt_dir: Path | None = None,
    activation_monitor: bool = False,
) -> dict[str, Any]:
    """Train one endpoint into `out_dir`, resuming from its newest checkpoint; return done.json.

    `model` is already loaded for `spec.regime` (see `load_model`): fp16 base for LoRA, NF4
    for QLoRA, fp32 masters for full FT. LoRA/QLoRA get the adapter from TRL. Writes
    `out_dir/{data.json, final/, train_log.json, done.json}`. If `done.json` exists, it is
    returned and nothing runs.
    """
    from datasets import Dataset
    from trl import SFTTrainer

    out_dir = Path(out_dir)
    done_path = out_dir / "done.json"
    if done_path.exists():
        return json.loads(done_path.read_text())
    _require_single_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    examples, data_stats = build_examples(rows, tokenizer, spec.max_length)
    (out_dir / "data.json").write_text(
        json.dumps({"spec": spec.key_fields(), **data_stats}, indent=2)
    )
    if not examples:
        raise ValueError(f"no trainable examples for {spec.slug}/{spec.regime}/{spec.split}")
    log.info(
        "%s/%s/%s: %d examples kept, %d dropped (prompt fills %d tokens), %d truncated",
        spec.slug,
        spec.regime,
        spec.split,
        data_stats["n_kept"],
        data_stats["n_dropped_prompt_fills_window"],
        spec.max_length,
        data_stats["n_truncated"],
    )

    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else out_dir / "checkpoints"
    context = {"model": spec.slug, "regime": spec.regime, "split": spec.split, "seed": spec.seed}
    guard_cb = GuardCallback(guard or GuardSettings(), context)
    throughput = ThroughputCallback()
    monitor = ActivationMonitor(context) if activation_monitor else None
    if monitor is not None:
        monitor.attach(model)
    time_limit = [TimeLimitCallback(max_minutes)] if max_minutes is not None else []
    # The LoRA A matrices are initialised inside SFTTrainer (get_peft_model): seed here so
    # the initial adapter depends on the spec's seed only, not on what ran before (B-017).
    from transformers import set_seed

    set_seed(spec.seed)
    trainer = SFTTrainer(
        model=model,
        args=sft_config(spec, ckpt_dir, save_steps),
        train_dataset=Dataset.from_list(examples),
        processing_class=tokenizer,
        peft_config=_peft_config(spec) if spec.regime in ("lora", "qlora") else None,
        callbacks=[
            guard_cb,
            WallClockSaveCallback(save_minutes),
            throughput,
            *time_limit,
            *extra_callbacks,
        ],
    )
    guard_cb.trainer = trainer
    resume = latest_checkpoint(ckpt_dir)
    if resume:
        log.info("resuming %s from %s", out_dir, resume)
    t0 = time.time()
    try:
        result = trainer.train(resume_from_checkpoint=resume)
    finally:
        if monitor is not None:
            monitor.detach()
    seconds = time.time() - t0
    guard_cb.check_counts(trainer.state.global_step)

    final = out_dir / "final"
    if spec.regime == "full":
        import torch

        trainer.model.to(torch.float16).save_pretrained(str(final), safe_serialization=True)
        tokenizer.save_pretrained(str(final))
    else:
        trainer.model.save_pretrained(str(final))
    history = trainer.state.log_history
    (out_dir / "train_log.json").write_text(json.dumps(history, indent=2))
    epochs_done = float(trainer.state.epoch or 0)
    done = {
        "train_key": spec.train_key,
        "global_step": int(trainer.state.global_step),
        "epochs": epochs_done,
        "train_loss": float(result.training_loss),
        "final_loss": next((h["loss"] for h in reversed(history) if "loss" in h), None),
        "resumed_from": Path(resume).name if resume else None,
        "seconds_this_process": round(seconds, 1),
        # Throughput for the M3c-T8 probe: tokens processed per second in this process.
        "tokens_per_second": round(data_stats["n_tokens"] * epochs_done / seconds, 1)
        if seconds > 0 and not resume
        else None,
        "n_examples": data_stats["n_kept"],
        **guard_cb.summary(),
        **throughput.summary(),
        **(monitor.summary() if monitor is not None else {}),
        "peak_mem_gib": _peak_mem_gib(),
    }
    # Full-run projection from this split's exact token count (M3c-T8; D-042, D-055).
    tps = done["tokens_per_second_steady"]
    done["projected_hours"] = {
        f"{e:g}_epochs": project_hours(data_stats["n_tokens"], e, tps) for e in (spec.epochs, 1)
    }
    done_path.write_text(json.dumps(done, indent=2))
    return done


def _peak_mem_gib() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated(0) / 2**30, 2)


def load_model(spec: TrainSpec) -> Any:
    """Load the base for `spec.regime` on the single visible GPU (D-003, D-009).

    lora: fp16 weights (fp32 if compute is fp32), LoRA weights fp32 via PEFT autocast;
    qlora: NF4 + double quantisation, compute dtype = spec.compute;
    full: fp32 master weights, fp16 autocast by the trainer.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute = torch.float16 if spec.compute == "fp16" else torch.float32
    kwargs: dict[str, Any] = {"revision": spec.revision, "device_map": {"": 0}}
    if spec.regime == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute,
        )
        kwargs["dtype"] = compute
    elif spec.regime == "lora":
        kwargs["dtype"] = compute
    else:
        kwargs["dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model


def spec_from_config(
    cfg: Any,
    model_entry: Mapping[str, Any],
    regime: str,
    split: str,
    seed: int,
    dkey: str,
    layout: tuple[int, int] | None = None,
) -> TrainSpec:
    """Build a `TrainSpec` from the `train:` config section and one `models:` entry.

    A model entry may set `train_compute` (D-005 Gemma fallback) and `revision`.
    """
    t = cfg.get("train", {})
    lora = t.get("lora", {})
    lr = t.get("lr", {})
    return TrainSpec(
        model_id=model_entry["id"],
        slug=model_entry["slug"],
        revision=model_entry.get("revision"),
        regime=regime,
        split=split,
        seed=int(seed),
        data_key=dkey,
        compute=model_entry.get("train_compute", t.get("compute", "fp16")),
        epochs=float(t.get("epochs", 3)),
        per_device_batch=int(layout[0] if layout else t.get("per_device_batch", 4)),
        grad_accum=int(layout[1] if layout else t.get("grad_accum", 8)),
        lr=lr.get(regime) if isinstance(lr, Mapping) else lr,
        warmup_ratio=float(t.get("warmup_ratio", 0.03)),
        lr_scheduler=t.get("lr_scheduler", "linear"),
        max_length=int(t.get("max_length", 1024)),
        lora_r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        lora_targets=tuple(lora.get("targets", LORA_TARGETS)),
        lora_target_regex=model_entry.get("lora_target_regex"),
        optim=(t.get("optim") or {}).get(regime) if isinstance(t.get("optim"), Mapping) else None,
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        max_steps=int(t.get("max_steps", -1)),
    )


def batch_layouts(train_cfg: Mapping[str, Any], regime: str) -> list[tuple[int, int]]:
    """(per-device batch, grad-acc) layouts to try in order on CUDA OOM (D-072).

    `train.batch_layouts` may be a list of [b, g] pairs or a {regime: list} mapping;
    without it, the single layout (`per_device_batch`, `grad_accum`) is used.
    """
    spec = train_cfg.get("batch_layouts")
    if isinstance(spec, Mapping):
        spec = spec.get(regime)
    if not spec:
        return [(int(train_cfg.get("per_device_batch", 4)), int(train_cfg.get("grad_accum", 8)))]
    return [(int(b), int(g)) for b, g in spec]


def seeds_for(cfg: Any, regime: str) -> list[int]:
    """Training seeds for `regime`: `train.seeds_by_regime[regime]`, else `train.seeds`."""
    t = cfg.get("train", {})
    by_regime = t.get("seeds_by_regime") or {}
    seeds = by_regime.get(regime, t.get("seeds", [cfg.get("seed", 0)]))
    return [int(x) for x in seeds]


OOM_MARKER = "oom.json"


@dataclass
class Job:
    """One endpoint to train: its inputs and its candidate (spec, dir) per batch layout.

    `spec`/`out_dir` resolve to the active candidate: the first with `done.json`, else the
    first not marked `oom.json` (D-072), else the last.
    """

    model_entry: dict[str, Any]
    regime: str
    split: str
    seed: int
    candidates: list[tuple[TrainSpec, Path]]
    ids_path: Path
    revision: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    def _active(self) -> tuple[TrainSpec, Path]:
        for spec, d in self.candidates:
            if (d / "done.json").exists():
                return spec, d
        for spec, d in self.candidates:
            if not (d / OOM_MARKER).exists():
                return spec, d
        return self.candidates[-1]

    @property
    def spec(self) -> TrainSpec:
        return self._active()[0]

    @property
    def out_dir(self) -> Path:
        return self._active()[1]

    @property
    def log_path(self) -> Path:
        """One log per job across layouts: `<first layout dir>/train.log` (B-021)."""
        return self.candidates[0][1] / "train.log"

    def has_untried_layout(self) -> bool:
        """True while some layout is neither finished nor marked OOM."""
        return any(
            not (d / "done.json").exists() and not (d / OOM_MARKER).exists()
            for _, d in self.candidates
        )


def plan_jobs(cfg: Any, artifacts_root: Path, ftdata_outputs: Sequence[str]) -> list[Job]:
    """Every (model, seed, regime, split) job under `cfg`, u/h pairs adjacent.

    Order is seed-major (all regimes at seed 0 before any extra seed), so the D-042 cut
    order (extra seeds first) can stop a session early without losing a core endpoint.
    `ftdata_outputs` are the upstream manifest's output paths (relative to the root);
    `stats.json` gives the dataset revision and `<split>_ids.json` the selected rows.
    """
    root = Path(artifacts_root)
    by_name = {Path(p).name: root / p for p in ftdata_outputs}
    stats = json.loads(by_name["stats.json"].read_text())
    revision = stats.get("revision")
    t = cfg.get("train", {})
    regimes = list(t.get("regimes", ["lora"]))
    all_seeds = sorted({s for r in regimes for s in seeds_for(cfg, r)})
    ids = {split: json.loads(by_name[f"{split}_ids.json"].read_text()) for split in SPLITS}
    jobs = []
    for entry in cfg.get("models", []):
        for seed in all_seeds:
            for regime in regimes:
                if seed not in seeds_for(cfg, regime):
                    continue
                for split in SPLITS:
                    dkey = data_key(revision, split, ids[split])
                    candidates = []
                    for layout in batch_layouts(t, regime):
                        spec = spec_from_config(cfg, entry, regime, split, seed, dkey, layout)
                        candidates.append((spec, job_dir(root, spec)))
                    jobs.append(
                        Job(dict(entry), regime, split, int(seed), candidates,
                            by_name[f"{split}_ids.json"], revision)
                    )  # fmt: skip
    return jobs


def load_rows(cfg: Any, ids_path: Path, revision: str | None) -> list[dict[str, Any]]:
    """WildGuardMix rows for one split, read through the HF cache (text never leaves it, D-037)."""
    from rbbd.data.ft_data import load_wgm_train

    fcfg = cfg.get("ftdata")
    ds, _ = load_wgm_train(fcfg["dataset"], fcfg["config"], revision)
    ids = json.loads(Path(ids_path).read_text())
    sub = ds.select(ids)
    return [{"prompt": r["prompt"], "response": r["response"]} for r in sub]


def checkpoint_dir(cfg: Any, spec: TrainSpec, out_dir: Path) -> Path:
    """Where trainer checkpoints go: `<job>/checkpoints` (default, synced, survives the
    session) or, with `train.checkpoint_root: {<regime>: ephemeral}`, the ephemeral disk.

    Full-FT checkpoints of 1B models hold fp32 weights plus optimizer state (≈ 7.5 GB per
    job), so two parallel jobs would overflow /kaggle/working (D-072).
    """
    from rbbd.utils.env import ephemeral_dir

    roots = cfg.get("train.checkpoint_root") or {}
    if roots.get(spec.regime) == "ephemeral":
        return ephemeral_dir() / "rbbd_ckpt" / spec.slug / spec.train_key
    return Path(out_dir) / "checkpoints"


def _is_oom(exc: BaseException) -> bool:
    import torch

    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


OOM_EXIT = 75  # train-one exit code: this layout ran out of memory; relaunch for the next


class LayoutOOM(RuntimeError):
    """A batch layout ran out of CUDA memory; `oom.json` has been written (B-021)."""


def run_job(cfg: Any, job: Job, single_attempt: bool = False) -> dict[str, Any]:
    """Load data, tokenizer and model for `job` and train it (`cli train-one`).

    Batch layouts are tried in order; a CUDA OOM writes `oom.json` into that layout's
    directory (D-072). With `single_attempt` (the `train-one` subprocess path) the OOM
    raises `LayoutOOM` instead of retrying here: an in-process retry cannot reliably free
    the failed trainer's GPU memory (the exception's traceback keeps it alive, B-021), so
    the parent relaunches a fresh process for the next layout. In-process retries remain
    for CPU tests and inline runs.
    """
    import gc

    import torch

    from rbbd.models.loading import load_tokenizer

    rows = load_rows(cfg, job.ids_path, job.revision)
    t = cfg.get("train", {})
    guard = GuardSettings(**t.get("guard", {}))
    last_error: BaseException | None = None
    for spec, out_dir in job.candidates:
        if (out_dir / "done.json").exists():
            return json.loads((out_dir / "done.json").read_text())
        if (out_dir / OOM_MARKER).exists():
            continue
        tokenizer = load_tokenizer(spec.model_id, spec.revision)
        model = load_model(spec)
        try:
            return train_one(
                spec,
                rows,
                tokenizer,
                model,
                out_dir,
                save_minutes=float(t.get("save_minutes", 20)),
                guard=guard,
                max_minutes=t.get("max_minutes"),
                ckpt_dir=checkpoint_dir(cfg, spec, out_dir),
                activation_monitor=bool(t.get("activation_monitor", False)),
            )
        except Exception as exc:
            if not _is_oom(exc):
                raise
            last_error = exc
            out_dir.mkdir(parents=True, exist_ok=True)
            layout = [spec.per_device_batch, spec.grad_accum]
            (out_dir / OOM_MARKER).write_text(
                json.dumps({"layout": layout, "error": str(exc)[:500]})
            )
            log.warning("CUDA OOM at batch %d x %d; trying the next layout", *layout)
            if single_attempt:
                raise LayoutOOM(f"batch {layout[0]} x {layout[1]} ran out of memory") from None
            del model
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f"every batch layout ran out of memory for {job.split}") from last_error


def stage(ctx: Any) -> Any:
    """`train` stage: every job of `plan_jobs`, u‖h as two processes on cuda:0/cuda:1 (D-003).

    Each job runs `python -m rbbd.cli train-one`, which skips itself on an existing
    `done.json`, so a re-run after a dead session resumes per job. With
    `train.parallel: false` (CPU tests) jobs run in this process, in order.
    """
    from rbbd.runner import StageResult

    root = ctx.env.artifacts_root
    jobs = plan_jobs(ctx.cfg, root, list(ctx.upstream["ftdata"].output_hashes))
    if ctx.cfg.get("train.parallel", True):
        _run_parallel(ctx, jobs)
    else:
        for job in jobs:
            run_job(ctx.cfg, job)
    outputs: dict[str, str] = {}
    for job in jobs:
        rel = job.out_dir.relative_to(root)
        name = f"{job.spec.slug}/{job.regime}/{job.split}/seed{job.seed}"
        outputs[f"{name}/final"] = str(rel / "final")
        outputs[f"{name}/done"] = str(rel / "done.json")
    ctx.checkpoint({"done": sorted(outputs)})
    return StageResult(outputs=outputs)


def sync_jobs(cfg: Any, artifacts_root: Path, jobs: Sequence[Job]) -> None:
    """Upload finished job directories to the private store right after they finish (D-072).

    A long Tier 2 session that dies at the 12 h limit then loses at most the pair in
    progress; the next session restores them with `cli restore`. Upload failures are
    logged, not raised: training results stay on local disk for the end-of-session sync.
    """
    from rbbd.utils.store import open_store

    try:
        store = open_store(cfg.get("store.repo_id"), cfg.get("store.repo_type", "auto"))
        for job in jobs:
            if (job.out_dir / "done.json").exists():
                rel = job.out_dir.relative_to(artifacts_root).as_posix()
                store.upload_dir(job.out_dir, rel, f"train {rel}")
    except Exception as exc:  # noqa: BLE001 - never lose a finished run to a sync error
        log.warning("per-pair sync failed (%s: %s); the end-of-session sync still applies",
                    type(exc).__name__, exc)  # fmt: skip


def _launch(ctx: Any, job: Job, gpu: int) -> tuple[Any, Any]:
    """Start `cli train-one` for `job` on `gpu`; returns (process, open log file).

    The log is `job.log_path` (one file per job, appended per attempt), so a failure can
    always be reported, whichever layout directory the job resolves to (B-021).
    """
    import os
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "rbbd.cli", "train-one",
           "--config", str(Path(ctx.cfg.path).resolve()), "--model", job.spec.slug,
           "--regime", job.regime, "--split", job.split, "--seed", str(job.seed)]  # fmt: skip
    for assignment in getattr(ctx.cfg, "overrides", ()):  # same effective config as the parent
        cmd += ["--set", assignment]
    # expandable_segments reduces fragmentation-driven OOMs; it does not change results.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # fmt: skip
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(job.log_path, "a")  # noqa: SIM115 - closed by the caller
    layout = (job.spec.per_device_batch, job.spec.grad_accum)
    logf.write(f"\n=== launch: batch {layout[0]} x {layout[1]} on cuda:{gpu} ===\n")
    logf.flush()
    log.info("launching %s (batch %d x %d) on cuda:%d -> %s", job.split, *layout, gpu, job.out_dir)
    return subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf


def _tail(path: Path, n: int = 40) -> str:
    return "\n".join(path.read_text().splitlines()[-n:]) if path.exists() else f"(no log at {path})"


def _run_parallel(
    ctx: Any, jobs: Sequence[Job], launch: Any = _launch, poll_seconds: float = 5.0
) -> None:
    """Run jobs in u/h pairs, one process per GPU; the next pair starts when both finish.

    A process that exits with `OOM_EXIT` is relaunched on the same GPU for the job's next
    batch layout (fresh CUDA context, B-021) until a layout fits or none is left. Any
    other non-zero exit fails the stage with the job's log tail.
    """
    import time as _time

    pending = [j for j in jobs if not (j.out_dir / "done.json").exists()]
    for i in range(0, len(pending), 2):
        pair = pending[i : i + 2]
        running = {gpu: (job, *launch(ctx, job, gpu)) for gpu, job in enumerate(pair)}
        failed = []
        while running:
            for gpu, (job, proc, logf) in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                logf.close()
                del running[gpu]
                if code == OOM_EXIT and job.has_untried_layout():
                    running[gpu] = (job, *launch(ctx, job, gpu))
                elif code != 0:
                    failed.append(f"{job.out_dir} (exit {code}):\n{_tail(job.log_path)}")
            if running:
                _time.sleep(poll_seconds)
        if failed:
            raise RuntimeError("train-one failed:\n" + "\n\n".join(failed))
        if ctx.cfg.get("train.sync_each_pair", False):
            sync_jobs(ctx.cfg, ctx.env.artifacts_root, pair)
