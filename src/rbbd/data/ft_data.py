"""WildGuardMix fine-tuning splits (App. C; D-010, D-011, D-040) and the token census (D-042).

Selection (pure Python, CPU-testable):
- unharmful: prompt and response both labelled unharmful, non-empty response (D-011);
- harmful: prompt and response labelled harmful, response labelled compliance, non-empty
  response; existing WildGuardMix rows only, nothing generated (D-010).
Each split is a seeded sample of `n_per_split` row indices; if fewer rows qualify, all
are used and the shortfall is recorded.

The census tokenises each selected example with every configured model's chat template
(user turn + assistant turn, as in SFT, D-040) and reports the length distribution and
the share over 512 / 1,024 tokens. D-042's fallback (max length 512 only if <= 5% of
examples are truncated) is decided from these numbers, as is the M3 time budget.

Only row indices and aggregate statistics are written; no WildGuardMix text leaves the
HF cache (D-037). The `ftdata` stage (`stage`) runs on a Kaggle CPU session because the
dataset is gated.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

REQUIRED_COLUMNS = (
    "prompt",
    "response",
    "prompt_harm_label",
    "response_harm_label",
    "response_refusal_label",
    "subcategory",
)
SPLITS = ("unharmful", "harmful")


class SchemaError(ValueError):
    """The dataset is missing columns this module relies on (B-006)."""


def check_schema(columns: Sequence[str]) -> None:
    """Raise `SchemaError` naming every missing column."""
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise SchemaError(f"WildGuardMix columns missing: {missing}; found {sorted(columns)}")


def _has_response(row: Mapping[str, Any]) -> bool:
    return isinstance(row.get("response"), str) and bool(row["response"].strip())


def is_unharmful(row: Mapping[str, Any]) -> bool:
    """D-011: prompt and response unharmful, response present."""
    return (
        row.get("prompt_harm_label") == "unharmful"
        and row.get("response_harm_label") == "unharmful"
        and _has_response(row)
    )


def is_harmful(row: Mapping[str, Any]) -> bool:
    """D-010: harmful prompt answered with a harmful, compliant response."""
    return (
        row.get("prompt_harm_label") == "harmful"
        and row.get("response_harm_label") == "harmful"
        and row.get("response_refusal_label") == "compliance"
        and _has_response(row)
    )


FILTERS: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "unharmful": is_unharmful,
    "harmful": is_harmful,
}


def select_split(
    rows: Sequence[Mapping[str, Any]], split: str, n: int = 8000, seed: int = 0
) -> dict[str, Any]:
    """Seeded sample of up to `n` qualifying row indices for `split`, plus aggregate stats.

    Returns {"ids": sorted indices, "n_available": int, "n_selected": int,
    "subcategory_counts": {...}, "adversarial_share": float | None}.
    """
    keep = FILTERS[split]
    eligible = np.array([i for i, row in enumerate(rows) if keep(row)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.permutation(eligible)[:n]) if len(eligible) else eligible
    ids = [int(i) for i in chosen]
    subcats = Counter(str(rows[i].get("subcategory")) for i in ids)
    adversarial = [
        rows[i].get("adversarial") for i in ids if rows[i].get("adversarial") is not None
    ]
    return {
        "ids": ids,
        "n_available": int(len(eligible)),
        "n_selected": len(ids),
        "shortfall": max(0, n - len(ids)),
        "subcategory_counts": dict(sorted(subcats.items())),
        "adversarial_share": (float(np.mean(adversarial)) if adversarial else None),
    }


def to_messages(row: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    """TRL conversational prompt-completion record (loss on the completion only, D-040)."""
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "completion": [{"role": "assistant", "content": row["response"]}],
    }


def token_lengths(rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> np.ndarray:
    """Chat-template token count of each (user prompt, assistant response) pair -> [n] int."""
    lengths = []
    for row in rows:
        msgs = to_messages(row)
        ids = tokenizer.apply_chat_template(msgs["prompt"] + msgs["completion"], tokenize=True)
        lengths.append(len(ids))
    return np.asarray(lengths, dtype=np.int64)


def census(lengths: Sequence[int], max_length: int = 1024, epochs: int = 3) -> dict[str, float]:
    """Length distribution and truncation shares; `tokens_trained` = sum(min(len, max)) x epochs."""
    x = np.asarray(lengths, dtype=np.int64)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": int(x.max()),
        "frac_over_512": float((x > 512).mean()),
        "frac_over_1024": float((x > 1024).mean()),
        "tokens_trained_at_1024": int(np.minimum(x, max_length).sum() * epochs),
        "tokens_trained_at_512": int(np.minimum(x, 512).sum() * epochs),
    }


def load_wgm_train(dataset_id: str, config: str, revision: str | None) -> tuple[Any, str]:
    """Load the gated WildGuardMix train split (token from $HF_TOKEN); return (dataset, sha)."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(dataset_id, revision=revision).sha
    ds = load_dataset(dataset_id, config, split="train", revision=sha)
    return ds, sha


def stage(ctx: Any) -> Any:
    """`ftdata` stage: select both splits and run the token census (FLOW.md › Stage: ftdata).

    Writes `ftdata/<run_key>/{unharmful,harmful}_ids.json` and `stats.json`.
    """
    from transformers import AutoTokenizer

    from rbbd.runner import StageResult

    cfg = ctx.cfg.get("ftdata")
    ds, sha = load_wgm_train(cfg["dataset"], cfg["config"], cfg.get("revision"))
    check_schema(ds.column_names)
    rows = ds.to_list()
    stats: dict[str, Any] = {"dataset": cfg["dataset"], "config": cfg["config"], "revision": sha}
    outputs: dict[str, str] = {}
    selected: dict[str, list[int]] = {}
    for split in SPLITS:
        sel = select_split(rows, split, n=int(cfg["n_per_split"]), seed=int(ctx.cfg["seed"]))
        selected[split] = sel.pop("ids")
        stats[split] = sel
        path = ctx.stage_dir / f"{split}_ids.json"
        path.write_text(json.dumps(selected[split]))
        outputs[f"{split}_ids"] = str(path.relative_to(ctx.env.artifacts_root))
    stats["census"] = {}
    for model_id in cfg.get("census_tokenizers", []):
        tok = AutoTokenizer.from_pretrained(model_id)
        stats["census"][model_id] = {
            split: census(token_lengths([rows[i] for i in selected[split]], tok))
            for split in SPLITS
        }
    path = ctx.stage_dir / "stats.json"
    path.write_text(json.dumps(stats, indent=2))
    outputs["stats"] = str(path.relative_to(ctx.env.artifacts_root))
    return StageResult(outputs=outputs)
