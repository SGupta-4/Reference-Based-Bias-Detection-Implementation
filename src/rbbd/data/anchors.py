"""Word / Alpaca / Tulu anchor pools and nested anchor-size subsets for the F4 ablation
(§4.5; D-015, D-081).

The `sentences` stage calls `build_pools` once per run, after the authored sets have been
validated, and adds the pools to the sentence union as `anchors/<source>`. Every model
then embeds them in the same single extraction pass as the neutral anchors (D-020), so
M6's anchor-source × size ablation reads cached embeddings only.

Pools (each `size` sentences, default 1,000):
- `alpaca`: instructions with an empty `input` from `tatsu-lab/alpaca`;
- `tulu`: first user turns from `allenai/tulu-3-sft-mixture`, read in a fixed number of
  rows from every parquet shard of the stream, minus safety-data sources (they overlap
  our WildGuardMix fine-tuning data and hold harmful requests);
- `word`: the most frequent lowercase alphabetic words over both candidate lists,
  minus stop-words, group terms and the P/N lexicon (the original word anchors of RR).
Alpaca and Tulu keep single-sentence instructions of 5–15 words (`instruction_ok`) and
sample them with `random.Random(seed)`, so a pool depends only on (dataset revision,
config). The rows are licensed data: they live in the private artifact store only, never
in git or `results/` (PLAN §5, DC-18).

`subset_rows` gives nested seeded subsets (sizes {50, 100, 250, 500}, seeds 0–4): one
permutation per seed, so a smaller subset is always a prefix of a larger one.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from rbbd.data.groups import GROUP_TOKENS
from rbbd.data.hashing import normalize
from rbbd.utils.logging import get_logger

log = get_logger(__name__)

POOL_SOURCES = ("alpaca", "tulu", "word")
ANCHOR_SIZES = (50, 100, 250, 500)  # plus "all" (the whole pool), D-015
ANCHOR_SEEDS = (0, 1, 2, 3, 4)
ALL = -1  # anchor_seed / n_anchors marker for "the whole pool, no subsampling"

_BAD_CHARS = re.compile(r"[=+*/\\^_{}\[\]<>|~#$%&@`\n\r\t]")
_INNER_BOUNDARY = re.compile(r"[.!?]\s+\S")
_LOWER_WORD = re.compile(r"^[a-z]+$")

# English function words removed from the word pool (D-015 "excluding stop-words").
WORD_STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    either even ever every few for from further get gets got had has have having he her
    here hers herself him himself his how however i if in into is it its itself just
    least less let like make many may me might more most much must my myself neither no
    nor not now of off often on once one only or other our ours ourselves out over own
    per please rather same shall she should since so some such than that the their
    theirs them themselves then there these they this those though through thus to too
    under until up upon us use used using very via was we were what when where whether
    which while who whom whose why will with within without would yet you your yours
    yourself yourselves""".split()
)


def instruction_ok(text: str, min_words: int = 5, max_words: int = 15) -> bool:
    """True for a single-line, single-sentence ASCII instruction of 5–15 words (D-015).

    Rejects code, maths and markup characters, URLs and anything with an inner sentence
    boundary ("Do X. Then Y.").
    """
    t = text.strip()
    if not t or not t.isascii() or _BAD_CHARS.search(t) or "http" in t.lower():
        return False
    if _INNER_BOUNDARY.search(t.rstrip(".!?")):
        return False
    return min_words <= len(t.split()) <= max_words


def alpaca_candidates(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Filtered instructions (rows with an empty `input`), in dataset order, deduplicated."""
    out: dict[str, str] = {}
    for row in rows:
        if (row.get("input") or "").strip():
            continue
        text = str(row.get("instruction", "")).strip()
        if instruction_ok(text):
            out.setdefault(normalize(text), text)
    return list(out.values())


def first_user_turn(messages: Sequence[Mapping[str, Any]]) -> str | None:
    """Content of the first `user` message of a chat, or None."""
    for m in messages or ():
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return None


def tulu_candidates(
    rows: Iterable[Mapping[str, Any]], exclude_sources: Sequence[str]
) -> tuple[list[str], Counter]:
    """Filtered first user turns, in stream order, deduplicated; plus kept rows per source.

    A row whose `source` contains any `exclude_sources` substring is skipped.
    """
    out: dict[str, str] = {}
    sources: dict[str, str] = {}
    for row in rows:
        source = str(row.get("source", ""))
        if any(x in source for x in exclude_sources):
            continue
        text = first_user_turn(row.get("messages", ()))
        if text is not None and instruction_ok(text):
            key = normalize(text)
            if key not in out:
                out[key] = text.strip()
                sources[key] = source
    return list(out.values()), Counter(sources.values())


def sample_pool(candidates: Sequence[str], size: int, seed: int, exclude: set[str]) -> list[str]:
    """`size` candidates drawn with `random.Random(seed)`, never one in `exclude`
    (normalised text of the authored sets). Raises if too few remain."""
    pool = [c for c in candidates if normalize(c) not in exclude]
    if len(pool) < size:
        raise ValueError(f"only {len(pool)} candidates for a pool of {size}")
    return random.Random(seed).sample(pool, size)


def word_pool(texts: Iterable[str], size: int, banned: set[str]) -> list[str]:
    """The `size` most frequent lowercase alphabetic words (≥ 3 letters) in `texts`,
    ties broken alphabetically, minus stop-words, group terms and `banned` (P/N words)."""
    counts: Counter = Counter()
    for t in texts:
        for w in t.split():
            w = w.strip(".,;:!?'\"()")
            if len(w) >= 3 and _LOWER_WORD.match(w):
                counts[w] += 1
    blocked = WORD_STOPWORDS | GROUP_TOKENS | banned
    ranked = sorted((w for w in counts if w not in blocked), key=lambda w: (-counts[w], w))
    if len(ranked) < size:
        raise ValueError(f"only {len(ranked)} distinct content words for a pool of {size}")
    return ranked[:size]


def lexicon(sentences: Iterable[str]) -> set[str]:
    """Lowercased alphabetic words of the P/N sentences (all variants)."""
    return {w for s in sentences for w in re.findall(r"[a-z]+", s.lower())}


def subset_rows(n_rows: int, size: int, seed: int) -> list[int]:
    """Positions 0..n_rows-1 of a nested seeded subset: the first `size` of one fixed
    permutation per seed, so subsets of one seed are nested. `size == ALL` → all rows."""
    if size == ALL or size >= n_rows:
        return list(range(n_rows))
    return sorted(np.random.default_rng(seed).permutation(n_rows)[:size].tolist())


Fetch = Callable[[str, Mapping[str, Any]], tuple[Iterable[Mapping[str, Any]], str]]


def fetch_rows(source: str, spec: Mapping[str, Any]) -> tuple[Iterable[Mapping[str, Any]], str]:
    """(rows, resolved revision sha) of one pool's dataset; the only network access here.

    Alpaca loads whole (≈ 24 MB). Tulu streams: `scan_rows` rows are read from the start
    of every parquet shard (`IterableDataset.shard`), so the sample spans the mixture
    without downloading it (B-027 records the method actually used at run time).
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(spec["dataset"], revision=spec.get("revision")).sha
    if source == "alpaca":
        return load_dataset(spec["dataset"], split="train", revision=sha), sha
    stream = load_dataset(spec["dataset"], split="train", revision=sha, streaming=True)
    total = int(spec.get("scan_rows", 60000))
    n_shards = int(getattr(stream, "n_shards", 1) or 1)
    per_shard = max(1, total // n_shards)

    def rows() -> Iterable[Mapping[str, Any]]:
        if n_shards > 1 and hasattr(stream, "shard"):
            for i in range(n_shards):
                yield from stream.shard(num_shards=n_shards, index=i).take(per_shard)
        else:
            yield from stream.take(total)

    log.info("tulu: %d shards, %d rows per shard", n_shards, per_shard)
    return rows(), sha


def build_pools(
    cfg: Mapping[str, Any],
    authored: set[str],
    pn_sentences: Iterable[str],
    fetch: Fetch | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """({source: sentences}, stats) for every pool in `cfg` (`sentences.anchor_pools`).

    `authored` = normalised text of every authored target/attribute/anchor sentence (a
    pool sentence equal to one would alias its embedding row). `pn_sentences` feed the
    word pool's banned lexicon. `fetch` is replaced by a stub in CPU tests.
    """
    fetch = fetch or fetch_rows
    size, seed = int(cfg.get("size", 1000)), int(cfg.get("seed", 0))
    pools: dict[str, list[str]] = {}
    stats: dict[str, Any] = {"size": size, "seed": seed}
    candidates: dict[str, list[str]] = {}
    if "alpaca" in cfg:
        rows, sha = fetch("alpaca", cfg["alpaca"])
        candidates["alpaca"] = alpaca_candidates(rows)
        stats["alpaca"] = {"dataset": cfg["alpaca"]["dataset"], "revision": sha,
                           "candidates": len(candidates["alpaca"])}  # fmt: skip
    if "tulu" in cfg:
        spec = cfg["tulu"]
        rows, sha = fetch("tulu", spec)
        candidates["tulu"], by_source = tulu_candidates(rows, spec.get("exclude_sources", ()))
        stats["tulu"] = {"dataset": spec["dataset"], "revision": sha,
                         "candidates": len(candidates["tulu"]),
                         "exclude_sources": list(spec.get("exclude_sources", ())),
                         "candidates_by_source": dict(by_source.most_common())}  # fmt: skip
    for src in ("alpaca", "tulu"):
        if src in candidates:
            pools[src] = sample_pool(candidates[src], size, seed, authored)
            words = [len(s.split()) for s in pools[src]]
            stats[src]["mean_words"] = round(float(np.mean(words)), 2)
    if "word" in cfg:
        from_src = list(cfg["word"].get("from", ["alpaca", "tulu"]))
        texts = [t for s in from_src for t in candidates.get(s, [])]
        pools["word"] = word_pool(texts, size, lexicon(pn_sentences))
        stats["word"] = {"from": from_src, "n_texts": len(texts)}
    return pools, stats
