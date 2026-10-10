"""One-off, frozen WildGuardMix prompt → topic mapping and its human validation (App. B;
D-024, D-043).

- `mapper_messages` asks `Qwen/Qwen2.5-7B-Instruct` (greedy, via vLLM) to pick one of the
  nine T2 topics, each listed with its member groups, or "none". `parse_topic` reads the
  answer. The map is frozen as `resources/topic_map.csv` (prompt SHA-256, topic: no text).
- `labelling_sheet` draws ~100 prompts stratified by predicted topic. The sheet the user
  labels holds the prompt text but *not* the predicted topic (blind labelling), so it
  stays in the private store and the Kaggle output only (D-043).
- `agreement` joins the user's labels to the map by SHA-256. The map is accepted at
  ≥ 80% agreement (D-043); only hashes and labels are kept (privately).
"""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rbbd.data.groups import GROUPS, TOPICS

MAPPER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
NONE = "none"
MAP_PATH = Path(__file__).resolve().parents[1] / "resources" / "topic_map.csv"
ACCEPT_AGREEMENT = 0.80


def _topic_lines() -> str:
    members: dict[str, list[str]] = defaultdict(list)
    for g in GROUPS:
        members[g.topic].append(g.name)
    return "\n".join(f"- {t}: {', '.join(members[t])}" for t in TOPICS)


def mapper_messages(prompt: str) -> list[dict[str, str]]:
    """Chat messages asking the mapper for exactly one topic name or "none"."""
    instruction = (
        "You label user requests by the social group they are about. Topics and their "
        f"member groups:\n{_topic_lines()}\n\n"
        "Read the request between <request> tags. Answer with exactly one topic name from "
        f'the list, or "{NONE}" if the request is not about any of these groups. '
        "Answer with the topic name only."
    )
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"<request>\n{prompt}\n</request>"},
    ]


def parse_topic(answer: str) -> str:
    """The first topic name found in the answer (case-insensitive), else "none"."""
    low = answer.strip().lower()
    hits = [(low.find(t.lower()), t) for t in TOPICS if t.lower() in low]
    return min(hits)[1] if hits else NONE


def write_map(rows: Iterable[tuple[str, str]], path: Path = MAP_PATH) -> Path:
    """Write (prompt sha256, topic) rows sorted by hash: the frozen map."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prompt_sha256", "topic"])
        for sha, topic in sorted(rows):
            w.writerow([sha, topic])
    return path


def read_map(path: Path = MAP_PATH) -> dict[str, str]:
    """{prompt sha256: topic} from the frozen map; empty if it does not exist yet."""
    if not Path(path).exists():
        return {}
    with open(path, newline="") as fh:
        return {r["prompt_sha256"]: r["topic"] for r in csv.DictReader(fh)}


def map_sha256(path: Path = MAP_PATH) -> str | None:
    """SHA-256 of the frozen map file (pinned as `bench.wgm.topic_map_sha256`)."""
    import hashlib

    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def labelling_sheet(
    prompts: Mapping[str, str], topic_of: Mapping[str, str], n: int = 100, seed: int = 0
) -> str:
    """CSV text for blind labelling: ~`n` prompts stratified by predicted topic.

    Each predicted topic (and "none") gets max(4, proportional) rows, capped by what
    exists. Columns: row, prompt_sha256, prompt, label (empty), with the allowed labels
    listed in the header comment row. Private: holds prompt text (D-043).
    """
    by_topic: dict[str, list[str]] = defaultdict(list)
    for sha in sorted(prompts):
        by_topic[topic_of.get(sha, NONE)].append(sha)
    total = sum(len(v) for v in by_topic.values())
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for topic in sorted(by_topic):
        pool = by_topic[topic]
        k = min(len(pool), max(4, round(n * len(pool) / total)))
        chosen += [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))]
    order = rng.permutation(len(chosen))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["row", "prompt_sha256", "prompt", "label"])
    w.writerow(["#", "allowed labels:", " | ".join([*TOPICS, NONE]), ""])
    for row, i in enumerate(order, 1):
        w.writerow([row, chosen[i], prompts[chosen[i]], ""])
    return buf.getvalue()


def read_labels(csv_text: str) -> dict[str, str]:
    """{prompt sha256: label} from a filled sheet; labels are matched case-insensitively
    to a topic or "none"; the comment row and blank labels are skipped."""
    canon = {t.lower(): t for t in [*TOPICS, NONE]}
    out = {}
    for r in csv.DictReader(io.StringIO(csv_text)):
        label = (r.get("label") or "").strip().lower()
        if r.get("row") == "#" or not label:
            continue
        if label not in canon:
            raise ValueError(f"row {r.get('row')}: unknown label {r.get('label')!r}")
        out[r["prompt_sha256"]] = canon[label]
    return out


def agreement(labels: Mapping[str, str], topic_of: Mapping[str, str]) -> dict[str, Any]:
    """Agreement of the frozen map with human labels (D-043 acceptance ≥ 80%).

    Returns {"n", "agreement", "accepted", "per_topic": {label: {"n", "agree"}},
    "confusion": {"label->predicted": count}} (aggregates only).
    """
    pairs = [(lab, topic_of.get(sha, NONE)) for sha, lab in labels.items()]
    if not pairs:
        raise ValueError("no labelled rows")
    agree = sum(a == b for a, b in pairs)
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for a, b in pairs:
        per[a][0] += 1
        per[a][1] += a == b
    rate = agree / len(pairs)
    return {
        "n": len(pairs),
        "agreement": rate,
        "accepted": rate >= ACCEPT_AGREEMENT,
        "per_topic": {t: {"n": n, "agree": k} for t, (n, k) in sorted(per.items())},
        "confusion": dict(Counter(f"{a}->{b}" for a, b in pairs if a != b)),
    }


def topic_counts(topic_of: Mapping[str, str], ids: Sequence[str]) -> dict[str, int]:
    """Prompts per topic among `ids` (reported next to WGM scores, D-023 Revisit-if)."""
    return dict(Counter(topic_of.get(i, NONE) for i in ids))


def run_mapper(
    prompts: Mapping[str, str], tp: int = 2, max_model_len: int = 4096, generate_fn: Any = None
) -> dict[str, str]:
    """{prompt sha256: topic} from the mapper model, greedy (D-043).

    `prompts`: {sha256: text}. Prompts too long for the context are mapped to "none" and
    counted by the caller. `generate_fn(chat_texts) -> answers` is vLLM by default.
    """
    from rbbd.models.loading import load_tokenizer

    tok = load_tokenizer(MAPPER_MODEL)
    shas, texts, too_long = [], [], []
    for sha, prompt in sorted(prompts.items()):
        text = tok.apply_chat_template(
            mapper_messages(prompt), tokenize=False, add_generation_prompt=True
        )
        if len(tok(text)["input_ids"]) > max_model_len - 16:
            too_long.append(sha)
            continue
        shas.append(sha)
        texts.append(text)
    if generate_fn is None:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=MAPPER_MODEL,
            tensor_parallel_size=tp,
            dtype="float16",
            max_model_len=max_model_len,
            enforce_eager=True,
        )
        sp = SamplingParams(temperature=0.0, max_tokens=12)

        def generate_fn(xs: list[str]) -> list[str]:
            return [o.outputs[0].text for o in llm.generate(xs, sp, use_tqdm=False)]

    answers = generate_fn(texts)
    out = {sha: parse_topic(a) for sha, a in zip(shas, answers, strict=True)}
    out.update({sha: NONE for sha in too_long})
    return out
