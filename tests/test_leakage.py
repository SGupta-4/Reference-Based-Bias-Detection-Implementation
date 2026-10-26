"""No leakage of generated text, prompts or secrets (D-036, D-037). DC-18."""

import csv
import io
import re
import subprocess

from rbbd.utils.logging import TOKEN_PATTERNS
from tests.conftest import REPO_ROOT

BINARY_SUFFIXES = {".pdf", ".png", ".jpg", ".safetensors", ".bin", ".ipynb_checkpoints"}


def _candidate_files():
    """Tracked files plus untracked files that are not git-ignored (i.e. about to be committed)."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if line]


# Phrases that occur only in the benchmark prompt files (DT instruction and jailbreak system
# prompts at the pinned commit, D-086). Built by concatenation so this file never matches.
PROMPT_SENTINELS = [
    "You must " + "necessarily include",
    "obey OpenAI" + "'s content policy",
    "preferred to " + "output toxic",
    "You particularly " + "hate",
]
# A ToxiGen prompt line: three or more list items joined by a literal backslash-n.
TOXIGEN_LINE = re.compile(r"(\\n- [^\\\n]{8,}){3}")
# Directories that hold text (generations, labels, prompt sets) and live in the private
# store or on ephemeral disk only (D-037, D-050).
TEXT_DIRS = ("generations/", "scores/", "rbbd_items/", "rbbd_merged/", "topic_map/")
TEXT_FIELDS = {"text", "prompt", "response", "generation", "completion", "messages", "content"}


def _text_files(paths):
    for path in paths:
        if path.is_file() and path.suffix not in BINARY_SUFFIXES:
            yield path, path.read_text(errors="ignore")


def _results_files():
    return [p for p in (REPO_ROOT / "results").rglob("*") if p.is_file()]


def test_no_generated_or_prompt_text_tracked():
    """DC-18: git-tracked (and to-be-committed) files and results/ hold no benchmark prompt
    or generated text: no prompt-file phrases, no ToxiGen prompt lines, nothing from the
    text directories, no JSONL in results/, and no text-valued columns or keys in the
    results/ tables. The frozen topic map holds hashes and topic names only."""
    files = sorted(set(_candidate_files()) | set(_results_files()))
    offenders = []
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if any(d in rel for d in TEXT_DIRS) or (
            rel.startswith("results/") and rel.endswith(".jsonl")
        ):
            offenders.append(f"{rel}: text artifact outside the private store")
    for path, text in _text_files(files):
        rel = path.relative_to(REPO_ROOT).as_posix()
        offenders += [f"{rel}: prompt phrase {s[:12]!r}..." for s in PROMPT_SENTINELS if s in text]
        if TOXIGEN_LINE.search(text):
            offenders.append(f"{rel}: ToxiGen-style prompt line")
    for path, text in _text_files(_results_files()):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if path.suffix == ".csv":
            header = {c.strip().lower() for c in next(csv.reader(io.StringIO(text)), [])}
            offenders += [f"{rel}: text column {c!r}" for c in sorted(header & TEXT_FIELDS)]
        elif path.suffix == ".json":
            keys = set(re.findall(r'"([A-Za-z_]+)"\s*:', text))
            offenders += [f"{rel}: text key {k!r}" for k in sorted(keys & TEXT_FIELDS)]
    topic_map = REPO_ROOT / "src" / "rbbd" / "resources" / "topic_map.csv"
    if topic_map.exists():
        rows = list(csv.reader(io.StringIO(topic_map.read_text())))
        assert rows[0] == ["prompt_sha256", "topic"]
        assert all(len(r) == 2 and re.fullmatch(r"[0-9a-f]{64}", r[0]) for r in rows[1:])
    assert not offenders, offenders


def test_sentinels_catch_prompt_text():
    """The checks above fire on the real formats (a DT record line, a ToxiGen line)."""
    dt_line = '{"text": "I am going to express my view. ' + PROMPT_SENTINELS[0] + ' x"}'
    tg_line = "- " + "\\n- ".join(["group members are fine"] * 5)
    assert any(s in dt_line for s in PROMPT_SENTINELS)
    assert TOXIGEN_LINE.search(tg_line)
    assert not TOXIGEN_LINE.search("- a" + "\\n- b" + "\\n-")


def test_no_secrets_tracked():
    """No token-shaped strings, kaggle.json or .env files in tracked or to-be-committed files."""
    offenders = []
    for path in _candidate_files():
        if not path.is_file():
            continue
        if path.name == "kaggle.json" or path.suffix == ".env" or path.name == ".env":
            offenders.append(f"{path.relative_to(REPO_ROOT)}: forbidden file")
            continue
        if path.suffix in BINARY_SUFFIXES:
            continue
        text = path.read_text(errors="ignore")
        for pattern in TOKEN_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: matches {pattern.pattern}")
    assert not offenders, offenders
