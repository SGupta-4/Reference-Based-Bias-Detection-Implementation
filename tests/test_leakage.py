"""No leakage of generated text, prompts or secrets (D-036, D-037). DC-18."""

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


def test_no_generated_or_prompt_text_tracked():
    """PLACEHOLDER(M5) git-tracked files and results/ contain no generation or prompt text."""


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
