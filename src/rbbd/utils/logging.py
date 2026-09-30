"""Structured logging with secret redaction (D-036).

Every module logs through `get_logger(__name__)`. The single `rbbd` handler writes
to the *current* `sys.stderr` and passes every record through `redact`, so a token
that reaches a log call (for example inside an exception message from
huggingface_hub) is replaced before it is written. Manifests and env.json call
`redact` directly for the same reason.
"""

from __future__ import annotations

import logging
import os
import re
import sys

# Environment variables whose *values* must never appear in logs or artifacts.
SECRET_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "KAGGLE_KEY")
REDACTED = "[REDACTED]"

# Token shapes redacted even when the value is not in the environment
# (HF user tokens are "hf_" + 34 alphanumerics; GitHub classic and fine-grained PATs).
# The 30-character floor keeps library identifiers such as `hf_overrides` in vLLM
# logs intact (D-045).
TOKEN_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
)

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def redact(text: str) -> str:
    """Return `text` with secret env-var values and token-shaped strings replaced.

    Values shorter than 8 characters are ignored so that an empty or dummy
    variable cannot blank out ordinary text.
    """
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name)
        if value and len(value) >= 8:
            text = text.replace(value, REDACTED)
    for pattern in TOKEN_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


class RedactingFilter(logging.Filter):
    """Handler filter that freezes the formatted message and redacts it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True


class _StderrHandler(logging.Handler):
    """Writes to whatever `sys.stderr` is at emit time (so pytest capture and
    notebook stream swaps both see the output)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


def get_logger(name: str = "rbbd") -> logging.Logger:
    """Return a logger under the `rbbd` hierarchy with the redacting handler installed once."""
    root = logging.getLogger("rbbd")
    if not any(isinstance(h, _StderrHandler) for h in root.handlers):
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler.addFilter(RedactingFilter())
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    if name == "rbbd" or name.startswith("rbbd."):
        return logging.getLogger(name)
    return logging.getLogger(f"rbbd.{name}")
