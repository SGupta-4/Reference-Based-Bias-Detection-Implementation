"""Text normalisation and content hashes for sentence sets and prompts.

A sentence set's identity is the SHA-256 of its NFC-normalised, whitespace-stripped
lines joined with "\\n", in order. The embedding cache key (D-019) includes these
hashes, so editing one sentence invalidates exactly the caches that used it.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable


def normalize(text: str) -> str:
    """NFC-normalise, collapse internal whitespace runs to one space and strip."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def text_hash(text: str) -> str:
    """SHA-256 hex of one normalised string (used for prompts and sentence rows)."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def set_hash(texts: Iterable[str]) -> str:
    """SHA-256 hex of an ordered list of normalised strings joined by newlines."""
    return hashlib.sha256("\n".join(normalize(t) for t in texts).encode("utf-8")).hexdigest()
