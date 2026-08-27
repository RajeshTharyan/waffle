"""Sentence splitting, tokenisation, and small numeric helpers."""

from __future__ import annotations

import re
from typing import List, Sequence

_WORD_RE = re.compile(r"[A-Za-z']+|\d+%?|£\d+|\$\d+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_ALPHA_RE = re.compile(r"[A-Za-z]")
_LOWER_ALPHA_RE = re.compile(r"[a-z]")


def split_sentences(text: str) -> List[str]:
    """Split business prose on punctuation followed by a capital or digit."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def word_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def count_matches(patterns: Sequence[str], text: str) -> int:
    return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)


def density_per_100(words_count: int, raw_count: int) -> float:
    return 0 if words_count == 0 else (raw_count / words_count) * 100


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def normalize(val: float, low: float, high: float) -> float:
    """Clip and scale to [0, 1]."""
    if val <= low:
        return 0.0
    if val >= high:
        return 1.0
    return (val - low) / (high - low)


def alphabetic_word_count(words: Sequence[str]) -> int:
    return len([w for w in words if _ALPHA_RE.match(w)])


def unique_alpha_term_count(words: Sequence[str]) -> int:
    return len({w for w in words if _LOWER_ALPHA_RE.match(w)})
