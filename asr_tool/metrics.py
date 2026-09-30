"""Accuracy metrics for evaluating transcripts against a reference.

WER = (substitutions + deletions + insertions) / number of reference words,
computed with a standard Levenshtein edit distance over word sequences.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace so formatting
    differences (e.g. "Hello," vs "hello") are not counted as errors."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s']", " ", text)
    text = text.replace("'", "")
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Minimum number of substitutions, insertions and deletions to turn
    `ref` into `hyp`. Uses two rows of the DP table to keep memory O(n)."""
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            current[j] = min(
                previous[j] + 1,         # deletion
                current[j - 1] + 1,      # insertion
                previous[j - 1] + cost,  # substitution / match
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str, normalize: bool = True) -> float:
    if normalize:
        reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    ref_words, hyp_words = reference.split(), hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def character_error_rate(reference: str, hypothesis: str, normalize: bool = True) -> float:
    if normalize:
        reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)
