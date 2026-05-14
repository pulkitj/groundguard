"""Tier 0 classifier — rules-based Extractive/Inferential atom classification."""
from __future__ import annotations
import re
from groundguard.models.internal import ClassifiedAtom

INFERENTIAL_SIGNALS = {
    "trend", "trajectory", "suggests", "indicates", "on track", "at risk",
    "appears to", "likely", "projected", "based on", "derived from",
    "analysis shows", "pattern", "forecast", "outlook", "implies",
    "consistent with", "points to", "expected to",
}

# Decimal-safe sentence splitter: splits on [.!?] NOT between two digits, or newlines.
# Preserves: $4.2M, v2.1, 3.14
_SENTENCE_SPLIT_RE = re.compile(r'(?<!\d)[.!?](?!\d)\s+|\n+')


def parse_and_classify(claim: str) -> list[ClassifiedAtom]:
    """
    Zero-cost, zero-LLM heuristic classifier.

    1. Split claim into atomic sentences using decimal-safe regex.
    2. For each sentence: classify as Inferential if any INFERENTIAL_SIGNALS token
       appears as a case-insensitive whole-word match; otherwise Extractive.

    Returns:
        List of ClassifiedAtom objects. Empty string returns empty list.
        Punctuation-only input returns empty list (no IndexError).
    """
    if not claim or not claim.strip():
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(claim) if s.strip()]

    if not sentences:
        return []

    atoms: list[ClassifiedAtom] = []
    for sentence in sentences:
        lower = sentence.lower()
        is_inferential = any(
            re.search(rf'\b{re.escape(signal)}\b', lower)
            for signal in INFERENTIAL_SIGNALS
        )
        atoms.append(ClassifiedAtom(
            claim_text=sentence,
            claim_type="Inferential" if is_inferential else "Extractive",
        ))

    return atoms
