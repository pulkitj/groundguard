"""Tests for Tier 0 classifier — TDD items #1 and #2."""
import pytest
from groundguard.core.classifier import parse_and_classify
from groundguard.models.internal import ClassifiedAtom


def test_single_sentence_with_decimal_stays_one_atom():
    """TDD #1: Decimal-safe split — '$4.2M' must not be split into two atoms."""
    atoms = parse_and_classify("Revenue was $4.2M.")
    assert len(atoms) == 1


def test_two_sentences_produce_two_atoms():
    """TDD #1: Two sentences produce exactly 2 atoms."""
    atoms = parse_and_classify("Q1 was $4.2M. Q2 was $5.1M.")
    assert len(atoms) == 2


def test_punctuation_only_claim_does_not_raise():
    """TDD #1 extension: Punctuation-only claim must not raise IndexError."""
    result = parse_and_classify("...")
    assert isinstance(result, list)  # Returns empty list or single atom — no crash


def test_inferential_signals_produce_inferential_classification():
    """TDD #2: Each inferential signal keyword produces Inferential classification."""
    signals = [
        "trend", "trajectory", "suggests", "indicates", "on track", "at risk",
        "appears to", "likely", "projected", "based on", "derived from",
        "analysis shows", "pattern", "forecast", "outlook", "implies",
        "consistent with", "points to", "expected to",
    ]
    for signal in signals:
        atoms = parse_and_classify(f"The data {signal} something important.")
        assert any(a.claim_type == "Inferential" for a in atoms), (
            f"Signal '{signal}' should produce at least one Inferential atom"
        )


def test_plain_statement_produces_extractive():
    """Non-inferential sentence defaults to Extractive."""
    atoms = parse_and_classify("Revenue was $5M in Q3.")
    assert len(atoms) == 1
    assert atoms[0].claim_type == "Extractive"


def test_multiline_claim_splits_correctly():
    """Newlines should produce separate atoms."""
    atoms = parse_and_classify("Line one.\nLine two.")
    assert len(atoms) == 2


def test_empty_string_does_not_raise():
    """Empty string should return empty list or single empty atom without raising."""
    result = parse_and_classify("")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Open Issue #5 — classifier line 36: return [] after empty split
# ---------------------------------------------------------------------------

def test_punctuation_delimiter_only_returns_empty_list():
    """Open Issue #5: claim that survives strip but produces no sentences after split returns []."""
    # ". " passes claim.strip() check ("." is truthy), but the regex splits on ". "
    # producing only empty strings -> sentences = [] -> line 36 return []
    atoms = parse_and_classify(". ")
    assert atoms == []
