"""Tier 2.5 — Numerical consistency pre-check."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groundguard.models.internal import VerificationContext
    from groundguard.loaders.chunker import Chunk

from groundguard.models.result import Citation

# Matches: 30%, 300%, $4.2M, $300, 1,000,000, 4.2, 2023, -5%, -$4.2M
_NUMBER_PATTERN = r'(?<!\w)-?[$]?\d[\d,]*(?:\.\d+)?[%MBKT]?'

_STOPWORDS = {"the", "a", "an", "and", "or", "in", "of", "to", "is", "was", "be", "see", "for",
              "section", "details", "reference", "per", "at", "by", "with", "that", "this",
              "which", "from", "are", "has", "have", "had", "not", "do", "does", "shall",
              "will", "would", "may", "can", "its", "it", "we", "our"}

# Year ONLY in temporal context: "in 2023", "for 2024", "Q3 2023", "FY2024", etc.
_YEAR_CONTEXT_PATTERN = r'(?:in|for|during|as of|fiscal|FY|Q[1-4])\s+(\d{4})\b'


def _normalise_number(s: str) -> str:
    """Strip currency symbols, commas, and suffix letters to get numeric string."""
    s = s.strip()
    # Extract and preserve leading minus sign
    sign = ''
    if s.startswith('-'):
        sign = '-'
        s = s[1:]
    # Remove leading $
    if s.startswith('$'):
        s = s[1:]
    # Remove trailing %, M, B, K, T (but if it ends with %, the number is the part before)
    if s.endswith('%'):
        return sign + s.rstrip('%').replace(',', '')
    # Remove M/B/K/T suffixes (abbreviations for million/billion/etc.)
    if s and s[-1].upper() in ('M', 'B', 'K', 'T'):
        s = s[:-1]
    return sign + s.replace(',', '')


def _has_sufficient_metric_context(text: str) -> bool:
    """Return True if text has >= 1 substantive (non-stopword) non-digit word."""
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    substantive = [t for t in tokens if t not in _STOPWORDS]
    return len(substantive) >= 1


def _is_within_range(source_val: float, claim_vals: list[float]) -> bool:
    """Return True if source_val falls between min and max of claim_vals."""
    if len(claim_vals) < 2:
        return False
    lo, hi = min(claim_vals), max(claim_vals)
    return lo <= source_val <= hi



def extract_contextual_years(text: str) -> list[str]:
    """Return only years that appear in temporal context."""
    # Find all temporal prefixes
    prefix_pattern = r'\b(?:in|for|during|as\s+of|fiscal|FY|Q[1-4])\b'
    years = []
    
    for match in re.finditer(prefix_pattern, text, re.IGNORECASE):
        start_idx = match.end()
        # Look ahead up to 100 characters
        sub = text[start_idx:start_idx+100]
        m = re.match(r'^\s*(\d{4})(?:\s*(?:and|or|to|,)\s*(\d{4}))*', sub, re.IGNORECASE)
        if m:
            matched_str = m.group(0)
            list_years = re.findall(r'\b\d{4}\b', matched_str)
            years.extend(list_years)
            
    # Direct formats like FY2024 or Q32023
    direct_pattern = r'\b(?:FY|Q[1-4])(\d{4})\b'
    for match in re.finditer(direct_pattern, text, re.IGNORECASE):
        years.append(match.group(1))
        
    return years


@dataclass
class NumericalCheckResult:
    claim_number: str
    source_number: str
    match: bool
    chunk_id: str


@dataclass
class Tier25Result:
    has_conflict: bool
    verification_method: str = "tier25_numerical"
    evidence_bundle: list = field(default_factory=list)
    conflict_citation: "Citation | None" = None
    numerical_checks: list = field(default_factory=list)


def extract_excerpt_from_chunk(chunk: "Chunk", pattern: str) -> "tuple[str, int, int] | None":
    """Find first regex match in chunk.text_content; return (text, start, end) or None."""
    m = re.search(pattern, chunk.text_content, re.IGNORECASE)
    if m:
        return m.group(0), m.start(), m.end()
    return None


def _bm25_score_single(query_tokens: list, chunk: "Chunk") -> float:
    """Return BM25 relevance score [0.0, 1.0] for a single chunk against query tokens."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return 0.0
    doc_tokens = chunk.text_content.split()
    if not doc_tokens:
        return 0.0
    bm25 = BM25Okapi([doc_tokens])
    scores = bm25.get_scores(query_tokens)
    raw = float(scores[0])
    return max(0.0, min(1.0, raw / 10.0))  # normalise to [0, 1]


def build_evidence_bundle(ctx: "VerificationContext", chunks: list, top_k: int = 3) -> list:
    """Return chunks that contain at least one number."""
    result = []
    for chunk in chunks:
        if re.search(_NUMBER_PATTERN, chunk.text_content):
            result.append(chunk)
        if len(result) >= top_k:
            break
    return result


def run(ctx: "VerificationContext", chunks: list) -> Tier25Result:
    """Run numerical consistency check."""
    # Extract numbers from claim
    claim_numbers_raw = re.findall(_NUMBER_PATTERN, ctx.claim)
    if not claim_numbers_raw:
        return Tier25Result(has_conflict=False, evidence_bundle=build_evidence_bundle(ctx, chunks))

    # Guard: insufficient metric context
    if not _has_sufficient_metric_context(ctx.claim):
        return Tier25Result(has_conflict=False, evidence_bundle=build_evidence_bundle(ctx, chunks))

    # Parse claim numbers to floats where possible
    claim_floats = []
    for n in claim_numbers_raw:
        try:
            claim_floats.append(float(_normalise_number(n)))
        except ValueError:
            pass

    # Extract contextual years from claim
    claim_years = set(extract_contextual_years(ctx.claim))

    # Check for range in claim (two numeric values)
    is_range_claim = len(claim_floats) >= 2

    checks = []
    conflict_found = False
    conflict_citation = None
    evidence_bundle = build_evidence_bundle(ctx, chunks)

    # Year conflict check loop
    year_conflict_citation = None
    for chunk in chunks:
        chunk_years = set(extract_contextual_years(chunk.text_content))
        if claim_years and chunk_years and claim_years.issubset(chunk_years):
            # at least one chunk supports the claim year — no conflict
            year_conflict_citation = None
            break
        if claim_years and chunk_years and not claim_years.issubset(chunk_years):
            # tentative conflict — keep looking in case a later chunk matches
            excerpt_result = extract_excerpt_from_chunk(chunk, _YEAR_CONTEXT_PATTERN)
            if excerpt_result and year_conflict_citation is None:
                excerpt_text, start, end = excerpt_result
                year_conflict_citation = Citation(
                    source_id=chunk.source_id,
                    excerpt=excerpt_text,
                    excerpt_char_start=chunk.char_start + start,
                    excerpt_char_end=chunk.char_start + end,
                )
    else:
        # Loop completed without finding a supporting chunk
        if year_conflict_citation is not None:
            conflict_found = True
            conflict_citation = year_conflict_citation

    matched_floats = set()
    for chunk in chunks:
        chunk_numbers_raw = re.findall(_NUMBER_PATTERN, chunk.text_content)
        chunk_floats = []
        for n in chunk_numbers_raw:
            try:
                chunk_floats.append(float(_normalise_number(n)))
            except ValueError:
                pass

        # For each claim number, check against chunk numbers
        # Skip year values — they are exclusively handled by the year conflict loop above.
        claim_year_floats = {float(y) for y in claim_years}
        for i, claim_float in enumerate(claim_floats):
            if claim_float in claim_year_floats:
                continue
            claim_raw = claim_numbers_raw[i] if i < len(claim_numbers_raw) else str(claim_float)

            if is_range_claim:
                # Range claim: check if any chunk value falls within range
                for chunk_float in chunk_floats:
                    if _is_within_range(chunk_float, claim_floats):
                        # source value is within claim range — not a conflict
                        checks.append(NumericalCheckResult(
                            claim_number=claim_raw,
                            source_number=str(chunk_float),
                            match=True,
                            chunk_id=chunk.chunk_id,
                        ))
                        break
                else:
                    # No chunk value within range
                    if chunk_floats:
                        checks.append(NumericalCheckResult(
                            claim_number=claim_raw,
                            source_number=chunk_numbers_raw[0] if chunk_numbers_raw else "",
                            match=False,
                            chunk_id=chunk.chunk_id,
                        ))
                break  # only check range once per chunk
            else:
                # Exact match check
                if chunk_floats and claim_float not in chunk_floats:
                    # Only flag when source has exactly one number: ambiguous arithmetic
                    # context (≥2 source values) passes to Tier 3 for LLM reasoning.
                    if len(chunk_floats) > 1:
                        continue
                    # Mismatch found
                    chunk_raw = chunk_numbers_raw[0] if chunk_numbers_raw else ""
                    checks.append(NumericalCheckResult(
                        claim_number=claim_raw,
                        source_number=chunk_raw,
                        match=False,
                        chunk_id=chunk.chunk_id,
                    ))
                    # Build citation pointing to the conflicting value in chunk
                    if chunk_raw and conflict_citation is None:
                        excerpt_result = extract_excerpt_from_chunk(chunk, re.escape(chunk_raw))
                        if excerpt_result:
                            excerpt_text, start, end = excerpt_result
                            conflict_citation = Citation(
                                source_id=chunk.source_id,
                                excerpt=excerpt_text,
                                excerpt_char_start=chunk.char_start + start,
                                excerpt_char_end=chunk.char_start + end,
                            )
                    continue
                elif chunk_floats and claim_float in chunk_floats:
                    matched_floats.add(claim_float)
                    checks.append(NumericalCheckResult(
                        claim_number=claim_raw,
                        source_number=str(claim_float),
                        match=True,
                        chunk_id=chunk.chunk_id,
                    ))

    if not conflict_found:
        if not is_range_claim and claim_floats:
            if all(cf in matched_floats for cf in claim_floats):
                conflict_citation = None
            else:
                if conflict_citation is not None:
                    conflict_found = True

    # For range claim: conflict if NO chunk value falls within the range
    # Only run if no conflict was already found (e.g. by the year check)
    if is_range_claim and not conflict_found:
        within_range_matches = [c for c in checks if c.match]
        if within_range_matches:
            conflict_found = False
        elif checks:
            conflict_found = True

    return Tier25Result(
        has_conflict=conflict_found,
        verification_method="tier25_numerical",
        evidence_bundle=evidence_bundle,
        conflict_citation=conflict_citation,
        numerical_checks=checks,
    )
