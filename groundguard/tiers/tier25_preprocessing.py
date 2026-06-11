"""Tier 2.5 — Numerical consistency pre-check."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groundguard.models.internal import VerificationContext
    from groundguard.loaders.chunker import Chunk

from groundguard.models.result import Citation

# Configuration Constants
APPROX_TOLERANCE: float = 0.10
APPROX_ZERO_ABS_TOLERANCE: float = 0.10
UNIT_ANCHOR_WINDOW_TOKENS: int = 2
HEDGE_SCAN_TOKENS: int = 5
RHETORICAL_SCAN_TOKENS: int = 3
MULTI_NUMBER_SKIP_THRESHOLD: int = 1
EU_INTEGER_MIN_LEAD_DIGITS: int = 1
EU_INTEGER_DECIMAL_DIGITS: int = 3
RANGE_CONTAINMENT_STRICT: bool = True
ABBREVIATED_YEAR_CENTURY_THRESHOLD: int = 100

# Matches: 30%, 300%, $4.2M, $300, 1,000,000, 4.2, 2023, -5%, -$4.2M
_CURRENCY_PREFIX = r'(?:[$€£¥₹₩₽]|(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|HKD)\s*)'
_MAGNITUDE_SUFFIX = r'(?:[MBKTmbkt](?:illion)?|bps?|basis\s+points?)?'

_NUMBER_PATTERN = re.compile(
    r'(?<!\w)'
    r'[+\-]?'
    r'(?:' + _CURRENCY_PREFIX + r')?'
    r'(?:'
        r'\d[\d,]*(?:\.\d+)?[eE][+\-]?\d+'
        r'|'
        r'\d[\d,]*(?:\.\d+)?'
    r')'
    r'(?:%|' + _MAGNITUDE_SUFFIX + r')?'
    r'(?!\w)'
)

_STOPWORDS = {"the", "a", "an", "and", "or", "in", "of", "to", "is", "was", "be", "see", "for",
              "section", "details", "reference", "per", "at", "by", "with", "that", "this",
              "which", "from", "are", "has", "have", "had", "not", "do", "does", "shall",
              "will", "would", "may", "can", "its", "it", "we", "our"}

# Year ONLY in temporal context: "in 2023", "for 2024", "Q3 2023", "FY2024", etc.
_YEAR_CONTEXT_PATTERN = r'(?:in|for|during|as of|fiscal|FY|Q[1-4])\s+(\d{4})\b'

# Gate 1 - Surface Regex Blocklist
_GATE1_PATTERNS = [
    r'^\s*\d+[.)]\s',
    r'^\s*\(\d+\)\s',
    r'\[\d+\]',
    r'\b(?i:section|sec\.?|ch\.?|chapter|appendix|part)\s*\d+[\d.]*',
    r'\b(?i:fig\.?|figure|table|tbl\.?|chart|diagram|exhibit)\s*\d+[\d.]*',
    r'\b(?i:eq\.?|equation|formula|theorem|lemma|corollary|prop\.?)\s*\d+',
    r'\b(?i:step|stage|passage|option|choice|item|entry|note)\s*\d+(?:\s+of\s+\d+)?',
    r'\b(?i:line|row|col\.?|column|page|pp?\.)\s*\d+',
    r'\b(?i:problem|exercise|question|q\.?)\s*\d+',
    r'(?i:phase|stage|part|chapter|study|world\s+war|act|scene)\s+[IVXLCDM]+\b',
    r'\b[IVXLCDM]+\s+(?i:phase|stage|trial|study)\b',
    r'\b\d+\s*(?:st|nd|rd|th)\b',
    r'\bv\d+(\.\d+)+\b',
    r'\b(?i:version)\s+\d[\d.]*\b',
    r'\b\d+\.\d+\.\d+\b',
    r'\b(?i:covid|sars|mers|h\d+n\d*|dsm|icd|hiv)-\d+[-\w]*\b',
    r'\b(?:[A-Z]{1,5}-\d+|(?!In\b|For\b|On\b|At\b|By\b|With\b|From\b|During\b|Fiscal\b|FY\b|Q[1-4]\b)[A-Z][a-zA-Z0-9]{2,10}\s+\d+)\b',
    r'\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    r'\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b',
    r'\d+\.?\d*°\s*[NSEWnsew]\b',
    r'\b(?i:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|'
    r'dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b(?!\s*,?\s*\d{4})',
    r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|AM|PM|UTC|GMT|EST|PST)?\b',
    r'\b\d{1,2}\s*(?:am|pm|AM|PM)\b',
]
_GATE1_BLOCKLIST = [re.compile(p, flags=re.MULTILINE) for p in _GATE1_PATTERNS]
_COMBINED_GATE1 = re.compile("|".join(_GATE1_PATTERNS), flags=re.MULTILINE)


def mask_structural(text: str) -> str:
    """Mask structural elements (like chapter headings, step numbers, etc.) with spaces."""
    if not re.search(r'\d', text):
        return text
    return _COMBINED_GATE1.sub(lambda m: ' ' * len(m.group(0)), text)


# Pre-Pass B - Composite Number Extraction constants
_SCALE_MAP = {
    "hundred": 1e2,
    "thousand": 1e3, "k": 1e3,
    "million": 1e6,  "m": 1e6,
    "billion": 1e9,  "b": 1e9,
    "trillion": 1e12, "t": 1e12,
}
_DIGIT_SCALE_PATTERN = re.compile(
    r'(\d[\d,]*(?:\.\d+)?)\s+(hundred|thousand|million|billion|trillion)\b',
    re.IGNORECASE
)
_WORD_NUMS = {
    "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "half a": 0.5, "half": 0.5,
    "quarter": 0.25, "quarter of a": 0.25, "a quarter of a": 0.25, "a quarter": 0.25,
    "three-quarters": 0.75, "three quarters": 0.75, "three quarters of a": 0.75,
}
_VERBAL_SCALE_PATTERN = re.compile(
    r'\b('
    r'a|one|two|three|four|five|six|seven|eight|nine|ten|'
    r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
    r'(?:twenty|thirty|forty|fourty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|'
    r'half a|half|'
    r'quarter|quarter of a|a quarter of a|a quarter|three-quarters|three quarters|three quarters of a'
    r')\s+'
    r'(hundred|thousand|million|billion|trillion)'
    r'(?:\s+(thousand|million|billion|trillion))?'
    r's?\b',
    re.IGNORECASE
)
_VAGUE_QUANTIFIER_PATTERN = re.compile(
    r'\b(?:several|'
    r'dozens?\s+of|hundreds\s+of|thousands\s+of|'
    r'millions\s+of|billions\s+of|tens\s+of|'
    r'a?\s*couple\s+of|a\s+dozen)\b',
    re.IGNORECASE
)


def _parse_verbal_word(word: str) -> float:
    """Convert number words like 'twenty-three' into float value."""
    w = word.lower()
    if w in _WORD_NUMS:
        return _WORD_NUMS[w]
    parts = re.split(r'[-\s]+', w)
    return sum(_WORD_NUMS[p] for p in parts if p in _WORD_NUMS)


def extract_composite_numbers_with_indices(text: str) -> tuple[list[tuple[float, str, int]], str]:
    """Extract composites with their character start index and remaining text."""
    results = []
    modified_text = text
    # Extract digit scale composites
    while True:
        match = _DIGIT_SCALE_PATTERN.search(modified_text)
        if not match:
            break
        digit_part = match.group(1).replace(',', '')
        scale_word = match.group(2).lower()
        value = float(digit_part) * _SCALE_MAP[scale_word]
        raw_span = match.group(0)
        start, end = match.span()
        results.append((value, raw_span, start))
        modified_text = modified_text[:start] + (' ' * len(raw_span)) + modified_text[end:]
        
    # Extract verbal scale composites
    while True:
        match = _VERBAL_SCALE_PATTERN.search(modified_text)
        if not match:
            break
        group1 = match.group(1)
        group2 = match.group(2).lower()
        group3 = match.group(3)
        value = _parse_verbal_word(group1) * _SCALE_MAP[group2]
        if group3:
            value *= _SCALE_MAP[group3.lower()]
        raw_span = match.group(0)
        start, end = match.span()
        results.append((value, raw_span, start))
        modified_text = modified_text[:start] + (' ' * len(raw_span)) + modified_text[end:]
        
    # Return raw results with indices and the modified text
    return results, modified_text


def extract_composite_numbers(text: str) -> tuple[list[tuple[float, str]], str]:
    """Return list of (value_float, raw_span) and text with those spans replaced by spaces of equal length."""
    raw_results, modified_text = extract_composite_numbers_with_indices(text)
    # Sort the raw results by their start index
    sorted_results = sorted(raw_results, key=lambda x: x[2])
    # Map 3-tuples to 2-tuples containing only value and raw span
    results = [(val, raw) for val, raw, _ in sorted_results]
    return results, modified_text


# Pre-Pass C - Accounting Negative Normalization constants
_ACCT_NEG_PATTERN = re.compile(
    r'\(\s*'
    r'(?:'
    r'[$€£¥₹₩₽]\s*\d[\d,]*(?:\.\d+)?[MBKTmbkt]?'
    r'|\d[\d,]*(?:\.\d+)?[MBKTmbkt]'
    r'|\d{1,3}(?:,\d{3})+(?:\.\d+)?'
    r')'
    r'\s*\)'
)


def normalize_accounting_negatives(text: str) -> str:
    """Convert accounting negatives like (1,234.56) to standard format -1234.56."""
    def replace(match):
        val = match.group(0)
        # Strip outer parentheses and spaces, and remove commas
        inner = val[1:-1].strip().replace(',', '')
        return '-' + inner
    return _ACCT_NEG_PATTERN.sub(replace, text)


# Pre-Pass D - European Number Normalization constants
_EU_NUMBER_PATTERN = re.compile(r'\b\d{1,3}(?:\.\d{3})+,\d+\b')
_EU_UNGROUPED_DECIMAL_RE = re.compile(r'\b(\d+),(\d{2})\b(?!,\d)')


def normalize_eu_numbers(text: str) -> str:
    """Normalize European formatted numbers (1.234,56 -> 1234.56)."""
    # Pass 1: grouped EU format using _EU_NUMBER_PATTERN
    def replace_grouped(match):
        val = match.group(0)
        return val.replace('.', '').replace(',', '.')
    
    text = _EU_NUMBER_PATTERN.sub(replace_grouped, text)
    
    # Pass 2: ungrouped EU decimal using _EU_UNGROUPED_DECIMAL_RE
    text = _EU_UNGROUPED_DECIMAL_RE.sub(r'\1.\2', text)
    
    return text



def _normalise_number(raw: str) -> float:
    if raw is None:
        raise TypeError("Input must be a string")
    if not isinstance(raw, str):
        raise TypeError("Input must be a string")
    raw = raw.strip()
    if not raw:
        raise ValueError("Input string cannot be empty")
    sign = 1.0
    if raw.startswith('-'):
        sign = -1.0
        raw = raw[1:].strip()
    elif raw.startswith('+'):
        sign = 1.0
        raw = raw[1:].strip()
    is_usd_style_prefix = False
    if raw.startswith('$'):
        is_usd_style_prefix = True
        raw = raw[1:].strip()
    else:
        non_usd_symbols = ('€', '£', '¥', '₹', '₩', '₽')
        for sym in non_usd_symbols:
            if raw.startswith(sym):
                raw = raw[len(sym):].strip()
                break
        else:
            iso_codes = ('USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'HKD')
            for iso in iso_codes:
                if raw.startswith(iso):
                    raw = raw[len(iso):].strip()
                    break
    if raw.endswith('%'):
        raw = raw[:-1].strip()
    is_bps = False
    bps_match = re.search(r'\s*(?:bps?|basis\s+points?)$', raw, re.IGNORECASE)
    if bps_match:
        is_bps = True
        raw = raw[:bps_match.start()].strip()
    scale_factor = 1.0
    mag_match = re.search(r'\s*([MBKTmbkt])(?:illion)?$', raw, re.IGNORECASE)
    if mag_match:
        char = mag_match.group(1).lower()
        scale_factor = _SCALE_MAP.get(char, 1.0)
        raw = raw[:mag_match.start()].strip()
    if re.match(r'^\d{1,3}(?:\.\d{3})+\,\d+$', raw):
        raw = raw.replace('.', '').replace(',', '.')
    elif ',' in raw and '.' not in raw and not re.search(r'\,\d{3}$', raw):
        raw = raw.replace(',', '.')
    else:
        raw = raw.replace(',', '')
    val = float(raw)
    val = val * scale_factor
    if is_bps:
        val = val / 100.0
    val = val * sign
    if val == 0.0:
        val = 0.0
    return val


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
    # Preprocess claim
    claim_text = ctx.claim
    claim_text = mask_structural(claim_text)
    claim_text = normalize_accounting_negatives(claim_text)
    claim_text = normalize_eu_numbers(claim_text)
    claim_composites_with_indices, claim_remaining = extract_composite_numbers_with_indices(claim_text)
    claim_numbers_raw = re.findall(_NUMBER_PATTERN, claim_remaining)
    
    # Check if there are no numbers at all
    if not claim_composites_with_indices and not claim_numbers_raw:
        return Tier25Result(has_conflict=False, evidence_bundle=build_evidence_bundle(ctx, chunks))

    # Guard: insufficient metric context
    if not _has_sufficient_metric_context(ctx.claim):
        return Tier25Result(has_conflict=False, evidence_bundle=build_evidence_bundle(ctx, chunks))

    # Construct claim numbers list: (value_float, raw_string, start_idx)
    claim_all = []
    for val, raw, start in claim_composites_with_indices:
        claim_all.append((val, raw, start))
        
    for match in re.finditer(_NUMBER_PATTERN, claim_remaining):
        raw = match.group(0)
        try:
            val = float(_normalise_number(raw))
            claim_all.append((val, raw, match.start()))
        except ValueError:
            pass
            
    # Sort claim numbers by their start index to keep left-to-right order
    claim_all.sort(key=lambda x: x[2])
    claim_numbers = [(val, raw) for val, raw, _ in claim_all]
    
    # Extract claim floats
    claim_floats = [val for val, _ in claim_numbers]
    
    # Extract contextual years from claim (using original ctx.claim)
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
        # Preprocess chunk
        chunk_text = chunk.text_content
        chunk_text = mask_structural(chunk_text)
        chunk_text = normalize_accounting_negatives(chunk_text)
        chunk_text = normalize_eu_numbers(chunk_text)
        chunk_composites_with_indices, chunk_remaining = extract_composite_numbers_with_indices(chunk_text)
        
        # Construct chunk numbers list: (value_float, raw_string, start_idx)
        chunk_all = []
        for val, raw, start in chunk_composites_with_indices:
            chunk_all.append((val, raw, start))
            
        for match in re.finditer(_NUMBER_PATTERN, chunk_remaining):
            raw = match.group(0)
            try:
                val = float(_normalise_number(raw))
                chunk_all.append((val, raw, match.start()))
            except ValueError:
                pass
                
        # Sort chunk numbers by start index
        chunk_all.sort(key=lambda x: x[2])
        chunk_numbers = [(val, raw) for val, raw, _ in chunk_all]
        chunk_floats = [val for val, _ in chunk_numbers]

        # For each claim number, check against chunk numbers
        # Skip year values — they are exclusively handled by the year conflict loop above.
        claim_year_floats = {float(y) for y in claim_years}
        for claim_float, claim_raw in claim_numbers:
            if claim_float in claim_year_floats:
                continue

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
                        first_chunk_raw = chunk_numbers[0][1] if chunk_numbers else ""
                        checks.append(NumericalCheckResult(
                            claim_number=claim_raw,
                            source_number=first_chunk_raw,
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
                    chunk_raw = chunk_numbers[0][1] if chunk_numbers else ""
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
                    # Find the chunk's raw string for this claim_float
                    chunk_raw = ""
                    for val, raw in chunk_numbers:
                        if val == claim_float:
                            chunk_raw = raw
                            break
                    checks.append(NumericalCheckResult(
                        claim_number=claim_raw,
                        source_number=chunk_raw if chunk_raw else str(claim_float),
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
