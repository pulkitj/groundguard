# Tier25 Number Extraction — Specification and Implementation Plan

## Background

Tier25 is a pure regex-based numerical pre-check that runs before any LLM call. If it finds a numerical conflict between the claim and source documents, it short-circuits and returns CONTRADICTED. A false positive here permanently blocks the claim from reaching Tier 3.

The benchmark analysis (`phase1_v013_rerun.jsonl`) identified two root causes of false positives:
1. Structural/ordinal numbers being extracted as metric values (14 of 16 Tier25 FPs)
2. Sign, magnitude, and format mismatches causing wrong numeric comparisons

This document specifies the complete fix.

---

## Research Foundation

The specification is grounded in three research passes:

- **Structural/factual distinction**: CQE (EMNLP 2023, arxiv 2305.08853), CogComp Quantifier (Roy et al., TACL 2015), OntoNotes NER annotation guidelines, ClarityNLP, VitaminC fact-checking benchmark
- **Financial number formats**: Duckling (Facebook), Quantulum3, Microsoft Recognizers-Text, QuanTemp (SIGIR 2024), ClaimIQ (CheckThat! 2025, arxiv 2509.11492)
- **Gap analysis**: "Representing Numbers in NLP: a Survey and a Vision" (arxiv 2103.13136), "Quantitative Information Extraction from Humanitarian Documents" (arxiv 2408.04941)

**Core principle (from CQE)**: The decision is made at the head noun, not the number. A number is factual if its governing noun describes a real-world observable entity or property. If the noun describes a document position, schema slot, or rhetorical element, the number is structural regardless of its value.

---

## Complete Case Inventory

Every case identified across all research passes:

| # | Case | Status | Gate |
|---|---|---|---|
| 1 | List markers `1.`, `2)`, `(1)` at line start | Structural → block | Gate 1 |
| 2 | Citation brackets `[1]`, `[23]` | Structural → block | Gate 1 |
| 3 | Document labels: Section/Ch./Appendix/Part N | Structural → block | Gate 1 |
| 4 | Figure/table: Fig./Figure/Table/Tbl./Chart/Exhibit N | Structural → block | Gate 1 |
| 5 | Math labels: Eq./Theorem/Lemma/Corollary N | Structural → block | Gate 1 |
| 6 | Step/Stage/Passage/Option/Choice/Item N | Structural → block | Gate 1 |
| 7 | Line/Row/Column/Page N | Structural → block | Gate 1 |
| 8 | Ordinal suffixes: 1st, 2nd, 42nd | Structural → block | Gate 1 |
| 9 | Roman numeral labels: Phase II, Chapter IV | Structural → block | Gate 1 |
| 10 | Version strings: v2.0, v1.3.2, 3.14.1 | Structural → block | Gate 1 |
| 11 | Disease/taxonomy IDs: COVID-19, H1N1, DSM-5 | Structural → block | Gate 1 |
| 12 | Product/model IDs: F-35, B-52, AK-47 | Structural → block | Gate 1 |
| 13 | Phone numbers: (555) 867-5309, +1-800-555-0100 | Structural → block | Gate 1 |
| 14 | IPv4 addresses: 192.168.0.1 | Structural → block | Gate 1 |
| 15 | IPv6 addresses | Structural → block | Gate 1 |
| 16 | Geographic coordinates: 40.7128° N | Structural → block | Gate 1 |
| 17 | Bare month+day without year: January 4 | Structural → block | Gate 1 |
| 18 | Time expressions: 3pm, 9:00 AM, 14:30 UTC | Ambiguous → block/escalate | Gate 1 |
| 19 | Rhetorical nouns: "3 reasons", "5 ways", "2 points" | Structural → skip | Gate 2 |
| 20 | Quantities with physical units: 20 kg, 4.2 km, 95% | Factual → fast-accept | Gate 3 |
| 21 | Financial values: $400M, €1,234, 30% revenue | Factual → fast-accept | Gate 3 |
| 22 | Concrete entity counts: 3 locations, 20 amino acids | Factual → fast-accept | Gate 3 |
| 23 | Age expressions: 42-year-old, aged 42 | Factual → fast-accept | Gate 3 |
| 24 | Composite word multipliers: "3 million", "2.5 billion" | Factual → pre-pass extract | Pre-Pass B |
| 25 | Verbalized numbers: "a million jobs", "half a billion" | Factual → pre-pass extract | Pre-Pass B |
| 26 | Accounting negatives: ($1,234.56) = negative | Normalize → -1234.56 | Pre-Pass C |
| 27 | European number format: 1.234,56 | Normalize in-place → 1234.56 | Pre-Pass D |
| 28 | Fractions: "1/3 of patients", "two-thirds" | Gate 3 fast-accept if entity noun, else escalate | Pre-Pass E |
| 29 | Non-$ currencies: €, £, ¥, ₹, ₩, ISO codes | Factual → updated regex | Phase 2 |
| 30 | Sign prefix: +5%, -3% (direction matters) | Preserve sign | Phase 2 |
| 31 | Basis points: 50bps, 50bp | Factual → updated regex + normalize | Phase 2 |
| 32 | Scientific notation: 1.5e6, 2.3×10^9 | Factual → updated regex | Phase 2 |
| 33 | Fuzzy lower bound: "at least 100", "over $5M" | No conflict if source ≥ claim | Phase 4 |
| 34 | Fuzzy upper bound: "fewer than 5", "under 30%" | No conflict if source ≤ claim | Phase 4 |
| 35 | Fuzzy approximate: "almost 32%", "about 20 people" | ±10% tolerance | Phase 4 |
| 36 | Hyphenated ranges: $10M–$20M, 50–60 patients | Both bounds must be grounded | Phase 5 |
| 37 | Word-form ranges: "20 to 30 percent", "between 5 and 10" | Both bounds must be grounded | Phase 5 |
| 38 | X-times/ratio: 3x growth, 2:1 ratio | Cannot verify → escalate | Phase 6 |
| 39 | Decade references: "in the 1980s", "the 90s" | Cannot verify precisely → escalate | Phase 6 |
| 40 | Score/index: "8 out of 10", "a score of 74" | Cannot verify without scale → escalate | Phase 6 |
| 41 | Percentage points vs percent: "rose 3pp" ≠ "rose 3%" | Tag unit; escalate if mismatch | Phase 6 |
| 42 | Per-unit compound rates: $5/share, 20mg/day | Factual → Gate 3 compound anchor | Phase 3 |
| 43 | The `len(chunk_floats) > 1` skip rule | Recalibrate after Phase 1 data | Phase 7 |
| 44 | Abbreviated year ranges: 2025-26, FY2024-25 | Expand to full range → escalate | Phase 6 |

---

## Architecture

The pipeline is **mostly symmetric**: Pre-Passes A–D and Gate 3 run identically on both claim text and source chunk text. Two steps are **asymmetric** by design:

| Step | Claim | Source | Reason |
|---|---|---|---|
| Pre-Pass A (Gate 1) | mask structural spans | mask structural spans | Source docs also contain citations, section labels, etc. |
| Pre-Pass B (composite numbers) | extract + remove spans | extract + remove spans | Source can have "3 million employees" too |
| Pre-Pass C (accounting negatives) | normalize | normalize | Financial tables appear in source docs |
| Pre-Pass D (EU format) | normalize | normalize | Source can be EU-authored |
| **Pre-Pass E (fractions)** | remove span + **escalate** | remove span + **normalize to decimal** | Claim fraction vs source % is ambiguous; source fraction is just a value to compare |
| **Gate 2 (rhetorical nouns)** | **skip number if rhetorical** | **not applied** | Source is ground truth — its numbers are trusted regardless of head noun |
| Gate 3 (unit-anchor fast-accept) | produce NumericalValue | produce NumericalValue | Unit labels needed on both sides for mismatch detection |
| Hedge detection | applied | not applied | Source numbers are ground truth |
| Vague quantifier escalation | applied | not applied | Only claim asserts things |

```
Claim text                              Source chunk text
     │                                       │
[PRE-PASS A] mask structural           [PRE-PASS A] mask structural
     │                                       │
[PRE-PASS B] extract composites        [PRE-PASS B] extract composites
     │                                       │
[PRE-PASS C] normalize acct-neg        [PRE-PASS C] normalize acct-neg
     │                                       │
[PRE-PASS D] normalize EU format       [PRE-PASS D] normalize EU format
     │                                       │
[PRE-PASS E] fractions →               [PRE-PASS E] fractions →
   remove span + ESCALATE                 remove span + normalize to decimal
     │                                       │       (no escalation)
[GATE 2] rhetorical noun filter        (GATE 2 NOT APPLIED)
   discard if head noun in                  │
   _RHETORICAL_NOUNS                        │
     │                                       │
[BASE EXTRACT] _NUMBER_PATTERN         [BASE EXTRACT] _NUMBER_PATTERN
     │                                       │
[GATE 3] _extract_unit_anchor          [GATE 3] _extract_unit_anchor
   → NumericalValue(value, unit)          → NumericalValue(value, unit)
     │                                       │
     └───────────── COMPARISON LAYER ────────┘
              (claim NumericalValues vs source NumericalValues)
                         │
              [HEDGE]  detect lower/upper/approx modifier
                       in claim text; adjust comparison rule
                         │
              [VAGUE]  detect vague quantifiers in claim;
                       escalate immediately if found
```

**Why Gate 2 is claim-only**: source documents provide ground truth values. A source sentence "there are 3 main reasons revenue fell" contains a rhetorical "3", but filtering it out of the source would hide a real number that might conflict with a factual claim number. Leaving it in is safer: if the claim number carries a recognized unit and the source "3" carries none, the `unit_unitless_mismatch` escalation routes to Tier 3 for resolution. Only on the claim side does rhetorical filtering unambiguously reduce false positives.

**Why Pre-Pass E is asymmetric**: a claim saying "1/3 of patients recovered" compared against a source saying "33% recovered" represents equivalent facts, but `0.333 ≠ 33.0`. Escalating the claim fraction ensures Tier 3 handles the equivalence. On the source side, a fraction is simply a ratio value (`0.333`) — if the claim says "0.33" the comparison succeeds; if the claim says "33%" the unit mismatch (`%` vs `None`) escalates correctly without needing a special fraction-escalation rule.

---

## Gate 1: Surface Regex Blocklist

Applied to **both claim and source text** before extraction. Matched spans are replaced with spaces so the base regex cannot see them.

```python
_GATE1_BLOCKLIST = [

    # List markers
    # No (?m) flag here — pass re.MULTILINE to re.compile() instead (see performance note)
    r'^\s*\d+[.)]\s',                             # "1. item", "2) item"
    r'^\s*\(\d+\)\s',                             # "(1) item" at line start
    r'\[\d+\]',                                   # [1], [23] citation brackets

    # Document-navigation labels (Arabic numerals)
    # No (?i) flag here — use (?i:...) local groups so flag cannot bleed into other branches
    r'\b(?i:section|sec\.?|ch\.?|chapter|appendix|part)\s*\d+[\d.]*',
    r'\b(?i:fig\.?|figure|table|tbl\.?|chart|diagram|exhibit)\s*\d+[\d.]*',
    r'\b(?i:eq\.?|equation|formula|theorem|lemma|corollary|prop\.?)\s*\d+',
    r'\b(?i:step|stage|passage|option|choice|item|entry|note)\s*\d+',
    r'\b(?i:line|row|col\.?|column|page|pp?\.)\s*\d+',
    r'\b(?i:problem|exercise|question|q\.?)\s*\d+',

    # Roman numeral structural labels
    # (?i:...) scopes case-insensitivity to trigger words only; [IVXLCDM]+ stays
    # uppercase-only to avoid colliding with words like "mix", "mid", "did".
    r'(?i:phase|stage|part|chapter|study|world\s+war|act|scene)\s+[IVXLCDM]+\b',
    r'\b[IVXLCDM]+\s+(?i:phase|stage|trial|study)\b',

    # Ordinal suffixes
    r'\b\d+\s*(?:st|nd|rd|th)\b',

    # Version / software strings
    r'\bv\d+(\.\d+)+\b',
    r'\b(?i:version)\s+\d[\d.]*\b',
    r'\b\d+\.\d+\.\d+\b',                        # semantic triples: 3.14.1

    # Disease / taxonomy identifiers
    r'\b(?i:covid|sars|mers|h\d+n\d*|dsm|icd|hiv)-\d+[-\w]*\b',

    # Product / model identifiers
    r'\b[A-Z]{1,5}-\d+\b',                        # F-35, GPT-4, B-52

    # Phone numbers
    r'\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',

    # IP addresses
    r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',   # IPv4
    r'\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b',  # IPv6

    # Geographic coordinates
    r'\d+\.?\d*°\s*[NSEWnsew]\b',

    # Bare month+day without year (not factual as standalone)
    r'\b(?i:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|'
    r'dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b(?!\s*,?\s*\d{4})',

    # Time expressions
    r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|AM|PM|UTC|GMT|EST|PST)?\b',
    r'\b\d{1,2}\s*(?:am|pm|AM|PM)\b',

    # ── Decade references (escalate, not block — see §Escalation) ─────────────
    # Not blocked here; handled separately in comparison logic.
]
```

**Performance implementation**: Compile all Gate 1 patterns into a **single combined `re.Pattern`** at module load time:

```python
_COMBINED_GATE1 = re.compile("|".join(_GATE1_BLOCKLIST), flags=re.MULTILINE)
```

Guard the masking call with a fast digit pre-check: `if not re.search(r'\d', text): return text`. Do not run 24+ sequential `re.sub` calls per chunk at runtime.

**Critical flag rules** (Python 3.11+ enforced):
- **No `(?i)` or `(?m)` inline global flags** anywhere in `_GATE1_BLOCKLIST`. In Python 3.11+, a global inline flag not at the very start of the combined pattern raises `re.error: global flags not at the start of the expression` at import time. In older Python, global flags applied inside one branch silently bleed into the entire combined pattern — making the Roman numeral `[IVXLCDM]+` restriction case-insensitive and defeating the collision fix.
- **Use `(?i:...)` local groups** for all case-insensitive matching within individual patterns (already done above).
- **Pass `re.MULTILINE`** to `re.compile()` instead of using `(?m)` in individual patterns. This correctly scopes `^`/`$` to line boundaries for the list-marker patterns.

**Implementation note on `(N)` parenthetical**: `\(\d+\)` is blocked when the number stands alone inside parentheses (surrounded by whitespace/word boundaries/sentence boundary). Do not block quantity expression like `(4.2 kg)` or `(25% stake)` or `(20.5M)` — the unit makes it factual.

---

## Pre-Pass B: Composite Number Extraction

Runs before the base regex. Extracts multi-token quantity expressions as single normalized floats and removes the matched spans from the working text copy so the base regex does not re-extract the bare digit.

### Pattern A — digit + scale word

```python
_SCALE_MAP = {
    "hundred": 1e2,
    "thousand": 1e3, "k": 1e3,
    "million": 1e6,  "m": 1e6,
    "billion": 1e9,  "b": 1e9,
    "trillion": 1e12, "t": 1e12,
}
_DIGIT_SCALE_PATTERN = r'(\d[\d,]*(?:\.\d+)?)\s+(hundred|thousand|million|billion|trillion)\b'
# "1.2 million" → 1_200_000.0
# "2.5 billion" → 2_500_000_000.0
# "$3 million" → 3_000_000.0 (currency prefix consumed separately)
```

### Pattern B — verbalized numbers (no digit)

```python
_WORD_NUMS = {
        # Units & Teens
    "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,

    # Tens (Base values; compounds like "twenty-one" can be parsed by summing parts)
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,

    # Fractions / Parts
    "half a": 0.5, "half": 0.5,
    "quarter": 0.25, "quarter of a": 0.25, "a quarter of a": 0.25, "a quarter": 0.25,
    "three-quarters": 0.75, "three quarters": 0.75, "three quarters of a": 0.75,

    # Vague / Approximate (None signals: escalate to LLM, do not compare)
    "several": None,
    "dozens of": None,
    "hundreds of": None,
    "thousands of": None,
    "millions of": None,
    "billions of": None,
    "tens of": None,
    "a few": None,
    "few": None,
    "some": None,
    "many": None,
    "couple": None,
    "couple of": None,
    "a couple": None,
    "a couple of": None,
    "a dozen": None,
    "dozens": None,
}
_VERBAL_SCALE_PATTERN = (r'\b('
    r'a|one|two|three|four|five|six|seven|eight|nine|ten|'
    r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
    r'(?:twenty|thirty|forty|fourty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|'
    r'half a|half|'
    r'quarter|quarter of a|a quarter of a|a quarter|three-quarters|three quarters|three quarters of a'
    r')\s+'
    r'(hundred|thousand|million|billion|trillion)'
    r'(?:\s+(thousand|million|billion|trillion))?'   # optional second scale word
    r's?\b')
# "a million jobs"            → 1_000_000.0
# "half a billion"            → 500_000_000.0
# "three hundred thousand"    → 300_000.0   (word × scale1 × scale2)
# "two hundred million"       → 200_000_000.0
#
# Normalization rule for double-scale: when group(3) is present,
# result = _WORD_NUMS[group(1)] * _SCALE_MAP[group(2)] * _SCALE_MAP[group(3)]
# Single-scale (group(3) absent): result = _WORD_NUMS[group(1)] * _SCALE_MAP[group(2)]
# Only "hundred" is valid as scale1 in double-scale (e.g. "two thousand million" is
# technically valid but extremely rare; "two million billion" is not standard English).
# Implementation may restrict double-scale to scale1 == "hundred" for safety.
#
# Compound-word lookup: group(1) may be a compound like "twenty-five". Do NOT do
# _WORD_NUMS[group(1)] directly — that raises KeyError for any hyphenated/spaced tens.
# Use a helper:
#
#   def _parse_verbal_word(word: str) -> float:
#       if word in _WORD_NUMS:
#           return _WORD_NUMS[word]
#       # Split on hyphen or space and sum parts (e.g. "twenty-five" → 20 + 5)
#       parts = re.split(r'[-\s]+', word)
#       return sum(_WORD_NUMS[p] for p in parts if p in _WORD_NUMS)
#
# Multi-scale verbal split guard: if the claim yields two separate verbal-scale
# matches separated only by "and" or whitespace (e.g. "one hundred and fifty thousand"
# → ["one hundred", "fifty thousand"]), do not compare them as independent numbers.
# Escalate the claim: set escalate_reason="verbal_compound_split".
```

**Design note**: The vague quantifiers (`several`, `dozens of`, `millions of`, etc.) are removed from `_VERBAL_SCALE_PATTERN` because that pattern requires a scale word as the second group — but vague quantifiers are typically followed by concrete nouns ("millions of jobs", "several patients"), not scale words. They must be detected by a **separate pattern** before composite extraction and routed directly to escalation:

```python
# Vague quantifiers: detected before composite extraction and escalated immediately.
# These do NOT go through _VERBAL_SCALE_PATTERN (which requires a scale-word complement).
#
# "some", "few", "many" are intentionally excluded: they are extremely common
# determiners in non-quantifier contexts ("5 stores in some countries", "many other
# factors"). Including them triggers immediate escalation for the entire claim, bypassing
# Tier25 comparison for the real factual numbers present. Only include quantifiers that
# are structurally tied to a scale word or explicit quantity (dozens of, hundreds of, etc.)
# or that cannot appear as bare attributive adjectives (several, a couple of).
_VAGUE_QUANTIFIER_PATTERN = re.compile(
    r'\b(?:several|'
    r'dozens?\s+of|hundreds\s+of|thousands\s+of|'
    r'millions\s+of|billions\s+of|tens\s+of|'
    r'a?\s*couple\s+of|a\s+dozen)\b',
    re.IGNORECASE
)
# If _VAGUE_QUANTIFIER_PATTERN matches in the claim, set escalate_reason="vague_quantifier"
# and return early without numeric comparison.
```

Updated `_WORD_NUMS` — remove the None-valued vague quantifiers (they are now handled by `_VAGUE_QUANTIFIER_PATTERN`):

```python
_WORD_NUMS = {
    # Units & Teens
    "a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    # Fractions / Parts (Pre-Pass B handles these)
    "half a": 0.5, "half": 0.5,
    "quarter": 0.25, "quarter of a": 0.25, "a quarter of a": 0.25, "a quarter": 0.25,
    "three-quarters": 0.75, "three quarters": 0.75, "three quarters of a": 0.75,
}
```

---

## Pre-Pass C: Accounting Negative Normalization

Bare `(5%)` in narrative prose is NOT an accounting negative — it is a positive percentage in a parenthetical aside. The pattern uses three tiers of evidence, from most to least certain:

```python
_ACCT_NEG_PATTERN = (
    r'\(\s*'
    r'(?:'
    # Tier 1 — currency-prefixed: unambiguous financial negative
    r'[$€£¥₹₩₽]\s*\d[\d,]*(?:\.\d+)?[MBKTmbkt]?'
    # Tier 2 — magnitude-suffixed without currency: (500M), (1.2B), (3.4T)
    r'|\d[\d,]*(?:\.\d+)?[MBKTmbkt]'
    # Tier 3 — comma-grouped integers/decimals: (1,234) or (1,234.56)
    # Requires at least one thousands-separator comma — rules out bare (5) and (12)
    # which are common in narrative prose as list markers or emphasis.
    r'|\d{1,3}(?:,\d{3})+(?:\.\d+)?'
    r')'
    r'\s*\)'
)
# Matches:  "($4.2M)"     → "-$4.2M"
#           "(€1,234)"    → "-€1,234"
#           "(500M)"      → "-500M"
#           "(1,234.56)"  → "-1234.56"   ← financial table bare negative
#           "(1,234)"     → "-1234"      ← financial table integer
# Does NOT match: "(5%)", "(1.5%)", "(15)", "(12)" — narrative/ambiguous forms
```

---

## Pre-Pass D: European Number Normalization

Runs after Pre-Pass C, before composite extraction. Detects unambiguous EU-format numbers (`1.234,56` — dot grouping separator, comma decimal separator) and rewrites them to US format in the working text copy. This allows the base regex to match them correctly without needing a permissive `[\d,.]` character class that would create US/EU ambiguity.

```python
# Unambiguous EU pattern: 1–3 digits, then one or more groups of exactly 3 digits
# separated by dots, then a comma-decimal part.
# e.g. 1.234,56  /  1.234.567,89  /  12.345,6
_EU_NUMBER_PATTERN = r'\b\d{1,3}(?:\.\d{3})+,\d+\b'

def normalize_eu_numbers(text: str) -> str:
    """Rewrite EU-format numbers to US format before base extraction."""
    def _swap(m: re.Match) -> str:
        s = m.group(0).replace('.', '')   # drop grouping dots: "1234,56"
        s = s.replace(',', '.')            # swap decimal comma:  "1234.56"
        return s
    return re.sub(_EU_NUMBER_PATTERN, _swap, text)
```

The pattern `\d{1,3}(?:\.\d{3})+,\d+` is unambiguous: it requires **at least one** three-digit group after a dot, which rules out decimal points (`1.5`, `3.14`) and version strings (`1.2.3` — no comma after). After rewriting, the base regex sees only US-format numbers.

**No-grouping EU decimal normalization**: EU decimals written without thousands-grouping dots (e.g. `"1234,56"`) bypass the pattern above. Without special handling, the base regex matches `"1234,56"` as a single token; `_normalise_number` strips the comma as a thousands separator, returning `123456` — 100× too large.

Detection: add a second sub-pattern that matches digit sequences with a comma followed by **exactly 2 digits** and no surrounding grouping structure:

```python
_EU_UNGROUPED_DECIMAL = r'\b\d+,\d{2}\b'
```

Exactly 2 digits after a comma is an EU decimal fingerprint: US thousands separators always use exactly 3 digits after the comma, so this pattern is unambiguous. `\b\d+,\d{1}\b` (1 digit) is excluded — too ambiguous with list context.

`normalize_eu_numbers` runs both `_EU_NUMBER_PATTERN` and `_EU_UNGROUPED_DECIMAL` in sequence. The `_swap` function is reused for both: strip the comma and replace with a period.

```python
_EU_UNGROUPED_DECIMAL_RE = re.compile(r'\b(\d+),(\d{2})\b(?!,\d)')
# (?!,\d) negative lookahead: reject "1,12,14,25" comma-separated lists.
# Without it, list items like "1,12" would be rewritten to "1.12".
# Pre-masking consecutive comma-separated sequences of 3+ numbers before
# this pass is an alternative: r'\b\d+(?:,\d+){2,}\b' → replace with spaces.

def normalize_eu_numbers(text: str) -> str:
    # Pass 1: grouped EU format (1.234,56)
    text = re.sub(_EU_NUMBER_PATTERN, _swap, text)
    # Pass 2: ungrouped EU decimal (1234,56) — comma + exactly 2 digits
    # Negative lookahead guards against comma-separated integer lists (e.g. "1,12,14").
    text = _EU_UNGROUPED_DECIMAL_RE.sub(lambda m: f"{m.group(1)}.{m.group(2)}", text)
    return text
```

**EU integer ambiguity — escalate, do not compare**: Pre-Pass D only handles the unambiguous EU form `1.234,56`. The pattern `1.234` (dot-separated, no trailing comma-decimal) is inherently ambiguous between a US decimal `1.234` and an EU grouping integer `1.234` (= 1234). Tier25 cannot resolve this without document-level locale inference.

Rule: if the base regex extracts a number matching `[1-9]\d{0,2}\.\d{3}$` exactly (1–3 non-zero leading digits, a dot, exactly 3 decimal digits, no further suffix, no comma in the string), set `escalate_reason="eu_integer_ambiguous"` and do not compare it numerically. The LLM handles it in Tier 3.

**Why `[1-9]` not `\d`**: numbers starting with `0.` (e.g. `0.123`, `0.500`) are unambiguously US decimals — no one writes an EU grouping integer starting with a leading zero. Requiring a non-zero first digit excludes correlation coefficients, weights, and ratios (all < 1) from spurious escalation.

---

## Pre-Pass E: Fraction Extraction

Runs after Pre-Pass D, before base extraction on **both claim and source text**. Behaviour is **asymmetric** between the two sides.

```python
_NUMERIC_FRACTION_PATTERN = r'\b(\d+)/(\d+)\b'
```

**Claim side** — remove span + escalate:

```python
def extract_fractions_claim(text: str) -> tuple[list[str], str]:
    """
    Remove fraction spans and signal escalation.

    Reason: claim "1/3 of patients recovered" vs source "33% of patients recovered"
    are equivalent facts, but 0.333 ≠ 33.0.  Direct float comparison always
    produces a false conflict.  Tier 3 handles the semantic equivalence check.

    Returns (raw_spans, text_with_spans_removed).
    If raw_spans is non-empty, caller sets escalate_reason="fraction".

      - "1/3 of patients" → remove span, return raw_span → caller escalates
      - "1/3 completed"   → remove span → caller escalates
      - "page 1/3"        → already blocked by Gate 1; not reached here

    Fractions with denominator 0 are skipped (division guard).
    """
```

**Source side** — remove span + normalize to decimal (no escalation):

```python
def extract_fractions_source(text: str) -> tuple[list[float], str]:
    """
    Remove fraction spans and return decimal values for comparison.

    Source fractions are ground-truth values, not claims to be verified.
    Normalizing them to decimals lets the comparison layer do its job:
      - Source "1/3" → NumericalValue(0.333, None)
      - If claim says "0.33"  → values match → no conflict
      - If claim says "33%"   → unit mismatch (% vs None) → unit_unitless_mismatch
                                 escalation → Tier 3 handles equivalence

    Returns (decimal_values, text_with_spans_removed).
    Caller adds NumericalValue(v, None) entries for each decimal_value.
    """
```

Shared implementation skeleton (both call this, differ only in what they do with matches):

```python
def _scan_fractions(text: str) -> tuple[list[re.Match], str]:
    """Remove fraction spans from text; return matches and cleaned text."""
    matches = []
    spans_to_remove = []
    for m in re.finditer(_NUMERIC_FRACTION_PATTERN, text):
        if int(m.group(2)) == 0:
            continue   # division guard
        matches.append(m)
        spans_to_remove.append((m.start(), m.end()))
    out = list(text)
    for start, end in reversed(spans_to_remove):
        out[start:end] = [' '] * (end - start)
    return matches, ''.join(out)
```

Verbal fractions (`half`, `one-third`, etc.) in Pre-Pass B (`_WORD_NUMS`) follow the same asymmetry: claim verbal fraction → escalate; source verbal fraction → normalize to the mapped float value.

**Fraction handling table — by side**:

| Form | Claim side | Source side |
|---|---|---|
| `1/3 of patients` | remove span → escalate | remove span → NumericalValue(0.333, None) |
| `two-thirds of patients` | Pre-Pass B → escalate | Pre-Pass B → NumericalValue(0.667, None) |
| `1/3` bare | remove span → escalate | remove span → NumericalValue(0.333, None) |
| `page 1/3` | Gate 1 blocks before Pre-Pass E | Gate 1 blocks before Pre-Pass E |

---

## Revised Base Extraction Regex

```python
_CURRENCY_PREFIX = r'(?:[$€£¥₹₩₽]|(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|HKD)\s*)'
_MAGNITUDE_SUFFIX = r'(?:[MBKTmbkt](?:illion)?|bps?|basis\s+points?)?'

_NUMBER_PATTERN = (
    r'(?<!\w)'
    r'[+\-]?'                                         # optional sign (preserved)
    r'(?:' + _CURRENCY_PREFIX + r')?'                 # optional currency prefix
    r'(?:'
        r'\d[\d,]*(?:\.\d+)?[eE][+\-]?\d+'           # scientific: 1.5e6
        r'|'
        r'\d[\d,]*(?:\.\d+)?'                         # standard: 1,234.56
    r')'
    r'(?:%|' + _MAGNITUDE_SUFFIX + r')?'              # optional unit suffix
    r'(?!\w)'
)
```

### `_normalise_number()` updates

The existing function must be extended to handle:

1. **Non-$ currency stripping**: strip `€`, `£`, `¥`, `₹`, `₩`, `₽` and ISO code prefixes before parsing
2. **Sign preservation**: detect leading `+` or `-`, preserve through normalization
3. **Basis points**: if suffix is `bps` or `bp` or `basis points`, divide result by 100 (50bps = 0.50%)
4. **Scientific notation**: `float("1.5e6")` natively handles this — pass directly to `float()`
5. **European format**: if the string matches `\d{1,3}(\.\d{3})*,\d{2}$`, swap dots/commas before `float()` conversion

---

## Gate 2: Head-Noun Check (claim only)

Applied to claim numbers **without** a recognized unit (Gate 3 fast-accepts those, bypassing this gate). Not applied to source text — see Architecture section for rationale. Checks 1–3 tokens immediately following the extracted number span.

```python
_RHETORICAL_NOUNS = {
    # Discourse elements
    "reason", "reasons", "point", "points", "way", "ways",
    "thing", "things", "factor", "factors", "aspect", "aspects",
    "consideration", "considerations", "example", "examples",
    "argument", "arguments", "finding", "findings", "issue", "issues",
    "element", "elements", "topic", "topics", "idea", "ideas",
    "concept", "concepts", "type", "types", "kind", "kinds",
    "category", "categories", "method", "methods", "approach", "approaches",
    # Document structure (belt-and-suspenders after Gate 1)
    "paragraph", "paragraphs", "sentence", "sentences",
    "clause", "clauses", "note", "notes", "footnote", "footnotes",
}
```

---

## Gate 3: Unit-Anchor Fast-Accept

If a number is immediately adjacent (within 1–2 tokens) to any of the following, it is factual. Skip Gate 2.

**Implementation**: `_is_unit_anchored(window_text)` must check **1-gram and 2-gram substrings** of the window — do not split into tokens and do a simple `token in set` lookup. Multi-word anchors like `"years old"`, `"amino acids"`, `"fl oz"` will never match via single-token lookup. The correct implementation:

```python
_PUNCT_STRIP = str.maketrans("", "", ".,;:!?()’‘“”'")

def _is_unit_anchored(window_text: str) -> bool:
    # Strip trailing/leading punctuation from each token before n-gram assembly.
    # Without this, "patients." fails the lookup because the period is attached.
    #
    # Normalize slashes to spaces before splitting so "/share" in "$5/share"
    # becomes "/ share" and each token is checked individually. Without this,
    # "/share" is a single token that matches neither "/" nor "share" in
    # _UNIT_ANCHORS, causing compound rates to miss Gate 3 fast-accept.
    normalized = window_text.replace("/", " / ")
    tokens = [t.translate(_PUNCT_STRIP) for t in normalized.split()]
    tokens = [t for t in tokens if t]  # drop any tokens that were pure punctuation
    for n in (1, 2):
        for i in range(len(tokens) - n + 1):
            if " ".join(tokens[i:i+n]) in _UNIT_ANCHORS:
                return True
    return False
```

The anchor list is split at Phase 3 implementation time into `_MEASURABLE_UNITS` (carry label for unit-mismatch comparison) and `_ENTITY_NOUNS` (fast-accept only). The combined set shown here is the authoritative source for both.

**Source**: quantulum3 entity list + CQE noun-unit examples + humanitarian extraction paper (arxiv 2408.04941) + FiNER-139 XBRL financial NER taxonomy. Entries marked `# existing` were in the original list.

```python
_UNIT_ANCHORS = {

    # ── MEASURABLE UNITS (carry label for unit-mismatch detection) ──────────

    # Percentage and rates
    "%", "percent", "percentage",                                    # existing

    # Mass / weight
    "kg", "g", "mg", "μg", "mcg", "lb", "lbs", "oz", "t", "tonne", # existing

    # Distance
    "km", "m", "cm", "mm", "mi", "ft", "in", "yd", "nm",            # existing

    # Area
    "sqm", "sq m", "sqft", "sq ft", "ha", "hectare", "hectares",
    "acre", "acres", "km2", "m2",

    # Volume
    "L", "mL", "μL", "dl", "gal", "fl oz",                          # existing

    # Energy / calories
    "cal", "kcal", "J", "kJ", "MJ",                                  # existing

    # Power / electrical
    "W", "kW", "MW", "GW", "V", "A", "kWh", "MWh",                  # existing

    # Frequency
    "Hz", "kHz", "MHz", "GHz",                                       # existing

    # Temperature
    "°C", "°F", "K",                                                  # existing

    # Concentration
    "ppm", "ppb", "mol", "mmol", "μmol",                             # existing

    # Speed
    "km/h", "mph", "m/s", "knot", "knots",

    # Data / storage
    "B", "KB", "MB", "GB", "TB", "PB", "Mbps", "Gbps",

    # Financial magnitude suffixes (only when directly adjacent to digits)
    "M", "K", "T",                                                    # existing ("B" moved to data)

    # Basis points
    "bps", "bp",                                                      # existing

    # Rate anchors
    "per", "/",                                                       # existing

    # ── ENTITY NOUNS (fast-accept only; no label stored for comparison) ─────
    # Source: CQE "noun units", humanitarian extraction paper, FiNER-139 XBRL

    # People / headcount — empirically top-frequency category (humanitarian paper)
    "person", "persons", "people",                                    # existing
    "individual", "individuals",                                      # existing
    "employee", "employees",                                          # existing
    "worker", "workers",                                              # existing
    "staff", "headcount", "personnel",
    "hire", "hires",
    "member", "members",
    "user", "users",                                                  # existing
    "child", "children",
    "student", "students",                                            # existing (CQE example)
    "recipient", "recipients",
    "beneficiary", "beneficiaries",
    "household", "households",
    "resident", "residents",
    "citizen", "citizens",
    "volunteer", "volunteers",
    "participant", "participants",
    "applicant", "applicants",
    "candidate", "candidates",
    "voter", "voters",
    "refugee", "refugees",

    # Healthcare / clinical
    "patient", "patients",                                            # existing
    "case", "cases",                                                  # existing
    "death", "deaths",                                                # existing
    "infection", "infections",                                        # existing
    "dose", "doses",                                                  # existing
    "trial", "trials",                                                # existing
    "study", "studies",                                               # existing
    "bed", "beds",
    "treatment", "treatments",
    "procedure", "procedures",
    "drug", "drugs",
    "vaccine", "vaccines",

    # Scientific / biomedical
    "gene", "genes",                                                  # existing
    "compound", "compounds",                                          # existing
    "species",                                                        # existing
    "amino acid", "amino acids",                                      # existing
    "protein", "proteins",                                            # existing
    "sample", "samples",
    "observation", "observations",
    "measurement", "measurements",

    # Infrastructure / locations — "noun units" (CQE)
    "location", "locations",                                          # existing
    "site", "sites",                                                  # existing
    "store", "stores",                                                # existing
    "branch", "branches",
    "office", "offices",
    "outlet", "outlets",
    "facility", "facilities",
    "plant", "plants",
    "factory", "factories",
    "warehouse", "warehouses",
    "center", "centers",
    "centre", "centres",
    "hub", "hubs",
    "campus", "campuses",
    "depot", "depots",
    "station", "stations",
    "clinic", "clinics",
    "hospital", "hospitals",
    "school", "schools",

    # Organizations
    "company", "companies",                                           # existing
    "firm", "firms",                                                  # existing
    "business", "businesses",                                         # existing
    "organization", "organizations",
    "entity", "entities",
    "subsidiary", "subsidiaries",
    "partner", "partners",
    "NGO", "NGOs",
    "startup", "startups",

    # Customers / relationships
    "customer", "customers",                                          # existing
    "client", "clients",                                              # existing
    "subscriber", "subscribers",                                      # existing

    # Products / items
    "product", "products",                                            # existing
    "item", "items",
    "unit", "units",                                                  # existing
    "good", "goods",
    "service", "services",
    "offering", "offerings",

    # Operational
    "project", "projects",
    "initiative", "initiatives",
    "program", "programs",
    "programme", "programmes",
    "deal", "deals",
    "partnership", "partnerships",
    "agreement", "agreements",                                        # existing (contract)
    "contract", "contracts",                                          # existing
    "order", "orders",
    "shipment", "shipments",
    "delivery", "deliveries",
    "incident", "incidents",
    "event", "events",
    "transaction", "transactions",                                    # existing
    "account", "accounts",                                            # existing
    "market", "markets",                                              # existing

    # Financial instruments / assets
    "share", "shares",                                                # existing
    "stock", "stocks",
    "bond", "bonds",
    "loan", "loans",
    "fund", "funds",
    "portfolio", "portfolios",
    "asset", "assets",
    "position", "positions",
    "holding", "holdings",
    "investment", "investments",
    "stake", "stakes",

    # Digital / tech
    "server", "servers",
    "node", "nodes",
    "instance", "instances",
    "query", "queries",
    "download", "downloads",
    "install", "installs",
    "session", "sessions",
    "device", "devices",
    "app", "apps",
    "application", "applications",
    "request", "requests",

    # Geography
    "country", "countries",                                           # existing
    "nation", "nations",                                              # existing
    "city", "cities",                                                 # existing
    "region", "regions",
    "state", "states",
    "territory", "territories",

    # Labour / employment
    "job", "jobs",                                                    # existing
    "role", "roles",
    "position", "positions",

    # Civic
    "vote", "votes",                                                  # existing
    "seat", "seats",

    # Age
    "year-old", "years old", "years-old",                            # existing
}

_AGED_PATTERN = r'\baged?\s+\d+'    # "aged 42", "age 42" → factual
```

### Unit-label propagation

Gate 3 currently acts as a binary gate (fast-accept or not). It must also return the matched anchor label so the comparison loop can detect unit mismatches between claim and source.

**Problem**: `"20 kg"` and `"20 lbs"` both produce `20.0` with no metadata. The comparison sees `20.0 == 20.0` and reports no conflict — a false negative.

**Fix**: Replace bare `float` with a `NumericalValue` named tuple throughout the extraction pipeline:

```python
from typing import NamedTuple

class NumericalValue(NamedTuple):
    value: float
    unit: str | None   # the matched anchor label, or None if no unit found
```

`_is_unit_anchored` is promoted to `_extract_unit_anchor(window_text) -> str | None` — returns the first matching anchor string (e.g. `"kg"`, `"patients"`, `"%"`) or `None`. All callers receive a `NumericalValue` instead of a bare float.

**Comparison rule for unit labels**:

- Both `None` (no units on either side): compare floats directly.
- Both present and equal (e.g. both `"kg"`): compare floats directly.
- Both present and different (e.g. `"kg"` vs `"lbs"`): **escalate** (`escalate_reason="unit_label_mismatch"`), do not flag `has_conflict=True`. Returning `has_conflict=True` here would cause a terminal CONTRADICTED short-circuit, permanently blocking the claim from Tier 3 even when the values are equivalent after unit conversion. Tier 3 resolves.
- One `None`, one present: handled by the existing `unit_unitless_mismatch` escalation rule.

**Scope**: entity-noun anchors (`"patients"`, `"locations"`, etc.) are NOT compared for equality — their presence signals factual context, not a specific unit. Unit-label comparison only applies to measurable-unit anchors (physical units, `%`, `bps`, currency magnitude suffixes). The implementation may split `_UNIT_ANCHORS` into `_MEASURABLE_UNITS` (subject to label comparison) and `_ENTITY_NOUNS` (fast-accept only, no label stored).

---

## Comparison Logic Changes

### 0. Year conflict check — aggregate across all chunks

The existing code checks `claim_years.issubset(chunk_years)` per chunk and breaks on the first supporting chunk. This fails when the claim covers multiple years ("revenue grew between 2023 and 2024") and those years are distributed across different source chunks — no single chunk passes the subset test, producing a false conflict.

**Correct logic**: collect years from **all** source chunks first, then do one subset check:

```python
all_source_years: set[str] = set()
for chunk in chunks:
    all_source_years.update(extract_contextual_years(chunk.text_content))

if claim_years and not claim_years.issubset(all_source_years):
    # Build citation from the first chunk that contains any year,
    # but only if no chunk contains ALL claim years
    conflict_found = True
    # ... set conflict_citation from first year-bearing chunk
```

This also eliminates the awkward `for…else` pattern in the current implementation.

### 1. Sign-aware comparison

After normalization, the sign of both claim and source values is preserved. `-3%` and `+3%` are different facts. The current code compares absolute float values — this must change. Comparison: `claim_float == source_float` where both carry sign.

### 2. Fuzzy quantifier tolerance

Before comparison, scan the claim text for hedge modifiers preceding each extracted number:

```python
_HEDGE_LOWER = {
    "at least", "more than", "over", "above", "exceeding",
    "greater than", "no fewer than", "no less than",
}
_HEDGE_UPPER = {
    "at most", "fewer than", "less than", "under", "below",
    "no more than", "no greater than",
}
_HEDGE_APPROX = {
    "about", "around", "approximately", "roughly", "nearly",
    "almost", "close to", "some", "up to", "as many as",
}
```

- **APPROX**: no conflict if `abs(source - claim) / claim <= 0.10` (±10%). Guard against zero-division: if `claim == 0.0`, use absolute tolerance `abs(source - claim) <= 0.10` instead.
- **LOWER**: no conflict if `source >= claim`
- **UPPER**: no conflict if `source <= claim`

`detect_hedge` signature uses the character offset of the matched number, not the raw string, to handle claims with repeated identical values correctly:

```python
def detect_hedge(claim: str, start_offset: int) -> Literal["lower", "upper", "approx", None]:
    # Scan 3–5 tokens immediately preceding start_offset.
    # Stop scanning at the nearest sentence-ending punctuation (. ! ?) before
    # start_offset — hedge words from a prior sentence must not bleed across
    # the boundary. E.g. "...price is under. 50 patients..." must not tag "50"
    # as LOWER.
    ...
```

### 3. Hyphenated and word-form ranges

A range is a single compound claim with two bounds. Both bounds must be grounded.

```python
# _BOUND captures an optional currency prefix, the numeric body, and an optional
# magnitude/percent suffix. This prevents "$10M–$20M" from extracting "10" and "20"
# instead of the full "$10M" / "$20M" strings that _normalise_number() needs.
#
# The numeric body uses a structured format (?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)
# instead of the greedy \d[\d,.]* to avoid consuming trailing punctuation.
# \d[\d,.]* would match "5.0." or "20," (trailing sentence/clause punctuation),
# causing float("5.0.") to raise ValueError at normalisation time.
_BOUND = (
    r'[+\-]?'
    r'(?:[$€£¥₹₩₽]|(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|HKD)\s*)?'
    r'(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)'
    r'(?:\s*(?:[MBKTmbkt](?:illion)?|bps?|%))?'
    # \s* before the suffix group is required: "5 million" and "10 percent" have
    # a space between the digit core and the scale/unit word. Without it the suffix
    # is not consumed, the range separator match fails, and the range splits into
    # two unrelated single numbers.
)

_RANGE_PATTERNS = [
    rf'({_BOUND})\s*[–—\-]\s*({_BOUND})',              # $10M–$20M, 10–20%
    rf'({_BOUND})\s+to\s+({_BOUND})',                   # 20 to 30 percent
    rf'between\s+({_BOUND})\s+and\s+({_BOUND})',        # between 5 and 10
]
```

Both bounds (lo, hi) are extracted as floats. Comparison: source value must satisfy `lo <= source <= hi`. If source value found outside range → conflict. If no source number found at all → escalate to LLM, do not flag conflict.

**Range-to-range containment**: if the source chunk also contains a range `(src_lo, src_hi)`, check sub-range containment in Tier25 before escalating:

```python
if claim_range and source_range:
    if claim_lo <= source_lo and source_hi <= claim_hi:
        return Tier25Result(has_conflict=False)   # source range fits inside claim range
    # Otherwise escalate; do not flag conflict — Tier 3 resolves partial overlaps.
```

**Suffix distribution**: When a range has a unit suffix or currency prefix on only one bound, distribute it to both before normalization. Examples:
- `10–20%` → both bounds are percentages: lo=10.0, hi=20.0 (same unit as `20%`)
- `$10M–$20M` → currency on both is explicit; no distribution needed
- `50–60 patients` → unit is a following noun, not a suffix; Gate 3 anchors to `patients` on the whole range
- `between 5 and 10 percent` → `percent` follows the upper bound; apply to both: lo=5.0%, hi=10.0%

Implementation rule: after extracting both numeric strings, check if the upper-bound string carries a `%` or magnitude suffix that the lower-bound string lacks. If so, append that suffix to the lower bound before calling `_normalise_number()`.

### 4. Percentage points vs percent

`"rose 3 percentage points"` and `"rose 3%"` are semantically different claims.

Detection: if claim contains `percentage point(s)` or bare `pp` suffix following a number, tag the extracted value with `unit=PERCENTAGE_POINT`. Only compare against source values that also appear in a percentage-point context. If source uses bare `%`, do not attempt comparison — escalate to Tier 3.

Detection pattern:
```python
_PP_PATTERN = re.compile(r'\bpercentage\s+points?\b|(?<![a-zA-Z])pp\b', re.IGNORECASE)
```
`\bpp\b` alone fails on `"3pp"` because `3` and `p` are both `\w`, so no word boundary exists between them. `(?<![a-zA-Z])pp\b` matches `pp` that is NOT preceded by a letter — a digit or space before `pp` satisfies it, while `"xpp"` does not.

---

## Escalation Cases

These are factual numbers that Tier25 cannot verify by numeric comparison. Return `has_conflict=False` with an `escalate_reason` so Tier 2 routing always sends them to Tier 3.

| Pattern | Example | Reason |
|---|---|---|
| X-times | "3x growth", "5× faster" | Requires baseline |
| Ratio | "a 2:1 ratio" | Requires both referents |
| Vague verbal quantity | "several thousand" | Range too wide |
| Decade reference | "in the 1980s", "the 90s" | 10-year span |
| Time expression | "market closes at 4pm" | Ambiguous factual context |
| Score/index | "8 out of 10", "a score of 74" | Requires knowing the scale |
| Bare fraction | "1/3" without entity noun | Ambiguous (could be date fragment) |
| Percentage points | "rose 3pp" vs source in % | Different units |
| Unit vs unitless | claim "50%" vs source "0.5" | Mathematically equal but unit mismatch |
| Abbreviated year range | "2025-26", "FY2024-25" | Range pattern extracts wrong hi bound (26 ≠ 2026) |

**Unit-vs-unitless rule**: if the claim number carries a recognized unit suffix (%, bps, M, B, K, T, or any physical unit) and the candidate source number carries no unit, do **not** compare them numerically — set `escalate_reason="unit_unitless_mismatch"` and let Tier 3 resolve. The converse also applies (unitless claim, unit-bearing source). Direct float comparison in this case produces false conflicts: `50.0 != 0.5` even though `50% == 0.5` as a proportion.

Add `escalate_reason: str | None` field to `Tier25Result`.

---

## Fraction Handling

| Form | Handling |
|---|---|
| `1/3 of [entity noun]` | Gate 3 fast-accept (entity noun present). Extract as `0.333`. |
| `two-thirds of [entity noun]` | Verbalized fraction. Map to float, Gate 3 fast-accept. |
| `1/3` bare (no entity noun) | Escalate — ambiguous with date fragments. |
| `page 1/3` | Blocked by Gate 1 (page label). |

```python
_VERBAL_FRACTIONS = {
    "half": 0.5,
    "one-half": 0.5,       "one half": 0.5,
    "one-third": 0.333,    "two-thirds": 0.667,
    "one-quarter": 0.25,   "three-quarters": 0.75,
    "one-fifth": 0.2,      "two-fifths": 0.4,
    "one-tenth": 0.1,
}
```

---

## The `len(chunk_floats) > 1` Skip Rule

Current logic: skip exact-match check when a source chunk has more than 1 number (too ambiguous for direct comparison).

**Interaction with Gate 1**: After applying the blocklist to source text, previously multi-number chunks will have fewer extracted numbers. Some chunks that previously had `len > 1` (because they contained structural labels alongside real metrics) will now have `len == 1` and become eligible for comparison.

**Resolution**: Do not change the threshold until Phases 1–3 are implemented. Then run the benchmark and measure empirically. Expected new optimal threshold is `> 3` or removal of the rule entirely. This is a calibration step, not a design step.

---

## Library Integration

Two third-party libraries cover sub-problems in Tier25. Neither is used at runtime in the hot path — they inform the implementation at build time or are wrapped with minimal overhead.

### numerizer — Pre-Pass B verbal number normalization

**Package**: `numerizer` (pip install numerizer)
**Use**: Replace the hand-rolled `_parse_verbal_word` helper with numerizer's parser for converting word-form numerals to integers/floats.

```python
from numerizer import numerize

# Preprocessing only — called once per text, not per match
numerize("twenty-five million")   # → "25000000"
numerize("half a billion")        # → "500000000"
numerize("a couple of")           # → vague, not a pure number → falls through to _VAGUE_QUANTIFIER_PATTERN
```

**Integration point**: in `extract_composite_numbers(text)` (Pre-Pass B), call `numerize(text)` on the input before running `_DIGIT_SCALE_PATTERN`. numerizer normalizes word forms to digit strings in-place, so the same downstream regex then matches `"25000000 employees"` without needing `_VERBAL_SCALE_PATTERN` at all.

**Scope limitation**: numerizer handles cardinal word-to-digit conversion only. It does not: detect rhetorical vs factual context, apply Gate 2/3, handle EU formats, or produce `NumericalValue` objects. All of that remains custom code.

**Dependency risk**: numerizer is a small library (~300 lines) with no heavy dependencies. Acceptable for a pre-processing utility. If numerizer is unavailable, fall back to the hand-rolled `_VERBAL_SCALE_PATTERN` + `_parse_verbal_word`. The fallback path is the current spec implementation.

### quantulum3 — unit anchor list source (offline, not runtime)

**Package**: `quantulum3` (pip install quantulum3)  
**Use**: Offline reference only. The `_UNIT_ANCHORS` set in this spec was partially derived from quantulum3's `entities.json` (75 entity types, 290+ units) and unit surface forms. quantulum3 is NOT called at Tier25 runtime.

**Why not runtime**: quantulum3 uses a dependency-parsed GloVe-based disambiguation classifier. That is 100–500ms per sentence — incompatible with Tier25's sub-millisecond target. It also cannot do the claim-vs-source comparison or CONTRADICTED decision.

**Ongoing maintenance**: when adding domain-specific noun units to `_UNIT_ANCHORS`, check `quantulum3/en/units.json` first for the canonical surface forms (plural, abbreviations, symbol variants). This prevents drift between our anchors and what quantulum3 recognizes.

---

## Configuration Constants

All tunable thresholds and magic numbers are defined as named module-level constants in `tier25_preprocessing.py`. They are never inline literals inside logic. This makes calibration (Phase 7 data-driven recalibration) a one-line change.

```python
# tier25_preprocessing.py — Constants block (top of file, below imports)

# Fuzzy hedge tolerance
APPROX_TOLERANCE: float = 0.10          # ±10% relative tolerance for "about", "roughly" etc.
APPROX_ZERO_ABS_TOLERANCE: float = 0.10 # absolute tolerance when claim value == 0.0

# Extraction windows
UNIT_ANCHOR_WINDOW_TOKENS: int = 2      # how many tokens after a number to scan for unit anchor
HEDGE_SCAN_TOKENS: int = 5              # how many tokens before a number to scan for hedge words
RHETORICAL_SCAN_TOKENS: int = 3         # how many tokens after a number to scan for rhetorical noun

# Multi-number skip rule (recalibrated in Phase 7)
MULTI_NUMBER_SKIP_THRESHOLD: int = 1    # skip comparison if source chunk has > N extracted numbers

# EU ambiguity detection
EU_INTEGER_MIN_LEAD_DIGITS: int = 1     # minimum non-zero leading digits before "." for ambiguity check
EU_INTEGER_DECIMAL_DIGITS: int = 3      # exactly 3 decimal digits after "." signals potential EU integer

# Range containment
RANGE_CONTAINMENT_STRICT: bool = True   # True = source range must be fully inside claim range
                                        # False = partial overlap → escalate

# Abbreviated year range
ABBREVIATED_YEAR_CENTURY_THRESHOLD: int = 100  # hi < this → hi is a 2-digit year suffix
```

These constants live separately from the pattern strings (`_GATE1_BLOCKLIST`, `_RHETORICAL_NOUNS`, `_UNIT_ANCHORS`, etc.) which are structural definitions, not tunable parameters. Pattern strings change only when the design changes; constants change during calibration.

---

## Implementation Plan

### Phase 1 — Gate 1 blocklist + composite pre-pass (highest impact)

**File**: `groundguard/tiers/tier25_preprocessing.py`  
**Expected impact**: -14 to -20 FP on benchmark  
**Risk**: Low

Tasks:
1. Add `_GATE1_BLOCKLIST` as a module-level list of compiled `re.Pattern` objects
2. Add `mask_structural(text: str) -> str` — applies blocklist, returns masked copy (spans → spaces)
3. Add `extract_composite_numbers(text: str) -> tuple[list[tuple[float, str]], str]` — returns `(value, raw_span)` pairs and the text with those spans removed
4. Add `normalize_accounting_negatives(text: str) -> str` — converts `(1,234.56)` → `-1234.56`
5. In `run()`, apply pre-passes A–D identically to both sides. Pre-Pass E uses `extract_fractions_claim` (remove + escalate) on claim and `extract_fractions_source` (remove + normalize to decimal) on source — do NOT use a shared `extract_numbers()` function here; the asymmetry is intentional. Gate 2 runs on claim text only. Gate 3 runs on both sides. Hedge detection and vague-quantifier escalation run on claim text only (comparison layer). Refer to the Architecture symmetry table for the canonical per-step breakdown.

Test cases:
- `"Passage 1 states the revenue was $400M"` → Gate 1 removes `Passage 1`; `$400M` extracted
- `"3 million users"` → composite pre-pass extracts `3000000.0`; base regex does not see bare `3`
- `"revenue of (500M)"` → accounting negative → `-500.0M`
- `"Fig. 3 shows 42 patients"` → Gate 1 removes `Fig. 3`; `42` extracted + Gate 3 fast-accept (`patients`)
- `"Step 2 of 5 completed"` → Gate 1 removes `Step 2` and `5`; zero claim numbers → skip Tier25
- `"Phase II trial enrolled 200 patients"` → Roman numeral label blocked; `200` extracted
- `"Windows 11 reached 20% market share"` → `Windows 11` blocked; `20%` extracted

---

### Phase 2 — Revised base regex + currencies + scientific notation

**File**: `groundguard/tiers/tier25_preprocessing.py`  
**Expected impact**: -2 to -4 FP, -2 to -3 FN (catches previously missed values)  
**Risk**: Medium (regex change touches all extraction)

Tasks:
1. Replace `_NUMBER_PATTERN` with the revised version above
2. Update `_normalise_number()` for: non-$ currency stripping, sign preservation, bps conversion, scientific notation, European format detection
3. Add `_CURRENCY_PREFIX` and `_MAGNITUDE_SUFFIX` as module-level constants

Test cases:
- `"€1,234.56 revenue"` → extracts `€1,234.56`, normalizes to `1234.56`
- `"+5% growth"` → extracts `+5%`, sign preserved as `+5.0`
- `"-3% decline"` → sign preserved as `-3.0`
- `"1.5e6 infections"` → extracts `1.5e6`, normalizes to `1500000.0`
- `"50bps rate increase"` → extracts `50bps`, normalizes to `0.50`
- `"1.234,56"` (EU format) → detects EU format, normalizes to `1234.56`
- `"USD 4.2M"` → extracts and normalizes to `4200000.0`

---

### Phase 3 — Gate 2 (rhetorical nouns) + Gate 3 (unit anchors)

**File**: `groundguard/tiers/tier25_preprocessing.py`  
**Expected impact**: -5 to -8 FP  
**Risk**: Low

Tasks:

1. Add `_RHETORICAL_NOUNS` set as module-level constant
2. Split `_UNIT_ANCHORS` into two sets: `_MEASURABLE_UNITS` (physical, financial, rate units — carry label for comparison) and `_ENTITY_NOUNS` (concrete entity nouns — fast-accept only, no label stored)
3. Add `NumericalValue = NamedTuple("NumericalValue", [("value", float), ("unit", "str | None")])` — replace bare `float` throughout extraction pipeline
4. Replace `_is_unit_anchored(following_text: str) -> bool` with `_extract_unit_anchor(following_text: str) -> str | None` — returns matched `_MEASURABLE_UNITS` anchor or `None`; also checks `_ENTITY_NOUNS` for fast-accept (returns sentinel `"_entity"` or a constant to distinguish from measurable)
5. Add `_has_rhetorical_head(following_text: str) -> bool`
6. In `run()`, after extracting claim numbers: for each number, call `_extract_unit_anchor` → if returns non-None, produce `NumericalValue(value, anchor)`; else call `_has_rhetorical_head` → if True, discard
7. In comparison logic: when comparing claim `NumericalValue` to source `NumericalValue`, apply unit-label rules: both measurable units present and differ → conflict; one measurable/one None → `unit_unitless_mismatch` escalation; entity labels ignored for unit comparison
8. Add `_AGED_PATTERN` and `_VERBAL_FRACTIONS` as module-level constants
9. Add per-unit compound anchor detection (e.g., `$5/share` → `share` fast-accept via normalized slash tokenization)

Test cases:
- `"3 reasons support this"` → rhetorical noun `reasons` → number discarded
- `"3 locations confirmed"` → entity noun `locations` → Gate 3 fast-accept (`unit=None`)
- `"20 amino acids found"` → entity noun `amino acids` → Gate 3 fast-accept (`unit=None`)
- `"a 42-year-old patient"` → `year-old` anchor → Gate 3 fast-accept
- `"$5/share dividend"` → `/` rate anchor → Gate 3 fast-accept
- `"5 ways to improve"` → rhetorical noun `ways` → discarded
- `"20 kg"` claim vs `"20 lbs"` source → both `NumericalValue(20.0, "kg"/"lbs")` → unit label mismatch → conflict
- `"20 kg"` claim vs `"20"` source → `NumericalValue(20.0, "kg")` vs `NumericalValue(20.0, None)` → `unit_unitless_mismatch` escalation

---

### Phase 4 — Fuzzy quantifier tolerance

**File**: `groundguard/tiers/tier25_preprocessing.py`  
**Expected impact**: -3 to -5 FP  
**Risk**: Low

Tasks:
1. Add `_HEDGE_LOWER`, `_HEDGE_UPPER`, `_HEDGE_APPROX` sets
2. Add `detect_hedge(claim: str, number_raw: str) -> Literal["lower", "upper", "approx", None]` — scans 3–5 tokens before the number for hedge words
3. Modify comparison logic in `run()`: when hedge detected, apply tolerance rule instead of exact match

Test cases:
- `"almost 32%"` vs source `"31.5%"` → APPROX, within 10% → no conflict
- `"at least 100 employees"` vs source `"150 employees"` → LOWER, 150 ≥ 100 → no conflict
- `"at least 100 employees"` vs source `"80 employees"` → LOWER, 80 < 100 → conflict
- `"fewer than 5 cases"` vs source `"3 cases"` → UPPER, 3 ≤ 5 → no conflict
- `"approximately $5M revenue"` vs source `"$4.6M"` → APPROX, within 10% → no conflict
- `"approximately $5M revenue"` vs source `"$3M"` → APPROX, 40% diff → conflict

---

### Phase 5 — Range handling

**File**: `groundguard/tiers/tier25_preprocessing.py`  
**Expected impact**: -3 FP, -1 FN  
**Risk**: Medium

Tasks:
1. Add `_RANGE_PATTERNS` list
2. Add `extract_ranges(text: str) -> list[tuple[float, float, str]]` — returns `(lo, hi, raw_span)` triples
3. Run range extraction before single-number extraction in `run()`; remove range spans from text to prevent double-extraction
4. Comparison: source value in `[lo, hi]` → no conflict; source value outside → conflict; no source number found → escalate (not a conflict); source range fully inside claim range → no conflict (sub-range containment)
5. Suffix distribution: after extracting both numeric strings, check if the upper-bound string carries a `%` or magnitude suffix that the lower-bound string lacks; if so, append that suffix to the lower bound before calling `_normalise_number()`

Test cases:
- `"between 20 and 30%"` vs source `"25%"` → in range → no conflict
- `"between 20 and 30%"` vs source `"35%"` → outside range → conflict
- `"50–60 patients enrolled"` vs source `"55 patients"` → in range → no conflict
- `"ages 18–65"` vs source `"age 45"` → in range → no conflict
- `"$10M–$20M revenue"` extracted as `(10000000.0, 20000000.0)` — not as two separate numbers

---

### Phase 6 — Escalation flags + score/fraction/ratio/decade/pp

**Files**: `groundguard/tiers/tier25_preprocessing.py`, `groundguard/models/internal.py`  
**Expected impact**: -5 to -10 FN (these were reaching LLM anyway but via wrong path)  
**Risk**: Medium

Tasks:
1. Add `escalate_reason: str | None = None` field to `Tier25Result`
2. Add detection patterns for each escalation case:
   - `r'\b\d+(?:\.\d+)?[x×]\b'` → ratio/times
   - `r'\b\d+\s*:\s*\d+\b'` → ratio notation
   - `r'\bthe\s+\d{4}s\b'` → decade reference
   - `r'\b\d{1,2}:\d{2}\b'` → time
   - `r'\b\d+\s+out\s+of\s+\d+\b'` → score expression
   - `r'\bpercentage\s+points?\b|(?<![a-zA-Z])pp\b'` → percentage points (handles attached form `3pp`; `\bpp\b` alone fails because `3` and `p` are both `\w`)
   - Bare fraction `r'\b\d+/\d+\b'` without adjacent entity noun
   - `r'\b(?:FY|Q\d\s)?((?:19|20)\d{2})[-–]((?:\d{2}|\d{4}))\b'` → abbreviated year range (`2025-26`, `FY2024-25`). Must run **before** `_RANGE_PATTERNS` in the pipeline so the hyphen is consumed here and not misread as a range with `hi=26`. Escalate with `escalate_reason="abbreviated_year_range"` — Tier 3 interprets it as a fiscal/calendar period.
3. When detected: return `Tier25Result(has_conflict=False, escalate_reason="<reason>")` — do not compare
4. Wire `escalate_reason` into Tier 2 routing so a non-None value forces Tier 3 regardless of BM25 score

Test cases:
- `"revenue grew 3x"` → escalate, `escalate_reason="ratio_times"`
- `"in the 1980s"` → escalate, `escalate_reason="decade_reference"`
- `"an 8 out of 10 rating"` → escalate, `escalate_reason="score_expression"`
- `"rose 3 percentage points"` → escalate, `escalate_reason="percentage_points"`
- `"1/3 of patients recovered"` → Gate 3 fast-accept (`patients`)
- `"1/3 completed"` → escalate (no entity noun)
- `"revenue in 2025-26"` → escalate, `escalate_reason="abbreviated_year_range"`
- `"FY2024-25 results"` → escalate, `escalate_reason="abbreviated_year_range"`

---

### Phase 7 — Recalibrate len(chunk_floats) > 1 skip rule

**File**: groundguard/tiers/tier25_preprocessing.py  
**Prerequisite**: Phases 1–3 complete and benchmark rerun

Tasks:
1. Run enchmarks/run_benchmark.py against enchmarks/results/phase1_v013_rerun.jsonl after Phases 1–3
2. Inspect how many chunks now have len(chunk_floats) == 1 that previously had len > 1 (Gate 1 filtered their structural numbers)
3. Measure FP/FN impact of raising threshold from > 1 to > 2, > 3, and removal
4. Update skip threshold to the value that maximizes F1 on the benchmark set
5. Document the chosen value and the data behind it in this file

#### Calibration Results (June 2026)

We ran an offline simulation of MULTI_NUMBER_SKIP_THRESHOLD across all 900 RAGTruth QA subset samples:
- **Total chunks processed**: 900
- **Chunks with > 1 float (with Gate 1 filtering)**: 618

| Skip Threshold | Skipped Claims | Not Skipped Claims | Conflict Flagged | TP (Evident Conflict) | FP (Evident Conflict) | Precision (EC) | TP (Any Halu) | FP (Any Halu) | Precision (Any Halu) |
|---|---|---|---|---|---|---|---|---|---|
| **1 (Default)** | **660** | **240** | **23** | **1** | **22** | **4.35%** | **6** | **17** | **26.09%** |
| 2 | 659 | 241 | 24 | 1 | 23 | 4.17% | 6 | 18 | 25.00% |
| 3 | 659 | 241 | 24 | 1 | 23 | 4.17% | 6 | 18 | 25.00% |
| inf (disabled) | 654 | 246 | 29 | 1 | 28 | 3.45% | 6 | 23 | 20.69% |

**Conclusion**:
Raising the skip threshold or disabling it increases False Positives (FP) from 22 to 28, without capturing any additional True Positives (TP remains at 1 for Evident Conflict and 6 for Any Hallucination). The optimal value that maximizes Precision and F1-score is **MULTI_NUMBER_SKIP_THRESHOLD = 1**.

---

## Priority and Expected Benchmark Impact

Phases must be implemented **in numerical order** (1 → 7). Phase 2 (revised extraction) is a hard prerequisite for Phase 3 (gating) and Phase 4 (fuzzy quantifiers): both Gate 2/3 and hedge logic operate on the extracted number spans and their surrounding context, so the extraction must be correct first. Do not skip ahead to Phase 3/4 for quick wins — the gating precision depends on Phase 2.

| Phase | Description | FP change | FN change | Risk |
|---|---|---|---|---|
| 1 | Gate 1 blocklist + composite pre-passes | -14 to -20 | neutral | Low |
| 2 | Revised base regex + currencies + scientific notation | -2 to -4 | -2 to -3 | Medium |
| 3 | Gate 2 (rhetorical nouns) + Gate 3 (unit anchors) | -5 to -8 | neutral | Low |
| 4 | Fuzzy quantifier tolerance | -3 to -5 | neutral | Low |
| 5 | Range handling (hyphenated + word-form) | -3 | -1 | Medium |
| 6 | Escalation flags + score/fraction/ratio/decade/pp | 0 direct | -5 to -10 | Medium |
| 7 | Skip rule recalibration (data-driven) | data-driven | data-driven | Low |

**Baseline**: FP=142, FN=93, Acc=72.6% (phase1_v013_rerun.jsonl)  
**Expected after all phases + abstention fix**: FP ~85–100, Acc ~81–83%, exceeding the 82.6% trivial baseline

---

## Files to Modify

| File | Change |
|---|---||
| groundguard/tiers/tier25_preprocessing.py | All extraction, normalization, comparison, and escalation logic |
| groundguard/models/internal.py | Add escalate_reason field to Tier25Result if not already there (currently in tier25_preprocessing.py) |
| groundguard/tiers/tier2_semantic.py | Read escalate_reason from Tier25Result; force Tier 3 routing when non-None |
| 	ests/test_tier25.py (new file) | All test cases listed above per phase |
