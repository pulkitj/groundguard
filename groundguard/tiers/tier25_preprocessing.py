"""Tier 2.5 — Numerical consistency pre-check."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, Literal

if TYPE_CHECKING:
    from groundguard.models.internal import VerificationContext
    from groundguard.loaders.chunker import Chunk

from groundguard.models.result import Citation

class NumericalValue(NamedTuple):
    value: float
    unit: str | None

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

_RHETORICAL_NOUNS = {
    "reason", "reasons", "point", "points", "way", "ways",
    "thing", "things", "factor", "factors", "aspect", "aspects",
    "consideration", "considerations", "example", "examples",
    "argument", "arguments", "finding", "findings", "issue", "issues",
    "element", "elements", "topic", "topics", "idea", "ideas",
    "concept", "concepts", "type", "types", "kind", "kinds",
    "category", "categories", "method", "methods", "approach", "approaches",
    "paragraph", "paragraphs", "sentence", "sentences",
    "clause", "clauses", "note", "notes", "footnote", "footnotes",
}

_MEASURABLE_UNITS = {
    "%", "percent", "percentage",
    "kg", "g", "mg", "μg", "mcg", "lb", "lbs", "oz", "t", "tonne",
    "km", "m", "cm", "mm", "mi", "ft", "in", "yd", "nm",
    "sqm", "sq m", "sqft", "sq ft", "ha", "hectare", "hectares",
    "acre", "acres", "km2", "m2",
    "L", "mL", "μL", "dl", "gal", "fl oz",
    "cal", "kcal", "J", "kJ", "MJ",
    "W", "kW", "MW", "GW", "V", "A", "kWh", "MWh",
    "Hz", "kHz", "MHz", "GHz",
    "°C", "°F", "K",
    "ppm", "ppb", "mol", "mmol", "μmol",
    "km/h", "mph", "m/s", "knot", "knots",
    "B", "KB", "MB", "GB", "TB", "PB", "Mbps", "Gbps",
    "M", "K", "T",
    "bps", "bp",
    "per", "/",
}

_ENTITY_NOUNS = {
    "person", "persons", "people",
    "individual", "individuals",
    "employee", "employees",
    "worker", "workers",
    "staff", "headcount", "personnel",
    "hire", "hires",
    "member", "members",
    "user", "users",
    "child", "children",
    "student", "students",
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
    "patient", "patients",
    "case", "cases",
    "death", "deaths",
    "infection", "infections",
    "dose", "doses",
    "trial", "trials",
    "study", "studies",
    "bed", "beds",
    "treatment", "treatments",
    "procedure", "procedures",
    "drug", "drugs",
    "vaccine", "vaccines",
    "gene", "genes",
    "compound", "compounds",
    "species",
    "amino acid", "amino acids",
    "protein", "proteins",
    "sample", "samples",
    "observation", "observations",
    "measurement", "measurements",
    "location", "locations",
    "site", "sites",
    "store", "stores",
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
    "company", "companies",
    "firm", "firms",
    "business", "businesses",
    "organization", "organizations",
    "entity", "entities",
    "subsidiary", "subsidiaries",
    "partner", "partners",
    "NGO", "NGOs",
    "startup", "startups",
    "customer", "customers",
    "client", "clients",
    "subscriber", "subscribers",
    "product", "products",
    "item", "items",
    "unit", "units",
    "good", "goods",
    "service", "services",
    "offering", "offerings",
    "project", "projects",
    "initiative", "initiatives",
    "program", "programs",
    "programme", "programmes",
    "deal", "deals",
    "partnership", "partnerships",
    "agreement", "agreements",
    "contract", "contracts",
    "order", "orders",
    "shipment", "shipments",
    "delivery", "deliveries",
    "incident", "incidents",
    "event", "events",
    "transaction", "transactions",
    "account", "accounts",
    "market", "markets",
    "share", "shares",
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
    "country", "countries",
    "nation", "nations",
    "city", "cities",
    "region", "regions",
    "state", "states",
    "territory", "territories",
    "job", "jobs",
    "role", "roles",
    "vote", "votes",
    "seat", "seats",
    "year-old", "years old", "years-old",
}

_AGED_PATTERN = re.compile(r'\baged?\s+\d+')
_VERBAL_FRACTIONS = {
    "half": 0.5, "one-half": 0.5, "one half": 0.5,
    "one-third": 0.333, "two-thirds": 0.667,
    "one-quarter": 0.25, "three-quarters": 0.75,
    "one-fifth": 0.2, "two-fifths": 0.4,
    "one-tenth": 0.1,
}

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
    r'(?:\s*(?:%|' + _MAGNITUDE_SUFFIX + r'))?'
    r'(?!\w)',
    re.IGNORECASE
)

_BOUND = (
    r'[+\-]?'
    r'(?:[$€£¥₹₩₽]|(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|HKD)\s*)?'
    r'(?:\d[\d,]*(?:\.\d+)?[eE][+\-]?\d+|\d[\d,]*(?:\.\d+)?)'
    r'(?:\s*(?:%|(?:[MBKTmbkt](?:illion)?|bps?|basis\s+points?)))?'
)

_RANGE_PATTERNS = [
    re.compile(rf'{_BOUND}\s*[–—\-]\s*{_BOUND}', re.IGNORECASE),              # $10M–$20M, 10–20%
    re.compile(rf'{_BOUND}\s+to\s+{_BOUND}', re.IGNORECASE),                   # 20 to 30 percent
    re.compile(rf'\bbetween\s+{_BOUND}\s+and\s+{_BOUND}', re.IGNORECASE),        # between 5 and 10
]


_STOPWORDS = {"the", "a", "an", "and", "or", "in", "of", "to", "is", "was", "be", "see", "for",
              "section", "details", "reference", "per", "at", "by", "with", "that", "this",
              "which", "from", "are", "has", "have", "had", "not", "do", "does", "shall",
              "will", "would", "may", "can", "its", "it", "we", "our"}

# Year ONLY in temporal context: "in 2023", "for 2024", "Q3 2023", "FY2024", etc.
_YEAR_CONTEXT_PATTERN = r'(?:in|for|during|as of|fiscal|FY|Q[1-4])\s*(\d{4})\b'

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


def extract_ranges(text: str) -> list[tuple[float, float, str]]:
    matches_by_start = {}
    for pattern in _RANGE_PATTERNS:
        for match in pattern.finditer(text):
            start = match.start()
            if start not in matches_by_start:
                matches_by_start[start] = match
            else:
                if len(match.group(0)) > len(matches_by_start[start].group(0)):
                    matches_by_start[start] = match

    results_with_start = []
    for start, match in matches_by_start.items():
        lo_str = match.group(1)
        hi_str = match.group(2)
        raw_span = match.group(0)

        # Suffix distribution
        suffix_pat = re.compile(
            r'\s*(?:%|percent(?:age)?|bps?|basis\s+points?|[MBKTmbkt](?:illion)?)$',
            re.IGNORECASE
        )
        lo_m = suffix_pat.search(lo_str)
        hi_m = suffix_pat.search(hi_str)
        if hi_m and not lo_m:
            lo_str = lo_str + hi_m.group(0)
        elif lo_m and not hi_m:
            hi_str = hi_str + lo_m.group(0)

        # Prefix distribution
        prefix_pat = re.compile(
            r'^([$€£¥₹₩₽]|(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|HKD)\b\s*)',
            re.IGNORECASE
        )
        
        # Inline split_sign_and_rest for lo_str
        lo_sign = ''
        lo_rest = lo_str
        if lo_str.startswith('+') or lo_str.startswith('-'):
            lo_sign = lo_str[0]
            lo_rest = lo_str[1:]
            
        # Inline split_sign_and_rest for hi_str
        hi_sign = ''
        hi_rest = hi_str
        if hi_str.startswith('+') or hi_str.startswith('-'):
            hi_sign = hi_str[0]
            hi_rest = hi_str[1:]

        lo_pref_m = prefix_pat.search(lo_rest)
        hi_pref_m = prefix_pat.search(hi_rest)

        if hi_pref_m and not lo_pref_m:
            lo_str = lo_sign + hi_pref_m.group(0) + lo_rest
        elif lo_pref_m and not hi_pref_m:
            hi_str = hi_sign + lo_pref_m.group(0) + hi_rest

        # Inline pre_normalise_bound for lo_str
        lo_norm_str = re.sub(r'\bpercent(?:age)?\b', '%', lo_str, flags=re.IGNORECASE)
        lo_norm_str = re.sub(r'\bbasis\s+points?\b', 'bps', lo_norm_str, flags=re.IGNORECASE)
        lo_norm_str = re.sub(r'\bthousand\b', 'k', lo_norm_str, flags=re.IGNORECASE)
        lo_norm_str = re.sub(r'\bmillion\b', 'm', lo_norm_str, flags=re.IGNORECASE)
        lo_norm_str = re.sub(r'\bbillion\b', 'b', lo_norm_str, flags=re.IGNORECASE)
        lo_norm_str = re.sub(r'\btrillion\b', 't', lo_norm_str, flags=re.IGNORECASE)

        # Inline pre_normalise_bound for hi_str
        hi_norm_str = re.sub(r'\bpercent(?:age)?\b', '%', hi_str, flags=re.IGNORECASE)
        hi_norm_str = re.sub(r'\bbasis\s+points?\b', 'bps', hi_norm_str, flags=re.IGNORECASE)
        hi_norm_str = re.sub(r'\bthousand\b', 'k', hi_norm_str, flags=re.IGNORECASE)
        hi_norm_str = re.sub(r'\bmillion\b', 'm', hi_norm_str, flags=re.IGNORECASE)
        hi_norm_str = re.sub(r'\bbillion\b', 'b', hi_norm_str, flags=re.IGNORECASE)
        hi_norm_str = re.sub(r'\btrillion\b', 't', hi_norm_str, flags=re.IGNORECASE)

        try:
            lo_val = _normalise_number(lo_norm_str)
            hi_val = _normalise_number(hi_norm_str)
            results_with_start.append((start, lo_val, hi_val, raw_span))
        except (ValueError, TypeError):
            continue

    results_with_start.sort(key=lambda x: x[0])
    return [(lo, hi, raw) for start, lo, hi, raw in results_with_start]


def _determine_range_unit(raw_span: str, text: str, start: int, end: int) -> str | None:
    suffix = text[end:]
    suffix_tokens = suffix.split()
    window_tokens = suffix_tokens[:UNIT_ANCHOR_WINDOW_TOKENS]
    window_text = " ".join(window_tokens)
    
    # Strip any leading hyphens/spaces/punctuation from this window
    stripped_window = window_text.lstrip("- \t\n\r.,;:!?()'" + '"')
    
    # Pass the stripped window text to _extract_unit_anchor
    unit = _extract_unit_anchor(stripped_window)
    if unit == "_entity":
        return None
    return unit


def _standardise_unit(u: str | None) -> str | None:
    if u is None:
        return None
    u_lower = u.lower().strip()
    if u_lower in ("%", "percent", "percentage"):
        return "%"
    if u_lower in ("bps", "bp", "basis point", "basis points"):
        return "bps"
    currencies = ("$", "€", "£", "¥", "₹", "₩", "₽", "usd", "eur", "gbp", "jpy", "chf", "cad", "aud", "hkd")
    for curr in currencies:
        if u_lower == curr:
            return curr.upper() if len(curr) > 1 else curr
    return u_lower


def _get_effective_unit(raw: str, parsed_unit: str | None = None) -> str | None:
    if parsed_unit is not None:
        std = _standardise_unit(parsed_unit)
        if std:
            return std

    raw_lower = raw.lower().strip()
    suffix_pat = re.compile(
        r'\s*(?:%|percent(?:age)?|bps?|basis\s+points?)$',
        re.IGNORECASE
    )
    m_sfx = suffix_pat.search(raw_lower)
    if m_sfx:
        return _standardise_unit(m_sfx.group(0).strip())

    prefix_pat = re.compile(
        r'^([$€£¥₹₩₽]|(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|HKD)\b\s*)',
        re.IGNORECASE
    )
    
    # split sign and rest inline
    rest = raw
    if raw.startswith('+') or raw.startswith('-'):
        rest = raw[1:]
        
    m_pref = prefix_pat.search(rest.strip())
    if m_pref:
        return _standardise_unit(m_pref.group(1).strip())

    return None


def _check_unit_mismatch_custom(claim_unit: str | None, chunk_unit: str | None) -> str | None:
    if claim_unit is not None and chunk_unit is not None:
        if claim_unit != chunk_unit:
            return "unit_label_mismatch"
    elif (claim_unit is not None) != (chunk_unit is not None):
        return "unit_unitless_mismatch"
    return None


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
    escalate_reason: str | None = None


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


_PUNCT_STRIP = str.maketrans("", "", ".,;:!?()''\"\"'’‘“”")


def _extract_unit_anchor(window_text: str) -> str | None:
    normalized = window_text.replace("/", " / ")
    tokens = [t.translate(_PUNCT_STRIP) for t in normalized.split()]
    tokens = [t for t in tokens if t]
    tokens = tokens[:UNIT_ANCHOR_WINDOW_TOKENS]
    for n in (2, 1):
        for i in range(len(tokens) - n + 1):
            candidate = " ".join(tokens[i:i+n])
            if candidate == "/":
                if i + 1 < len(tokens):
                    next_tok = tokens[i+1]
                    return "/" + next_tok
                return "/"
            if candidate == "per":
                if i + 1 < len(tokens):
                    next_tok = tokens[i+1]
                    return "per " + next_tok
                return "per"
            if candidate in _MEASURABLE_UNITS:
                return candidate
            if candidate in _ENTITY_NOUNS:
                return "_entity"
    return None


def _has_rhetorical_head(following_text: str) -> bool:
    tokens = following_text.split()
    stripped_tokens = [t.translate(_PUNCT_STRIP) for t in tokens]
    stripped_tokens = [t for t in stripped_tokens if t]
    limit = min(RHETORICAL_SCAN_TOKENS, len(stripped_tokens))
    for i in range(limit):
        if stripped_tokens[i].lower() in _RHETORICAL_NOUNS:
            return True
    return False


def detect_hedge(claim: str, start_offset: int) -> Literal["lower", "upper", "approx", None]:
    if start_offset < 0 or start_offset >= len(claim):
        return None
    preceding = claim[:start_offset]
    boundary_idx = -1
    for char in (".", "!", "?"):
        idx = preceding.rfind(char)
        if idx > boundary_idx:
            boundary_idx = idx
    if boundary_idx != -1:
        preceding = preceding[boundary_idx + 1:]
    tokens = [t.translate(_PUNCT_STRIP).lower() for t in preceding.split()]
    tokens = [t for t in tokens if t]
    tokens = tokens[-5:]
    for end_idx in range(len(tokens), 0, -1):
        for start_idx in range(max(0, end_idx - 3), end_idx):
            phrase = " ".join(tokens[start_idx:end_idx])
            if phrase in _HEDGE_LOWER:
                return "lower"
            if phrase in _HEDGE_UPPER:
                return "upper"
            if phrase in _HEDGE_APPROX:
                return "approx"
    return None


def _check_unit_mismatch(claim_num: NumericalValue, chunk_num: NumericalValue) -> str | None:
    if claim_num.unit is not None and chunk_num.unit is not None:
        if claim_num.unit != chunk_num.unit:
            return "unit_label_mismatch"
    elif (claim_num.unit is not None) != (chunk_num.unit is not None):
        return "unit_unitless_mismatch"
    return None


def _extract_numerical_values(preprocessed_text: str, is_claim: bool) -> list[tuple[NumericalValue, str, int]]:
    # Extract standard numeric fractions \b\d+/\d+\b first to prevent split match
    fractions = []
    modified_text = preprocessed_text
    while True:
        match = re.search(r"\b(\d+)/(\d+)\b", modified_text)
        if not match:
            break
        num = float(match.group(1))
        den = float(match.group(2))
        val = round(num / den, 3)
        raw_span = match.group(0)
        start, end = match.span()
        fractions.append((val, raw_span, start, end))
        modified_text = modified_text[:start] + (' ' * len(raw_span)) + modified_text[end:]

    # Extract composites
    composites, remaining = extract_composite_numbers_with_indices(modified_text)
    
    # Store all candidate matches as (value, raw_span, start, end)
    candidates = []
    for val, raw, start, end in fractions:
        candidates.append((val, raw, start, end))
        
    for val, raw, start in composites:
        candidates.append((val, raw, start, start + len(raw)))
        
    for match in re.finditer(_NUMBER_PATTERN, remaining):
        raw = match.group(0)
        try:
            val = float(_normalise_number(raw))
            candidates.append((val, raw, match.start(), match.end()))
        except ValueError:
            pass
            
    # Sort candidate matches by start index to keep left-to-right order
    candidates.sort(key=lambda x: x[2])
    
    results = []
    for val, raw, start, end in candidates:
        # Extract the window of text following it (up to UNIT_ANCHOR_WINDOW_TOKENS tokens)
        suffix = preprocessed_text[end:]
        suffix_tokens = suffix.split()
        window_tokens = suffix_tokens[:UNIT_ANCHOR_WINDOW_TOKENS]
        window_text = " ".join(window_tokens)
        
        # Strip any leading hyphens/spaces/punctuation from this window
        stripped_window = window_text.lstrip("- \t\n\r.,;:!?()'" + '"')
        
        # Pass the stripped window text to _extract_unit_anchor
        unit = _extract_unit_anchor(stripped_window)
        
        if unit is not None:
            if unit == "_entity":
                results.append((NumericalValue(val, None), raw, start))
            else:
                results.append((NumericalValue(val, unit), raw, start))
        else:
            # Gate 2 rhetorical noun check (only for claim text)
            discard = False
            if is_claim:
                if _has_rhetorical_head(stripped_window):
                    discard = True
            
            if not discard:
                results.append((NumericalValue(val, None), raw, start))
                
    return results


def run(ctx: "VerificationContext", chunks: list) -> Tier25Result:
    """Run numerical consistency check."""
    # Preprocess claim
    claim_text = ctx.claim
    claim_text = mask_structural(claim_text)
    claim_text = normalize_accounting_negatives(claim_text)
    claim_text = normalize_eu_numbers(claim_text)
    
    # Check for verbal or standard fractions in claim to escalate immediately
    has_verbal_fraction = False
    for vf in _VERBAL_FRACTIONS:
        if re.search(r"\b" + re.escape(vf) + r"\b", ctx.claim, re.IGNORECASE):
            has_verbal_fraction = True
            break
    has_numeric_fraction = bool(re.search(r"\b\d+/\d+\b", ctx.claim))
    if has_verbal_fraction or has_numeric_fraction:
        return Tier25Result(
            has_conflict=False,
            escalate_reason="fraction",
            evidence_bundle=build_evidence_bundle(ctx, chunks),
        )
        
    claim_ranges = extract_ranges(claim_text)
    temp_claim_text = claim_text
    for lo, hi, raw in claim_ranges:
        idx = temp_claim_text.find(raw)
        if idx != -1:
            temp_claim_text = temp_claim_text[:idx] + (' ' * len(raw)) + temp_claim_text[idx + len(raw):]
            
    claim_numbers = _extract_numerical_values(temp_claim_text, is_claim=True)
    
    # If no numbers and no ranges remain in the claim, return Tier25Result(has_conflict=False).
    if not claim_numbers and not claim_ranges:
        return Tier25Result(has_conflict=False, evidence_bundle=build_evidence_bundle(ctx, chunks))

    # Guard: insufficient metric context
    if not _has_sufficient_metric_context(ctx.claim):
        return Tier25Result(has_conflict=False, evidence_bundle=build_evidence_bundle(ctx, chunks))
    
    # Extract claim floats
    claim_floats = [num.value for num, _, _ in claim_numbers]
    
    # Extract contextual years from claim (using original ctx.claim)
    claim_years = set(extract_contextual_years(ctx.claim))

    # Check for range in claim
    is_range_claim = len(claim_ranges) > 0

    claim_ranges_with_units = []
    for claim_lo, claim_hi, claim_raw in claim_ranges:
        idx = claim_text.find(claim_raw)
        if idx != -1:
            end_idx = idx + len(claim_raw)
            raw_unit = _determine_range_unit(claim_raw, claim_text, idx, end_idx)
            unit = _get_effective_unit(claim_raw, raw_unit)
            claim_ranges_with_units.append((claim_lo, claim_hi, claim_raw, unit))
        else:
            unit = _get_effective_unit(claim_raw)
            claim_ranges_with_units.append((claim_lo, claim_hi, claim_raw, unit))

    checks = []
    conflict_found = False
    conflict_citation = None
    evidence_bundle = build_evidence_bundle(ctx, chunks)

    # Year conflict check loop: aggregate across all chunks first
    all_source_years = set()
    for chunk in chunks:
        all_source_years.update(extract_contextual_years(chunk.text_content))

    if claim_years and not claim_years.issubset(all_source_years):
        conflict_found = True
        # Find the first year-bearing chunk to build the conflict_citation.
        for chunk in chunks:
            chunk_years = set(extract_contextual_years(chunk.text_content))
            if chunk_years:
                excerpt_result = extract_excerpt_from_chunk(chunk, _YEAR_CONTEXT_PATTERN)
                if excerpt_result:
                    excerpt_text, start, end = excerpt_result
                    conflict_citation = Citation(
                        source_id=chunk.source_id,
                        excerpt=excerpt_text,
                        excerpt_char_start=chunk.char_start + start,
                        excerpt_char_end=chunk.char_start + end,
                    )
                    break

    claim_year_floats = {float(y) for y in claim_years}
    non_year_claim_numbers = [
        (num, raw, start) for num, raw, start in claim_numbers
        if num.value not in claim_year_floats
    ]
    matched_claim_keys = set()
    
    def is_year_val(val: float) -> bool:
        return val in claim_year_floats or (1900.0 <= val <= 2100.0 and val.is_integer())

    for chunk in chunks:
        # Preprocess chunk
        chunk_text = chunk.text_content
        chunk_text = mask_structural(chunk_text)
        chunk_text = normalize_accounting_negatives(chunk_text)
        chunk_text = normalize_eu_numbers(chunk_text)
        
        if is_range_claim:
            chunk_ranges = extract_ranges(chunk_text)
            temp_chunk_text = chunk_text
            for lo, hi, raw in chunk_ranges:
                idx = temp_chunk_text.find(raw)
                if idx != -1:
                    temp_chunk_text = temp_chunk_text[:idx] + (' ' * len(raw)) + temp_chunk_text[idx + len(raw):]
            
            chunk_numbers = _extract_numerical_values(temp_chunk_text, is_claim=False)
            
            chunk_ranges_with_units = []
            for chunk_lo, chunk_hi, chunk_raw in chunk_ranges:
                idx = chunk_text.find(chunk_raw)
                if idx != -1:
                    end_idx = idx + len(chunk_raw)
                    raw_unit = _determine_range_unit(chunk_raw, chunk_text, idx, end_idx)
                    unit = _get_effective_unit(chunk_raw, raw_unit)
                    chunk_ranges_with_units.append((chunk_lo, chunk_hi, chunk_raw, unit))
                else:
                    unit = _get_effective_unit(chunk_raw)
                    chunk_ranges_with_units.append((chunk_lo, chunk_hi, chunk_raw, unit))
            
            chunk_numbers_with_units = []
            for chunk_num, chunk_raw, chunk_start in chunk_numbers:
                unit = _get_effective_unit(chunk_raw, chunk_num.unit)
                chunk_numbers_with_units.append((chunk_num.value, chunk_raw, unit))
                
            # Check unit mismatches
            for claim_lo, claim_hi, claim_raw, claim_unit in claim_ranges_with_units:
                if is_year_val(claim_lo) and is_year_val(claim_hi):
                    continue
                
                for chunk_lo, chunk_hi, chunk_raw, chunk_unit in chunk_ranges_with_units:
                    mismatch = _check_unit_mismatch_custom(claim_unit, chunk_unit)
                    if mismatch:
                        return Tier25Result(
                            has_conflict=False,
                            escalate_reason=mismatch,
                            evidence_bundle=evidence_bundle,
                            conflict_citation=conflict_citation,
                            numerical_checks=checks,
                        )
                for chunk_val, chunk_raw, chunk_unit in chunk_numbers_with_units:
                    mismatch = _check_unit_mismatch_custom(claim_unit, chunk_unit)
                    if mismatch:
                        return Tier25Result(
                            has_conflict=False,
                            escalate_reason=mismatch,
                            evidence_bundle=evidence_bundle,
                            conflict_citation=conflict_citation,
                            numerical_checks=checks,
                        )
            
            # Check if chunk has no ranges and no numbers; if so, escalate
            if not chunk_ranges_with_units and not chunk_numbers_with_units:
                return Tier25Result(
                    has_conflict=False,
                    escalate_reason="no_source_numbers",
                    evidence_bundle=evidence_bundle,
                    conflict_citation=conflict_citation,
                    numerical_checks=checks,
                )
                
            sources_to_compare = []
            for src_lo, src_hi, src_raw, _ in chunk_ranges_with_units:
                sources_to_compare.append((src_lo, src_hi, src_raw))
            for src_val, src_raw, _ in chunk_numbers_with_units:
                sources_to_compare.append((src_val, src_val, src_raw))
                
            for claim_lo, claim_hi, claim_raw, claim_unit in claim_ranges_with_units:
                if is_year_val(claim_lo) and is_year_val(claim_hi):
                    continue
                
                found_match = False
                found_overlap = False
                best_src_raw = None
                
                for src_lo, src_hi, src_raw in sources_to_compare:
                    if claim_lo <= src_lo and src_hi <= claim_hi:
                        found_match = True
                        best_src_raw = src_raw
                        break
                    elif not RANGE_CONTAINMENT_STRICT:
                        if max(claim_lo, src_lo) <= min(claim_hi, src_hi):
                            found_overlap = True
                            best_src_raw = src_raw
                
                if found_match:
                    checks.append(NumericalCheckResult(
                        claim_number=claim_raw,
                        source_number=best_src_raw,
                        match=True,
                        chunk_id=chunk.chunk_id,
                    ))
                elif found_overlap:
                    return Tier25Result(
                        has_conflict=False,
                        escalate_reason="range_overlap",
                        evidence_bundle=evidence_bundle,
                        conflict_citation=conflict_citation,
                        numerical_checks=checks,
                    )
                else:
                    first_src_raw = sources_to_compare[0][2] if sources_to_compare else ""
                    checks.append(NumericalCheckResult(
                        claim_number=claim_raw,
                        source_number=first_src_raw,
                        match=False,
                        chunk_id=chunk.chunk_id,
                    ))
                    if first_src_raw and conflict_citation is None:
                        excerpt_result = extract_excerpt_from_chunk(chunk, re.escape(first_src_raw))
                        if excerpt_result:
                            excerpt_text, start, end = excerpt_result
                            conflict_citation = Citation(
                                source_id=chunk.source_id,
                                excerpt=excerpt_text,
                                excerpt_char_start=chunk.char_start + start,
                                excerpt_char_end=chunk.char_start + end,
                            )
        else:
            # Non-range claim check
            chunk_numbers = _extract_numerical_values(chunk_text, is_claim=False)
            chunk_floats = [num.value for num, _, _ in chunk_numbers]
            
            for claim_num, claim_raw, claim_start in non_year_claim_numbers:
                claim_float = claim_num.value
                hedge = detect_hedge(claim_text, claim_start)
                matching_num_info = None
                for chunk_num, chunk_raw, chunk_start_val in chunk_numbers:
                    is_match = False
                    if hedge == "approx":
                        if claim_float == 0.0:
                            is_match = abs(chunk_num.value - claim_float) <= APPROX_ZERO_ABS_TOLERANCE
                        else:
                            is_match = abs(chunk_num.value - claim_float) / abs(claim_float) <= APPROX_TOLERANCE
                    elif hedge == "lower":
                        is_match = chunk_num.value >= claim_float
                    elif hedge == "upper":
                        is_match = chunk_num.value <= claim_float
                    else:
                        is_match = chunk_num.value == claim_float
                    if is_match:
                        matching_num_info = (chunk_num, chunk_raw)
                        break

                if chunk_floats and matching_num_info is None:
                    if len(chunk_floats) > 1 and len(non_year_claim_numbers) <= 1:
                        continue
                    chunk_num, chunk_raw, _ = chunk_numbers[0]
                    mismatch = _check_unit_mismatch(claim_num, chunk_num)
                    if mismatch:
                        return Tier25Result(
                            has_conflict=False,
                            escalate_reason=mismatch,
                            evidence_bundle=evidence_bundle,
                            conflict_citation=conflict_citation,
                            numerical_checks=checks,
                        )
                    checks.append(NumericalCheckResult(
                        claim_number=claim_raw,
                        source_number=chunk_raw,
                        match=False,
                        chunk_id=chunk.chunk_id,
                    ))
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
                elif chunk_floats and matching_num_info is not None:
                    matched_claim_keys.add((claim_float, claim_start))
                    chunk_num, chunk_raw = matching_num_info
                    mismatch = _check_unit_mismatch(claim_num, chunk_num)
                    if mismatch:
                        return Tier25Result(
                            has_conflict=False,
                            escalate_reason=mismatch,
                            evidence_bundle=evidence_bundle,
                            conflict_citation=conflict_citation,
                            numerical_checks=checks,
                        )
                    checks.append(NumericalCheckResult(
                        claim_number=claim_raw,
                        source_number=chunk_raw if chunk_raw else str(claim_float),
                        match=True,
                        chunk_id=chunk.chunk_id,
                    ))

    if not conflict_found:
        if not is_range_claim and claim_floats:
            claim_keys = [(num.value, start) for num, _, start in non_year_claim_numbers]
            if all(key in matched_claim_keys for key in claim_keys):
                conflict_citation = None
            else:
                if conflict_citation is not None:
                    conflict_found = True

    if is_range_claim:
        conflict_found = any(not c.match for c in checks) or conflict_found

    return Tier25Result(
        has_conflict=conflict_found,
        verification_method="tier25_numerical",
        evidence_bundle=evidence_bundle,
        conflict_citation=conflict_citation,
        numerical_checks=checks,
    )
