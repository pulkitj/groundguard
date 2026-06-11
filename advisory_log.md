# groundguard: Advisory Log

## Module: tier25-p1 — Code Review Cycle 1 — 2026-06-11
- spec_compliance | `extract_composite_numbers` replaces matches with spaces of equal length rather than deleting them from the string. | WHY: This preserves index alignments for subsequent regex operations (such as base extraction and citation offsets) but technically deviates from the literal word "removed" in the spec.
- scale_performance | `_EU_UNGROUPED_DECIMAL_RE` restricts ungrouped decimal normalization to exactly two digits after the comma (`\d{2}`). | WHY: Single-digit or >2 digit decimals (e.g., `12,3` or `12,345`) will not be normalized.
