# groundguard: Advisory Log

## Module: tier25-p1 — Code Review Cycle 1 — 2026-06-11
- spec_compliance | `extract_composite_numbers` replaces matches with spaces of equal length rather than deleting them from the string. | WHY: This preserves index alignments for subsequent regex operations (such as base extraction and citation offsets) but technically deviates from the literal word "removed" in the spec.
- scale_performance | `_EU_UNGROUPED_DECIMAL_RE` restricts ungrouped decimal normalization to exactly two digits after the comma (`\d{2}`). | WHY: Single-digit or >2 digit decimals (e.g., `12,3` or `12,345`) will not be normalized.

## Module: tier25-p2 — Code Review Cycle 2 — 2026-06-11
- risks | Float conversion of extremely large values or scientific notation strings (e.g. `1e999999`) in `run()` can raise `OverflowError` rather than `ValueError`, which is not caught and would crash the program. | WHY: The existing `try...except` block only handles `ValueError`.
- test_coverage_gaps | `tests/test_tier25.py` is missing tests for US thousands separators without decimals (e.g., verifying that `'1,000'` or `'1,234'` normalizes to `1000.0` or `1234.0`). | WHY: There is no test in `tests/test_tier25.py` verifying this behavior.

## Module: tier25-p2 — Code Review Cycle 3 — 2026-06-11
- dead_code | The local variable `is_usd_style_prefix` is defined and set but never read. | WHY: Leftover from removing the USD prefix guard.
- dead_code | The original docstring of `_normalise_number()` was removed. | WHY: Cleaned up but not replaced.

## Module: tier25-p2 — Code Review Cycle 4 — 2026-06-11
- dead_code | The local variable `is_usd_style_prefix` is defined and set but never read. | WHY: Still present in the merged codebase.
- test_coverage_gaps | `tests/test_tier25.py` is missing tests for mixed casing and spaced suffixes (e.g., `'50 BPS'`, `'4.2 million'`). | WHY: There are no tests in `tests/test_tier25.py` checking casing and spacing permutations.

## Module: tier25-p3 — Test Review Cycle 5 — 2026-06-20
- spec_compliance | Minor spec contradiction: the detailed spec section (line 845) states that mismatching units should escalate (no conflict), whereas the Phase 3 task summary (lines 1164 & 1175) states it should conflict. | WHY: Test suite has minor inconsistencies: test_t25p3_unit_label_mismatch_escalates expects escalation only, while test_t25p3_differing_rate_denominators_conflict_or_escalate allows either conflict or escalation.
