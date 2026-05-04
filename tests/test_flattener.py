"""Tests for dict_to_string_flattener from groundguard.core.verifier (T-29a)."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Example 1: flat dict
# ---------------------------------------------------------------------------

def test_flat_dict():
    """{"revenue": "5M"} -> "revenue: 5M" """
    from groundguard.core.verifier import dict_to_string_flattener

    result = dict_to_string_flattener({"revenue": "5M"})
    assert result == "revenue: 5M"


# ---------------------------------------------------------------------------
# Example 2: nested dict (dot notation)
# ---------------------------------------------------------------------------

def test_nested_dict_dot_notation():
    """{"company": {"q3": {"revenue": "5M"}}} -> "company.q3.revenue: 5M" """
    from groundguard.core.verifier import dict_to_string_flattener

    result = dict_to_string_flattener({"company": {"q3": {"revenue": "5M"}}})
    assert result == "company.q3.revenue: 5M"


# ---------------------------------------------------------------------------
# Example 3: list of scalars (bracket notation)
# ---------------------------------------------------------------------------

def test_list_of_scalars_bracket_notation():
    """{"risks": ["regulatory", "market"]} -> "risks[0]: regulatory\nrisks[1]: market" """
    from groundguard.core.verifier import dict_to_string_flattener

    result = dict_to_string_flattener({"risks": ["regulatory", "market"]})
    assert result == "risks[0]: regulatory\nrisks[1]: market"


# ---------------------------------------------------------------------------
# Example 4: list of dicts
# ---------------------------------------------------------------------------

def test_list_of_dicts():
    """{"items": [{"name": "A"}, {"name": "B"}]} -> "items[0].name: A\nitems[1].name: B" """
    from groundguard.core.verifier import dict_to_string_flattener

    result = dict_to_string_flattener({"items": [{"name": "A"}, {"name": "B"}]})
    assert result == "items[0].name: A\nitems[1].name: B"
