"""Tests for Phase 20: VerificationProfile and profile constants."""
import pytest


def test_strict_profile_fields():
    from groundguard.profiles import STRICT_PROFILE
    assert STRICT_PROFILE.faithfulness_threshold == 0.97
    assert STRICT_PROFILE.tier2_lexical_threshold == 2.0
    assert STRICT_PROFILE.bm25_top_k == 6
    assert STRICT_PROFILE.majority_vote is True
    assert STRICT_PROFILE.audit is True

def test_general_profile_fields():
    from groundguard.profiles import GENERAL_PROFILE
    assert GENERAL_PROFILE.faithfulness_threshold == 0.80
    assert GENERAL_PROFILE.tier2_lexical_threshold == 0.85
    assert GENERAL_PROFILE.bm25_top_k == 3
    assert GENERAL_PROFILE.majority_vote is False
    assert GENERAL_PROFILE.audit is False

def test_research_profile_fields():
    from groundguard.profiles import RESEARCH_PROFILE
    assert RESEARCH_PROFILE.faithfulness_threshold == 0.70
    assert RESEARCH_PROFILE.tier2_lexical_threshold == 0.85
    assert RESEARCH_PROFILE.bm25_top_k == 4
    assert RESEARCH_PROFILE.majority_vote is False
    assert RESEARCH_PROFILE.audit is False

def test_profile_is_frozen():
    from groundguard.profiles import GENERAL_PROFILE
    import dataclasses
    assert dataclasses.fields(GENERAL_PROFILE)  # is a dataclass
    try:
        GENERAL_PROFILE.faithfulness_threshold = 0.5
        assert False, "should be frozen"
    except (dataclasses.FrozenInstanceError, AttributeError):
        pass

def test_profile_name_field():
    from groundguard.profiles import STRICT_PROFILE, GENERAL_PROFILE, RESEARCH_PROFILE
    assert STRICT_PROFILE.name == "strict"
    assert GENERAL_PROFILE.name == "general"
    assert RESEARCH_PROFILE.name == "research"
