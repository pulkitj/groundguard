"""Tests for Phase 19 new data models: Citation, ContextualizedClaimUnit, GroundingResult, VerificationAuditRecord."""
import pytest


# Citation char-offset fields
def test_citation_has_excerpt_fields():
    from groundguard.models.result import Citation
    c = Citation(source_id="s1", excerpt="foo", excerpt_char_start=0,
                 excerpt_char_end=3, citation_confidence=0.9)
    assert c.excerpt == "foo"
    assert c.citation_confidence == 0.9

def test_citation_excerpt_optional():
    from groundguard.models.result import Citation
    c = Citation(source_id="s1")
    assert c.excerpt is None
    assert c.citation_confidence is None

# ContextualizedClaimUnit
def test_contextualized_claim_unit_defaults():
    from groundguard.models.result import ContextualizedClaimUnit
    u = ContextualizedClaimUnit(display_text="X", claim_text="X")
    assert u.verification_status == "PENDING"
    assert u.citation is None
    assert u.numerical_fast_exit is False

def test_contextualized_claim_unit_fields():
    from groundguard.models.result import ContextualizedClaimUnit
    u = ContextualizedClaimUnit(
        display_text="Revenue was $4.2M",
        claim_text="Revenue was $4.2M",
        enrichment_method="sentence_split",
        structural_type="financial",
        heading_path=["Section 1"],
        verification_status="VERIFIED",
        confidence=0.95,
        numerical_fast_exit=True,
    )
    assert u.structural_type == "financial"
    assert u.numerical_fast_exit is True

# GroundingResult
def test_grounding_result_defaults():
    from groundguard.models.result import GroundingResult
    gr = GroundingResult(is_grounded=True, score=0.9, status="GROUNDED",
                         evaluation_method="sentence_entailment")
    assert gr.unit_results == []
    assert gr.audit_records is None

def test_grounding_result_statuses():
    from groundguard.models.result import GroundingResult
    for s in ("GROUNDED", "PARTIALLY_GROUNDED", "NOT_GROUNDED", "ERROR"):
        gr = GroundingResult(is_grounded=False, score=0.0, status=s,
                             evaluation_method="claim_extraction")
        assert gr.status == s

def test_grounding_result_evaluation_methods():
    from groundguard.models.result import GroundingResult
    for m in ("sentence_entailment", "claim_extraction"):
        gr = GroundingResult(is_grounded=True, score=1.0, status="GROUNDED",
                             evaluation_method=m)
        assert gr.evaluation_method == m

# VerificationAuditRecord
def test_audit_record_fields():
    from groundguard.models.result import VerificationAuditRecord
    import dataclasses, json
    r = VerificationAuditRecord(
        boundary_id="abc123def456",
        claim_text="claim",
        verdict="VERIFIED",
        tier_path=["tier2_lexical"],
        model="gpt-4o",
        cost_usd=0.001,
        timestamp_utc="2026-01-01T00:00:00Z",
        profile_name="general",
    )
    assert r.majority_vote_triggered is False
    assert r.vote_breakdown is None
    d = dataclasses.asdict(r)
    json.dumps(d)  # must be JSON-serializable

def test_audit_record_optional_fields():
    from groundguard.models.result import VerificationAuditRecord
    r = VerificationAuditRecord(
        boundary_id="abc123def456",
        claim_text="claim",
        verdict="VERIFIED",
        tier_path=["tier3_llm"],
        model="gpt-4o",
        cost_usd=0.0,
        timestamp_utc="2026-01-01T00:00:00Z",
        profile_name="strict",
        majority_vote_triggered=True,
        vote_breakdown={"VERIFIED": 2, "UNVERIFIABLE": 1},
    )
    assert r.majority_vote_triggered is True
    assert r.vote_breakdown == {"VERIFIED": 2, "UNVERIFIABLE": 1}

def test_audit_record_profile_override_fields():
    from groundguard.models.result import VerificationAuditRecord
    r = VerificationAuditRecord(
        boundary_id="abc123def456",
        claim_text="claim",
        verdict="VERIFIED",
        tier_path=["tier3_llm"],
        model="gpt-4o",
        cost_usd=0.001,
        timestamp_utc="2026-01-01T00:00:00Z",
        profile_name="strict",
        profile_override=True,
        effective_faithfulness_threshold=0.40,
    )
    assert r.profile_override is True
    assert r.effective_faithfulness_threshold == 0.40

def test_audit_record_no_override_defaults():
    from groundguard.models.result import VerificationAuditRecord
    r = VerificationAuditRecord(
        boundary_id="abc123def456",
        claim_text="claim",
        verdict="VERIFIED",
        tier_path=["tier3_llm"],
        model="gpt-4o",
        cost_usd=0.001,
        timestamp_utc="2026-01-01T00:00:00Z",
        profile_name="strict",
    )
    assert r.profile_override is False
    assert r.effective_faithfulness_threshold is None

# Source schema (ENH-ADD-01)
def test_source_type_default_is_document():
    """source_type must default to the string 'document', not None."""
    from groundguard.models.result import Source
    s = Source(content="text", source_id="doc.pdf")
    assert s.source_type == "document"

def test_source_type_is_not_optional():
    """source_type field must be str, not str|None."""
    from groundguard.models.result import Source
    import dataclasses
    field = next(f for f in dataclasses.fields(Source) if f.name == "source_type")
    assert "None" not in str(field.type)

def test_source_prev_next_context_none_by_default():
    from groundguard.models.result import Source
    s = Source(content="x", source_id="s")
    assert s.prev_context is None
    assert s.next_context is None

def test_source_section_id_none_by_default():
    from groundguard.models.result import Source
    assert Source(content="x", source_id="s").section_id is None

def test_source_derived_from_llm_false_by_default():
    from groundguard.models.result import Source
    assert Source(content="x", source_id="s").derived_from_llm is False

def test_source_original_document_id_none_by_default():
    from groundguard.models.result import Source
    assert Source(content="x", source_id="s").original_document_id is None

# GroundingResult full-field coverage
def test_grounding_result_all_fields():
    """Verify all 17 GroundingResult fields per engineering_design_v4.md §2d."""
    from groundguard.models.result import GroundingResult
    gr = GroundingResult(
        is_grounded=True, score=0.9, status="GROUNDED",
        evaluation_method="sentence_entailment",
        total_units=3, grounded_units=3, ungrounded_units=0, unverifiable_units=0,
        unit_results=[], sources_used=["s1"], total_cost_usd=0.001, summary="ok",
    )
    assert gr.provenance_warning is False
    assert gr.units_capped is False
    assert gr.numerical_fast_exits == 0
    assert gr.majority_votes_triggered == 0
    assert gr.audit_records is None
