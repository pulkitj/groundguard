"""Real Suite integration tests — Phase 12 (T-34 to T-51).

All tests require a running Ollama instance with qwen3:30b (or the model
specified via --llm-model / LLM_TEST_MODEL env var).

Run with:
    pytest tests/integration/ -m llm -q --timeout=300
"""
from __future__ import annotations

import asyncio

import pytest

from agentic_verifier.core.verifier import verify, verify_batch_async
from agentic_verifier.exceptions import HallucinatedEvidenceError
from agentic_verifier.models.internal import ClaimInput
from agentic_verifier.models.result import Source

VALID_STATUSES = {"VERIFIED", "CONTRADICTED", "UNVERIFIABLE", "PARSE_ERROR", "ERROR", "SKIPPED_DUE_TO_COST"}


# ---------------------------------------------------------------------------
# T-34 — Fixture A: Perfect Match
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_a_perfect_match(llm_model: str):
    """T-34: Perfect verbatim match should be VERIFIED via tier2_lexical (BM25 high-confidence, no LLM call)."""
    claim = "The Q3 revenue was $5 million."
    sources = [Source(content="The Q3 revenue was $5 million.", source_id="report.pdf")]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status == "VERIFIED"
    assert result.verification_method == "tier2_lexical"


# ---------------------------------------------------------------------------
# T-35 — Fixture B: Financial Loophole (wrong number)
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_b_financial_loophole_wrong_number(llm_model: str):
    """T-35: Claim with wrong number (300% vs 30%) should be CONTRADICTED via tier3_llm."""
    claim = "Revenue grew by 300% year-over-year."
    sources = [Source(content="Revenue grew by 30% year-over-year.", source_id="report.pdf")]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status == "CONTRADICTED"
    assert result.verification_method == "tier3_llm"


# ---------------------------------------------------------------------------
# T-36 — Fixture C: Casual Wording / Typos
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_c_casual_wording_typos(llm_model: str):
    """T-36: Casual phrasing with typo should still be VERIFIED (BM25 + LLM bridges imperfection)."""
    claim = "The companys revenue was 5 million bucks in Q3"  # typo + casual
    sources = [Source(content="The company's Q3 revenue was $5 million.", source_id="report.pdf")]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status == "VERIFIED"


# ---------------------------------------------------------------------------
# T-37 — Fixture D: Formatting Drift
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_d_formatting_drift(llm_model: str):
    """T-37: Formatting difference ($4M vs 'four million dollars') should NOT produce false CONTRADICTED."""
    claim = "Revenue was four million dollars."
    sources = [Source(content="Revenue was $4M.", source_id="report.pdf")]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status in ("VERIFIED", "UNVERIFIABLE"), (
        f"Expected VERIFIED or UNVERIFIABLE for formatting drift, got: {result.status}"
    )
    assert result.status != "CONTRADICTED", (
        "Formatting difference ('four million dollars' vs '$4M') must not produce false CONTRADICTED"
    )


# ---------------------------------------------------------------------------
# T-38 — Fixture E: All-Low Score Paraphrase (Branch C)
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_e_all_low_score_paraphrase(llm_model: str):
    """T-38: Highly paraphrased claim with no lexical overlap (Branch C) should go through LLM."""
    claim = "Profits increased substantially during the period."
    sources = [Source(content="Net income rose significantly in the quarter.", source_id="report.pdf")]

    result = verify(claim=claim, sources=sources, model=llm_model)

    # Branch C escalates to LLM; result can be any valid status
    assert result.verification_method in ("tier3_llm", "tier2_lexical")
    assert result.status in ("VERIFIED", "UNVERIFIABLE", "CONTRADICTED")


# ---------------------------------------------------------------------------
# T-39 — Fixture F: Gibberish / Unclear Pronouns
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_f_gibberish_unclear_pronouns(llm_model: str):
    """T-39: Vague claim with unclear pronouns should be UNVERIFIABLE."""
    claim = "It went up because of the thing with those results."
    sources = [Source(content="Q3 revenue was $5 million.", source_id="report.pdf")]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status == "UNVERIFIABLE"


# ---------------------------------------------------------------------------
# T-40 — Fixture G: Out-of-Domain Hallucination
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_g_out_of_domain_hallucination(llm_model: str):
    """T-40: Claim about product launch not present in financial source should NOT be VERIFIED."""
    claim = "The company launched a new AI product in 2023."
    sources = [Source(
        content="Q3 revenue was $5 million. Net income was $1 million.",
        source_id="report.pdf"
    )]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status in ("CONTRADICTED", "UNVERIFIABLE"), (
        f"Out-of-domain hallucination must not be VERIFIED, got: {result.status}"
    )


# ---------------------------------------------------------------------------
# T-41 — Fixture H: Neutral Entailment Coverage Independence
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_h_neutral_entailment_coverage_independence(llm_model: str):
    """T-41: Vague claim about 'expectations' against specific source — UNVERIFIABLE or VERIFIED acceptable.

    Tests that Neutral entailment maps to UNVERIFIABLE regardless of coverage,
    but VERIFIED is also acceptable if the LLM determines the claim is grounded.
    """
    claim = "The results were within expectations based on industry trends."
    sources = [Source(
        content="Q3 revenue was $5 million, beating analyst consensus of $4.8 million.",
        source_id="report.pdf"
    )]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status in ("UNVERIFIABLE", "VERIFIED"), (
        f"Expected UNVERIFIABLE or VERIFIED for vague claim, got: {result.status}"
    )


# ---------------------------------------------------------------------------
# T-42 — Fixture I: Valid Inferential Synthesis
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_i_valid_inferential_synthesis(llm_model: str):
    """T-42: Inferential claim derivable from multiple data points should be VERIFIED via tier3_llm."""
    claim = "The company is likely on track to meet annual targets based on Q3 performance."
    sources = [Source(
        content="Q3 revenue was $5 million. Annual target is $18 million. Q1 was $4M, Q2 was $4.5M.",
        source_id="report.pdf"
    )]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status == "VERIFIED"
    assert result.verification_method == "tier3_llm"


# ---------------------------------------------------------------------------
# T-43 — Fixture J: Multi-Hop Math
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_j_multi_hop_math(llm_model: str):
    """T-43: Claim requiring arithmetic (Q1 + Q2 = H1 total) should be VERIFIED."""
    claim = "Total H1 revenue was $15 million."
    sources = [Source(
        content="Q1 revenue was $5 million. Q2 revenue was $10 million.",
        source_id="report.pdf"
    )]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status == "VERIFIED"


# ---------------------------------------------------------------------------
# T-44 — Fixture K: Flawed Inferential Logic
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_k_flawed_inferential_logic(llm_model: str):
    """T-44: Claim of improved profitability contradicted by source showing net loss should be CONTRADICTED."""
    claim = "Profitability improved in Q3."
    sources = [Source(
        content="Q3 revenue increased by 20% but operating costs rose by 35%, resulting in a net loss.",
        source_id="report.pdf"
    )]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status == "CONTRADICTED"


# ---------------------------------------------------------------------------
# T-45 — Fixture L: Needle in Haystack
# ---------------------------------------------------------------------------

@pytest.mark.llm
@pytest.mark.timeout(120)
def test_fixture_l_needle_in_haystack(llm_model: str):
    """T-45: Contradicting sentence buried in 8000+ char source should be found — result CONTRADICTED."""
    padding = "The company operates in multiple sectors. " * 200  # ~8200 chars
    contradicting = "The Q3 net profit was negative, with a loss of $2 million."
    source_content = padding + contradicting
    claim = "The company reported a profit of $2 million in Q3."
    sources = [Source(content=source_content, source_id="report.pdf")]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status == "CONTRADICTED"


# ---------------------------------------------------------------------------
# T-46 — Fixture M: Batch Scale
# ---------------------------------------------------------------------------

@pytest.mark.llm
@pytest.mark.timeout(120)
def test_fixture_m_batch_scale(llm_model: str):
    """T-46: Batch of 10 identical-structure claims should all complete without uncaught exceptions."""
    inputs = [
        ClaimInput(
            claim=f"Revenue in period {i} was ${i} million.",
            sources=[Source(
                content=f"Revenue in period {i} was ${i} million.",
                source_id=f"doc{i}.pdf"
            )]
        )
        for i in range(1, 11)
    ]

    results = asyncio.run(
        verify_batch_async(inputs=inputs, model=llm_model, max_concurrency=3, max_spend=5.0)
    )

    assert len(results) == 10
    for result in results:
        assert result.status in VALID_STATUSES, (
            f"Unexpected status in batch result: {result.status}"
        )


# ---------------------------------------------------------------------------
# T-47 — Fixture N: Tier 1 Pass → Pipeline Continues
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_n_tier1_pass_pipeline_continues(llm_model: str):
    """T-47: Valid agent_provided_evidence should pass Tier 1 gate; pipeline continues to produce a result.

    Tier 1 is gate-only — it never produces a terminal result. The verification
    method must NOT be 'tier1_authenticity'.
    """
    claim = "The contract value was $2 million."
    agent_evidence = "contract value was $2 million"
    sources = [Source(
        content="The contract value was $2 million as of Q3.",
        source_id="contract.pdf"
    )]

    result = verify(
        claim=claim,
        sources=sources,
        agent_provided_evidence=agent_evidence,
        model=llm_model
    )

    assert result.status in ("VERIFIED", "UNVERIFIABLE")
    assert result.verification_method != "tier1_authenticity", (
        "Tier 1 is gate-only and must never produce a terminal result"
    )


# ---------------------------------------------------------------------------
# T-48 — Fixture O: Tier 1 Failure → HallucinatedEvidenceError
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_o_tier1_failure_hallucinated_evidence_error(llm_model: str):
    """T-48: Fabricated agent_provided_evidence not found in source should raise HallucinatedEvidenceError."""
    claim = "Revenue was $5 million."
    agent_evidence = "profits skyrocketed to $50 billion"  # fabricated — not in source
    sources = [Source(content="Revenue was $5 million.", source_id="report.pdf")]

    with pytest.raises(HallucinatedEvidenceError):
        verify(
            claim=claim,
            sources=sources,
            agent_provided_evidence=agent_evidence,
            model=llm_model
        )


# ---------------------------------------------------------------------------
# T-49 — Fixture P: Cross-Source Synthesis
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_p_cross_source_synthesis(llm_model: str):
    """T-49: Claim requiring synthesis across two sources should be VERIFIED with both source IDs in sources_used."""
    claim = "Total H1 revenue was $8 million."
    sources = [
        Source(content="Q1 revenue was $3 million.", source_id="q1_report.pdf"),
        Source(content="Q2 revenue was $5 million.", source_id="q2_report.pdf"),
    ]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status == "VERIFIED"
    assert "q1_report.pdf" in result.sources_used, (
        "q1_report.pdf should appear in sources_used for cross-source synthesis"
    )
    assert "q2_report.pdf" in result.sources_used, (
        "q2_report.pdf should appear in sources_used for cross-source synthesis"
    )


# ---------------------------------------------------------------------------
# T-50 — Fixture Q: Cross-Source Contradiction
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_q_cross_source_contradiction(llm_model: str):
    """T-50: Claim of 90-day terms contradicted by both sources (60 and 30 days) should be CONTRADICTED."""
    claim = "The payment terms are 90 days."
    sources = [
        Source(
            content="Payment terms are 60 days per the master agreement.",
            source_id="master.pdf"
        ),
        Source(
            content="Payment terms are 30 days per the amendment.",
            source_id="amendment.pdf"
        ),
    ]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.status == "CONTRADICTED"


# ---------------------------------------------------------------------------
# T-51 — Fixture R: Branch B Partial Match
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_fixture_r_branch_b_partial_match(llm_model: str):
    """T-51: Partial-match claim should go through LLM (Branch B) and be VERIFIED or UNVERIFIABLE."""
    claim = "Quarterly revenue showed improvement."
    sources = [Source(
        content="Q3 revenue was $5 million, up from $4 million in Q2.",
        source_id="report.pdf"
    )]

    result = verify(claim=claim, sources=sources, model=llm_model)

    assert result.verification_method == "tier3_llm", (
        "Partial-match claim should go through Tier 3 LLM, not the lexical pass"
    )
    assert result.status in ("VERIFIED", "UNVERIFIABLE"), (
        f"Expected VERIFIED or UNVERIFIABLE for partial match, got: {result.status}"
    )
