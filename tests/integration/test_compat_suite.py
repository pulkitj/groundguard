# tests/integration/test_compat_suite.py
"""
Multi-model compatibility suite — @pytest.mark.compat

Runs the complete real suite (T-34 through T-51, 18 fixtures) against every
model in compat_models.ALL_COMPAT_MODELS. Each function delegates directly to
the corresponding test_real_suite function so that contract changes are
automatically reflected here without duplication.

Failure classification (human judgment, not encoded in test logic):
  - Pipeline/adapter bug:  fails across most models or on model-agnostic tests
    (test_fixture_a, test_fixture_o_tier1_failure — these are deterministic)
  - Model limitation:      fails for a specific model on reasoning-quality tests
    (test_fixture_c, test_fixture_d, test_fixture_k, etc.)

Run all available models (API keys auto-determine which are skipped):
    pytest -m compat -v --timeout=300 -p no:cov

Run one model:
    pytest -m compat -v -k "nim-llama33" --timeout=300 -p no:cov

Run Ollama only (no API keys needed):
    pytest -m compat -v -k "ollama" --timeout=600 -p no:cov
"""
from __future__ import annotations

import pytest

from tests.integration import test_real_suite
from tests.integration.compat_models import CompatModel


# ---------------------------------------------------------------------------
# T-34 / Fixture A: Perfect Match — BM25 lexical pass, no LLM call
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_a_perfect_match(compat_model: CompatModel):
    test_real_suite.test_fixture_a_perfect_match(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-35 / Fixture B: Financial Loophole — 10× numeric error → CONTRADICTED
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_b_financial_loophole_wrong_number(compat_model: CompatModel):
    test_real_suite.test_fixture_b_financial_loophole_wrong_number(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-36 / Fixture C: Casual Wording & Typos — must not produce CONTRADICTED
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_c_casual_wording_typos(compat_model: CompatModel):
    test_real_suite.test_fixture_c_casual_wording_typos(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-37 / Fixture D: Formatting Drift ($4M vs "four million") — not CONTRADICTED
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_d_formatting_drift(compat_model: CompatModel):
    test_real_suite.test_fixture_d_formatting_drift(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-38 / Fixture E: All-Low Score Paraphrase — Branch C escalation
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_e_all_low_score_paraphrase(compat_model: CompatModel):
    test_real_suite.test_fixture_e_all_low_score_paraphrase(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-39 / Fixture F: Gibberish / Unclear Pronouns — must not be VERIFIED
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_f_gibberish_unclear_pronouns(compat_model: CompatModel):
    test_real_suite.test_fixture_f_gibberish_unclear_pronouns(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-40 / Fixture G: Out-of-Domain Hallucination — must not be VERIFIED
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_g_out_of_domain_hallucination(compat_model: CompatModel):
    test_real_suite.test_fixture_g_out_of_domain_hallucination(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-41 / Fixture H: Neutral Entailment Coverage Independence
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_h_neutral_entailment_coverage_independence(compat_model: CompatModel):
    test_real_suite.test_fixture_h_neutral_entailment_coverage_independence(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-42 / Fixture I: Valid Inferential Synthesis — must not be CONTRADICTED
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_i_valid_inferential_synthesis(compat_model: CompatModel):
    test_real_suite.test_fixture_i_valid_inferential_synthesis(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-43 / Fixture J: Multi-Hop Math ($5M + $10M = $15M) — must not be CONTRADICTED
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_j_multi_hop_math(compat_model: CompatModel):
    test_real_suite.test_fixture_j_multi_hop_math(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-44 / Fixture K: Flawed Inferential Logic — revenue up, costs up more → CONTRADICTED
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_k_flawed_inferential_logic(compat_model: CompatModel):
    test_real_suite.test_fixture_k_flawed_inferential_logic(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-45 / Fixture L: Needle in Haystack — contradiction buried in 8K+ chars
# ---------------------------------------------------------------------------

@pytest.mark.compat
@pytest.mark.timeout(300)
def test_fixture_l_needle_in_haystack(compat_model: CompatModel):
    test_real_suite.test_fixture_l_needle_in_haystack(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-46 / Fixture M: Batch Scale — 5-item batch, no uncaught exceptions
# ---------------------------------------------------------------------------

@pytest.mark.compat
@pytest.mark.timeout(300)
async def test_fixture_m_batch_scale(compat_model: CompatModel):
    await test_real_suite.test_fixture_m_batch_scale(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-47 / Fixture N: Tier 1 Pass — valid evidence passes gate; pipeline continues
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_n_tier1_pass_pipeline_continues(compat_model: CompatModel):
    test_real_suite.test_fixture_n_tier1_pass_pipeline_continues(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-48 / Fixture O: Tier 1 Failure — fabricated evidence raises HallucinatedEvidenceError
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_o_tier1_failure_hallucinated_evidence_error(compat_model: CompatModel):
    test_real_suite.test_fixture_o_tier1_failure_hallucinated_evidence_error(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-49 / Fixture P: Cross-Source Synthesis — $3M + $5M = $8M, both sources cited
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_p_cross_source_synthesis(compat_model: CompatModel):
    test_real_suite.test_fixture_p_cross_source_synthesis(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-50 / Fixture Q: Cross-Source Contradiction — 90-day terms vs 60 and 30 day sources
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_q_cross_source_contradiction(compat_model: CompatModel):
    test_real_suite.test_fixture_q_cross_source_contradiction(compat_model.model_str, compat_model.api_base)


# ---------------------------------------------------------------------------
# T-51 / Fixture R: Branch B Partial Match — goes through LLM, not lexical pass
# ---------------------------------------------------------------------------

@pytest.mark.compat
def test_fixture_r_branch_b_partial_match(compat_model: CompatModel):
    test_real_suite.test_fixture_r_branch_b_partial_match(compat_model.model_str, compat_model.api_base)
