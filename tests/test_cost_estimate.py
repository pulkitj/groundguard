import pytest


def test_estimate_verify_faithfulness_cost_basic():
    from groundguard.cost_estimate import estimate_verify_faithfulness_cost
    est = estimate_verify_faithfulness_cost(
        claim="Revenue was $4.2M in Q3.",
        sources=["Annual report states revenue of 4.2 million USD in Q3."],
        model="gpt-4o-mini",
    )
    assert isinstance(est, float)
    assert est > 0.0
    assert est < 0.10  # sanity: single short claim < 10 cents


def test_estimate_verify_faithfulness_cost_majority_vote():
    from groundguard.cost_estimate import estimate_verify_faithfulness_cost
    from groundguard.profiles import STRICT_PROFILE
    est_single = estimate_verify_faithfulness_cost(
        claim="x", sources=["y"], model="gpt-4o-mini"
    )
    est_majority = estimate_verify_faithfulness_cost(
        claim="x", sources=["y"], model="gpt-4o-mini", profile=STRICT_PROFILE
    )
    # STRICT_PROFILE.majority_vote=3 -> ~3x higher estimate
    assert est_majority >= est_single * 2


def test_estimate_verify_analysis_cost_basic():
    from groundguard.cost_estimate import estimate_verify_analysis_cost
    est = estimate_verify_analysis_cost(
        output="Sentence one. Sentence two.",
        sources=["supporting text"],
        model="gpt-4o-mini",
    )
    assert isinstance(est, float)
    assert est > 0.0


def test_estimate_verify_analysis_cost_scales_with_sentences():
    from groundguard.cost_estimate import estimate_verify_analysis_cost
    est_short = estimate_verify_analysis_cost(
        output="One sentence.", sources=["x"], model="gpt-4o-mini"
    )
    est_long = estimate_verify_analysis_cost(
        output=" ".join([f"Sentence {i}." for i in range(10)]),
        sources=["x"], model="gpt-4o-mini",
    )
    assert est_long > est_short


def test_estimate_returns_zero_for_unknown_model():
    from groundguard.cost_estimate import estimate_verify_faithfulness_cost
    est = estimate_verify_faithfulness_cost(
        claim="x", sources=["y"], model="unknown/custom-model"
    )
    assert est == 0.0  # graceful fallback: unknown pricing -> 0


def test_estimate_cost_breakdown_field():
    from groundguard.cost_estimate import estimate_verify_faithfulness_cost, CostEstimate
    est = estimate_verify_faithfulness_cost(
        claim="x", sources=["y"], model="gpt-4o-mini", return_breakdown=True
    )
    assert isinstance(est, CostEstimate)
    assert est.input_tokens > 0
    assert est.output_tokens > 0
    assert est.total_usd == pytest.approx(est.input_cost_usd + est.output_cost_usd, rel=1e-5)
