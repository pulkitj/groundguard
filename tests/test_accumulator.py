"""Tests for GroundingAccumulator and SourceAccumulator — T-107."""
from groundguard.models.result import GroundingResult, Source


def _gr(is_grounded: bool, score: float, grounded_units: int = 1, ungrounded_units: int = 0) -> GroundingResult:
    return GroundingResult(
        is_grounded=is_grounded,
        score=score,
        status="GROUNDED" if is_grounded else "NOT_GROUNDED",
        evaluation_method="sentence_entailment",
        total_units=grounded_units + ungrounded_units,
        grounded_units=grounded_units,
        ungrounded_units=ungrounded_units,
    )


# ---------------------------------------------------------------------------
# GroundingAccumulator
# ---------------------------------------------------------------------------

def test_grounding_accumulator_empty_score_is_zero():
    from groundguard.loaders.accumulator import GroundingAccumulator
    acc = GroundingAccumulator()
    assert acc.overall_score == 0.0


def test_grounding_accumulator_is_grounded_empty_is_true():
    from groundguard.loaders.accumulator import GroundingAccumulator
    acc = GroundingAccumulator()
    assert acc.is_grounded is True  # all() of empty = True


def test_grounding_accumulator_overall_score_weighted():
    from groundguard.loaders.accumulator import GroundingAccumulator
    acc = GroundingAccumulator()
    # Result A: 1 unit, grounded (score=1.0)
    acc.add(_gr(True, 1.0, grounded_units=1, ungrounded_units=0))
    # Result B: 49 units, 49 ungrounded (score=0.0)
    acc.add(_gr(False, 0.0, grounded_units=0, ungrounded_units=49))
    # Overall score should be 1 / 50 = 0.02
    assert abs(acc.overall_score - 0.02) < 1e-9


def test_grounding_accumulator_is_grounded_false_if_any_not():
    from groundguard.loaders.accumulator import GroundingAccumulator
    acc = GroundingAccumulator()
    acc.add(_gr(True, 0.9))
    acc.add(_gr(False, 0.3))
    assert acc.is_grounded is False


def test_grounding_accumulator_reset_clears():
    from groundguard.loaders.accumulator import GroundingAccumulator
    acc = GroundingAccumulator()
    acc.add(_gr(True, 0.9))
    acc.reset()
    assert acc.overall_score == 0.0


def test_grounding_accumulator_falls_back_to_score_average_when_unit_counts_absent():
    """verify() results use claim_extraction and do not populate grounded/ungrounded_units.
    When all results have zero unit counts, overall_score must fall back to the
    simple average of each result's .score rather than returning 0.0.
    """
    from groundguard.loaders.accumulator import GroundingAccumulator
    # GroundingResults from verify() leave grounded_units=0, ungrounded_units=0
    r1 = GroundingResult(
        is_grounded=True, score=0.9, status="GROUNDED",
        evaluation_method="claim_extraction",
    )
    r2 = GroundingResult(
        is_grounded=True, score=0.7, status="GROUNDED",
        evaluation_method="claim_extraction",
    )
    acc = GroundingAccumulator()
    acc.add(r1)
    acc.add(r2)
    assert abs(acc.overall_score - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# SourceAccumulator
# ---------------------------------------------------------------------------

def test_source_accumulator_deduplicates_by_source_id():
    from groundguard.loaders.accumulator import SourceAccumulator
    acc = SourceAccumulator()
    s = Source(source_id="s1", content="x")
    acc.add([s, s])
    assert len(acc) == 1


def test_source_accumulator_mark_llm_derived():
    from groundguard.loaders.accumulator import SourceAccumulator
    acc = SourceAccumulator()
    acc.add([Source(source_id="s1", content="x")], mark_llm_derived=True)
    assert acc.has_llm_derived() is True


def test_source_accumulator_no_llm_derived_by_default():
    from groundguard.loaders.accumulator import SourceAccumulator
    acc = SourceAccumulator()
    acc.add([Source(source_id="s1", content="x")])
    assert acc.has_llm_derived() is False


def test_source_accumulator_boundary_context_same_base():
    from groundguard.loaders.accumulator import SourceAccumulator
    acc = SourceAccumulator()
    s1 = Source(source_id="doc::chunk_1", content="Revenue grew. Net income rose.")
    s2 = Source(source_id="doc::chunk_2", content="Costs fell. EBITDA improved.")
    acc.add([s1, s2], populate_boundary_context=True)
    sources = acc.all_sources()
    assert sources[0].next_context == "Costs fell."
    assert sources[1].prev_context == "Net income rose."


def test_source_accumulator_no_boundary_context_across_different_docs():
    from groundguard.loaders.accumulator import SourceAccumulator
    acc = SourceAccumulator()
    acc.add([
        Source(source_id="doc_a::chunk_1", content="Revenue grew."),
        Source(source_id="doc_b::chunk_1", content="Costs fell."),
    ], populate_boundary_context=True)
    sources = acc.all_sources()
    assert sources[0].next_context is None
    assert sources[1].prev_context is None


def test_source_accumulator_clear():
    from groundguard.loaders.accumulator import SourceAccumulator
    acc = SourceAccumulator()
    acc.add([Source(source_id="s1", content="x")])
    acc.clear()
    assert len(acc) == 0


def test_source_accumulator_chaining():
    from groundguard.loaders.accumulator import SourceAccumulator
    acc = SourceAccumulator()
    result = acc.add([Source(source_id="s1", content="x")])
    assert result is acc
