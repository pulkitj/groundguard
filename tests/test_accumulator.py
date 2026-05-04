"""Tests for GroundingAccumulator and SourceAccumulator — T-107."""
from groundguard.models.result import GroundingResult, Source


def _gr(is_grounded: bool, score: float) -> GroundingResult:
    return GroundingResult(
        is_grounded=is_grounded,
        score=score,
        status="GROUNDED" if is_grounded else "NOT_GROUNDED",
        evaluation_method="sentence_entailment",
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


def test_grounding_accumulator_overall_score_averages():
    from groundguard.loaders.accumulator import GroundingAccumulator
    acc = GroundingAccumulator()
    acc.add(_gr(True, 0.9))
    acc.add(_gr(True, 0.7))
    assert abs(acc.overall_score - 0.8) < 1e-9


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
