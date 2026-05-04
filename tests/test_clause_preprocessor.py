"""Tests for legal.py preprocessors — TDD T-98 (legal suite)."""
import pytest


@pytest.mark.legal
def test_passive_voice_normalizer_exceeded():
    from groundguard.loaders.legal import PassiveVoiceNormalizer
    pv = PassiveVoiceNormalizer()
    result = pv.normalize("X shall not be exceeded by Y")
    assert result == "Y shall not exceed X"


@pytest.mark.legal
def test_passive_voice_normalizer_determined():
    from groundguard.loaders.legal import PassiveVoiceNormalizer
    pv = PassiveVoiceNormalizer()
    result = pv.normalize("X shall be determined by Y")
    assert result == "Y shall determine X"


@pytest.mark.legal
def test_passive_voice_normalizer_defined():
    from groundguard.loaders.legal import PassiveVoiceNormalizer
    pv = PassiveVoiceNormalizer()
    result = pv.normalize("Term is defined as D in Schedule 1")
    assert result == "Schedule 1 defines Term as D"


@pytest.mark.legal
def test_passive_voice_normalizer_unknown_passthrough():
    from groundguard.loaders.legal import PassiveVoiceNormalizer
    pv = PassiveVoiceNormalizer()
    original = "Some unknown passive pattern here"
    assert pv.normalize(original) == original


@pytest.mark.legal
def test_passive_voice_normalizer_failsafe():
    from groundguard.loaders.legal import PassiveVoiceNormalizer
    pv = PassiveVoiceNormalizer()
    result = pv.normalize("valid sentence")
    assert isinstance(result, str)


@pytest.mark.legal
def test_term_registry_from_sources():
    from groundguard.loaders.legal import TermRegistry
    from groundguard.models.result import Source
    src = Source(source_id="s1", content='"Permitted Costs" means costs approved by the board.',
                 source_type="legal_definition")
    registry = TermRegistry.from_sources([src])
    assert "permitted costs" in registry.known_terms()


@pytest.mark.legal
def test_term_registry_resolve_case_insensitive():
    from groundguard.loaders.legal import TermRegistry
    from groundguard.models.result import Source
    src = Source(source_id="s1", content='"Permitted Costs" means approved costs.',
                 source_type="legal_definition")
    registry = TermRegistry.from_sources([src])
    assert registry.resolve("permitted costs") is not None
    assert registry.resolve("Permitted Costs") is not None


@pytest.mark.legal
def test_term_registry_resolve_miss_returns_none():
    from groundguard.loaders.legal import TermRegistry
    registry = TermRegistry()
    assert registry.resolve("nonexistent term") is None


@pytest.mark.legal
def test_term_registry_register_manual():
    from groundguard.loaders.legal import TermRegistry
    from groundguard.models.result import Source
    registry = TermRegistry()
    src = Source(source_id="s1", content="definition text")
    registry.register("My Term", src)
    assert "my term" in registry.known_terms()


@pytest.mark.legal
def test_decompose_clause_modal_shall_not():
    from groundguard.loaders.legal import decompose_clause
    unit = decompose_clause("The fee shall not exceed 30% of revenue.")
    assert unit.modal_operator == "shall not"


@pytest.mark.legal
def test_decompose_clause_modal_may():
    from groundguard.loaders.legal import decompose_clause
    unit = decompose_clause("The party may terminate this agreement.")
    assert unit.modal_operator == "may"


@pytest.mark.legal
def test_decompose_clause_subordinate_modifiers():
    from groundguard.loaders.legal import decompose_clause
    unit = decompose_clause("The fee as defined in Schedule 1 shall not exceed 30%.")
    assert any("Schedule 1" in m for m in unit.subordinate_modifiers)


@pytest.mark.legal
def test_decompose_clause_defined_terms():
    from groundguard.loaders.legal import decompose_clause
    unit = decompose_clause("The Permitted Costs shall be approved by the Board.")
    assert "Permitted Costs" in unit.defined_terms_referenced or \
           "Board" in unit.defined_terms_referenced
