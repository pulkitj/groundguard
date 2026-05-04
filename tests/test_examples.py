import importlib
import inspect

import pytest


def _mock_grounding_result():
    from groundguard.models.result import GroundingResult

    return GroundingResult(
        is_grounded=True,
        score=0.9,
        status="GROUNDED",
        evaluation_method="sentence_entailment",
    )


@pytest.mark.parametrize(
    "module_path,fn_name",
    [
        ("examples.langchain_retrieval_qa", "build_verified_retrieval_chain"),
        ("examples.cohere_rag", "build_verified_cohere_rag"),
        ("examples.bedrock_rag", "build_verified_bedrock_rag"),
        ("examples.llamaindex_citation", "build_verified_citation_engine"),
        ("examples.openai_assistants", "build_verified_assistant_thread"),
        ("examples.full_output_verification", "run_full_output_example"),
    ],
)
def test_example_module_importable_and_fn_exists(module_path, fn_name):
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        pytest.skip(f"optional dep missing: {e}")
    assert hasattr(mod, fn_name), f"{fn_name} not found in {module_path}"
    fn = getattr(mod, fn_name)
    assert callable(fn)


def test_full_output_verification_example_runs(mocker):
    mocker.patch(
        "groundguard.verify_analysis",
        return_value=_mock_grounding_result(),
    )
    from examples.full_output_verification import run_full_output_example

    result = run_full_output_example()
    assert result is not None
