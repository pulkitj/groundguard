"""Assertion-style circuit breakers for grounding verification."""
from groundguard.models.result import GroundingResult, Source
from groundguard.core.verifier import verify_answer, verify_analysis


class GroundingError(Exception):
    pass


def assert_faithful(output: str, sources: list[Source], **kwargs) -> None:
    result = verify_answer(output, sources, **kwargs)
    if not result.is_grounded:
        raise GroundingError(
            f"Output not grounded: score={result.score:.2f}, status={result.status}"
        )


def assert_grounded(analysis: str, sources: list[Source], **kwargs) -> None:
    result = verify_analysis(analysis, sources, **kwargs)
    if not result.is_grounded:
        raise GroundingError(
            f"Analysis not grounded: score={result.score:.2f}"
        )


def verify_or_retry(generator, sources: list[Source], max_retries: int = 3, **kwargs) -> str:
    for attempt in range(max_retries):
        output = generator()
        result = verify_answer(output, sources, **kwargs)
        if result.is_grounded:
            return output
    raise GroundingError(f"Output not grounded after {max_retries} attempts")
