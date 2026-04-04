from agentic_verifier._log import logger


def test_logger_name():
    assert logger.name == "agentic_verifier"


def test_logger_no_handlers():
    assert len(logger.handlers) == 0
