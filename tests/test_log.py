from groundguard._log import logger


def test_logger_name():
    assert logger.name == "groundguard"


def test_logger_no_handlers():
    assert len(logger.handlers) == 0
