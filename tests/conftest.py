import pytest

from src.common.logging import reset_run_logging_configuration


@pytest.fixture(autouse=True)
def _reset_run_logging_between_tests() -> None:
    reset_run_logging_configuration()
    yield
    reset_run_logging_configuration()
