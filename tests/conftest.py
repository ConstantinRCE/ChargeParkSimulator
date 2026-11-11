import pytest

from chargepark.config import load_config


@pytest.fixture
def cfg():
    """The project config.yaml, validated."""
    return load_config()
