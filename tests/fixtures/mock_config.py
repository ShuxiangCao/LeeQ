"""Configuration fixtures for daemon tests."""
import pytest


@pytest.fixture
def minimal_daemon_config():
    """Provide minimal valid daemon configuration."""
    return {
        "port": 50051,
        "max_workers": 10,
        "skip_health_checks": True
    }


@pytest.fixture  
def full_daemon_config():
    """Provide full daemon configuration."""
    return {
        "port": 50051,
        "max_workers": 10,
        "setup_type": "simulation",
        "debug_grpc": False,
        "skip_health_checks": False
    }