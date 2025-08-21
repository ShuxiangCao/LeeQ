"""
Pytest configuration for notebook testing.

This module provides fixtures and configuration for testing LeeQ notebooks,
including Chronicle integration and output verification.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import os


@pytest.fixture(scope="session")
def project_root():
    """Project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def notebooks_dir(project_root):
    """Notebooks directory."""
    return project_root / "notebooks"


@pytest.fixture(scope="function")
def temp_log_dir():
    """Temporary log directory for Chronicle testing."""
    temp_dir = Path(tempfile.mkdtemp(prefix='leeq_test_logs_'))
    yield temp_dir
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except:
        pass


@pytest.fixture(scope="function")
def isolated_environment(temp_log_dir):
    """Isolated environment for notebook execution."""
    # Save original environment
    original_env = os.environ.copy()

    # Set test environment
    os.environ['LEEQ_LOG_DIR'] = str(temp_log_dir)
    os.environ['LEEQ_TEST_MODE'] = 'true'

    yield {
        'log_dir': temp_log_dir,
        'original_env': original_env
    }

    # Restore environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="session")
def notebook_test_config():
    """Configuration for notebook testing."""
    return {
        'execution_timeout': 300,  # 5 minutes
        'required_chronicle_imports': [
            'leeq.chronicle',
            'Chronicle',
            'log_and_record'
        ],
        'expected_plot_types': [
            'image/png',
            'application/vnd.plotly.v1+json'
        ],
        'tutorial_requirements': {
            'min_cells': 5,
            'requires_plots': True,
            'requires_chronicle': True
        },
        'example_requirements': {
            'min_cells': 3,
            'requires_plots': False,
            'requires_chronicle': True
        },
        'workflow_requirements': {
            'min_cells': 8,
            'requires_plots': True,
            'requires_chronicle': True
        }
    }


def pytest_collection_modifyitems(config, items):
    """Automatically mark notebook tests."""
    for item in items:
        # Mark notebook tests
        if "notebook" in str(item.fspath):
            item.add_marker(pytest.mark.notebook)

        # Mark Chronicle tests
        if "chronicle" in item.name.lower():
            item.add_marker(pytest.mark.chronicle)

        # Mark slow tests (notebook execution)
        if any(marker in item.name.lower() for marker in ['execute', 'integration', 'full']):
            item.add_marker(pytest.mark.slow)


def pytest_configure(config):
    """Configure pytest for notebook testing."""
    # Register custom markers
    config.addinivalue_line("markers", "notebook: notebook-related tests")
    config.addinivalue_line("markers", "chronicle: Chronicle integration tests")
    config.addinivalue_line("markers", "slow: slow-running tests")
    config.addinivalue_line("markers", "integration: integration tests")
