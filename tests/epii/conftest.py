"""
Pytest fixtures for EPII testing

Provides common test fixtures for EPII service testing including:
- Mock LeeQ setups
- gRPC test channels
- Sample experiment configurations
- Test data generators
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import tempfile
import json
import os
from typing import Dict, Any, Generator


@pytest.fixture
def mock_leeq_setup():
    """
    Creates a mock LeeQ setup for testing without requiring actual hardware
    or full simulation initialization.
    """
    setup = Mock()
    setup.name = "test_setup"

    # Mock virtual qubits
    q0 = Mock()
    q0.name = "q0"
    q0.get_c1 = Mock(return_value={"qubit": Mock()})

    q1 = Mock()
    q1.name = "q1"
    q1.get_c1 = Mock(return_value={"qubit": Mock()})

    setup.get_virtual_qubit = Mock(side_effect=lambda name: q0 if name == "q0" else q1)
    setup.qubits = {"q0": q0, "q1": q1}

    # Mock status parameters
    setup.get_status = Mock(return_value={
        "q0.f01": 5.0e9,
        "q0.pi_amp": 0.5,
        "q0.pi_len": 40,
        "q1.f01": 5.1e9,
        "q1.pi_amp": 0.45,
        "q1.pi_len": 38,
    })

    setup.set_status = Mock()

    return setup


@pytest.fixture
def minimal_config_file(tmp_path) -> Generator[str, None, None]:
    """
    Creates a minimal JSON configuration file for testing daemon startup.
    """
    config = {
        "setup_type": "simulation",
        "setup_name": "test_setup",
        "parameters": {
            "num_qubits": 2,
            "simulation_backend": "numpy"
        }
    }

    config_path = tmp_path / "minimal.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    yield str(config_path)


@pytest.fixture
def simulation_2q_config(tmp_path) -> Generator[Dict[str, Any], None, None]:
    """
    Creates a 2-qubit simulation configuration dictionary.
    """
    config = {
        "setup_type": "simulation",
        "setup_name": "simulation_2q",
        "parameters": {
            "num_qubits": 2,
            "simulation_backend": "numpy",
            "qubits": {
                "q0": {
                    "f01": 5.0e9,
                    "anharmonicity": -0.33e9,
                    "characterizations.SimpleT1": 20e-6,
                    "t2": 15e-6,
                    "pi_amp": 0.5,
                    "pi_len": 40
                },
                "q1": {
                    "f01": 5.1e9,
                    "anharmonicity": -0.32e9,
                    "characterizations.SimpleT1": 25e-6,
                    "t2": 18e-6,
                    "pi_amp": 0.45,
                    "pi_len": 38
                }
            },
            "couplings": {
                "q0-q1": {
                    "strength": 5e6,
                    "type": "capacitive"
                }
            }
        }
    }

    yield config


@pytest.fixture
def sample_experiment_request():
    """
    Creates a sample experiment request dictionary for testing.
    """
    return {
        "experiment_name": "calibrations.NormalisedRabi",
        "parameters": {
            "qubit": "q0",
            "amp_range": [0.0, 1.0],
            "num_points": 51,
            "num_shots": 1024
        }
    }


@pytest.fixture
def sample_numpy_data():
    """
    Creates sample numpy arrays for testing serialization.
    """
    return {
        "1d_array": np.linspace(0, 1, 100),
        "2d_array": np.random.randn(50, 3),
        "complex_array": np.array([1+2j, 3+4j, 5+6j]),
        "int_array": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "large_array": np.random.randn(1000, 100)
    }


@pytest.fixture
def mock_grpc_context():
    """
    Creates a mock gRPC context for testing service methods.
    """
    context = Mock()
    context.set_code = Mock()
    context.set_details = Mock()
    context.abort = Mock(side_effect=Exception("gRPC abort"))
    return context


@pytest.fixture
def temp_log_dir(tmp_path):
    """
    Creates a temporary directory for test log files.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return str(log_dir)


@pytest.fixture
def temp_pid_file(tmp_path):
    """
    Creates a temporary PID file path for daemon testing.
    """
    return str(tmp_path / "epii.pid")


# Test data generators

def generate_experiment_result(experiment_type: str, num_points: int = 50):
    """
    Generates mock experiment result data for different experiment types.

    Args:
        experiment_type: Type of experiment (rabi, t1, ramsey, etc.)
        num_points: Number of data points to generate

    Returns:
        Dictionary with 'data' and 'fit_params' keys
    """
    if experiment_type == "calibrations.NormalisedRabi":
        x = np.linspace(0, 1, num_points)
        y = 0.5 * (1 + np.cos(2 * np.pi * x + 0.1))
        fit_params = {
            "amplitude": 0.5,
            "frequency": 1.0,
            "phase": 0.1,
            "offset": 0.5
        }
    elif experiment_type == "characterizations.SimpleT1":
        x = np.linspace(0, 100e-6, num_points)
        y = np.exp(-x / 20e-6) + 0.1 * np.random.randn(num_points)
        fit_params = {
            "characterizations.SimpleT1": 20e-6,
            "amplitude": 1.0,
            "offset": 0.0
        }
    elif experiment_type == "calibrations.SimpleRamseyMultilevel":
        x = np.linspace(0, 50e-6, num_points)
        y = np.exp(-x / 15e-6) * np.cos(2 * np.pi * 1e6 * x)
        fit_params = {
            "t2": 15e-6,
            "frequency": 1e6,
            "amplitude": 1.0,
            "phase": 0.0
        }
    else:
        x = np.linspace(0, 1, num_points)
        y = np.random.randn(num_points)
        fit_params = {}

    return {
        "x": x,
        "data": y,
        "fit_params": fit_params
    }
