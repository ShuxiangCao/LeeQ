"""
Basic tests to verify EPII module structure is properly set up
"""

import pytest
import os
import json


def test_epii_module_exists():
    """Test that the epii module can be imported"""
    import leeq.epii
    assert leeq.epii.__version__ == "1.0.0"


def test_epii_directory_structure():
    """Test that all required directories exist"""
    import leeq
    
    base_path = os.path.dirname(leeq.__file__)
    epii_path = os.path.join(base_path, "epii")
    
    # Check main directory exists
    assert os.path.exists(epii_path), "leeq/epii directory should exist"
    assert os.path.isdir(epii_path), "leeq/epii should be a directory"
    
    # Check proto subdirectory exists
    proto_path = os.path.join(epii_path, "proto")
    assert os.path.exists(proto_path), "leeq/epii/proto directory should exist"
    assert os.path.isdir(proto_path), "leeq/epii/proto should be a directory"
    
    # Check __init__.py files exist
    assert os.path.exists(os.path.join(epii_path, "__init__.py")), "leeq/epii/__init__.py should exist"
    assert os.path.exists(os.path.join(proto_path, "__init__.py")), "leeq/epii/proto/__init__.py should exist"


def test_test_directory_structure():
    """Test that test directories are properly set up"""
    test_path = os.path.dirname(os.path.abspath(__file__))
    
    # Check fixtures directory exists
    fixtures_path = os.path.join(test_path, "fixtures")
    assert os.path.exists(fixtures_path), "tests/epii/fixtures directory should exist"
    assert os.path.isdir(fixtures_path), "tests/epii/fixtures should be a directory"
    
    # Check test files exist
    expected_test_files = [
        "test_service.py",
        "test_experiments.py", 
        "test_parameters.py",
        "test_serialization.py",
        "test_daemon.py",
        "conftest.py"
    ]
    
    for test_file in expected_test_files:
        file_path = os.path.join(test_path, test_file)
        assert os.path.exists(file_path), f"{test_file} should exist in tests/epii/"


def test_fixture_config_valid():
    """Test that the fixture configuration file is valid JSON"""
    test_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(test_path, "fixtures", "simulation_2q.json")
    
    assert os.path.exists(config_path), "Fixture config file should exist"
    
    # Try to load and validate the JSON
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check required fields
    assert "setup_type" in config, "Config should have setup_type"
    assert "setup_name" in config, "Config should have setup_name"
    assert "parameters" in config, "Config should have parameters"
    assert config["setup_type"] == "simulation", "Setup type should be simulation"
    assert config["parameters"]["num_qubits"] == 2, "Should be 2-qubit simulation"


def test_pytest_fixtures_import():
    """Test that pytest fixtures can be imported"""
    from tests.epii.conftest import (
        mock_leeq_setup,
        minimal_config_file,
        simulation_2q_config,
        sample_experiment_request,
        sample_numpy_data,
        mock_grpc_context,
        generate_experiment_result
    )
    
    # Test that fixture functions exist
    assert callable(generate_experiment_result), "generate_experiment_result should be callable"
    
    # Test data generator
    rabi_data = generate_experiment_result("rabi", 50)
    assert "data" in rabi_data, "Generated data should have 'data' key"
    assert "fit_params" in rabi_data, "Generated data should have 'fit_params' key"
    assert len(rabi_data["data"]) == 50, "Should generate correct number of points"