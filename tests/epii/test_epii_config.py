"""
Tests for EPII configuration management.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from leeq.epii.config import EPIIConfig
from leeq.setups.setup_base import ExperimentalSetup


class TestEPIIConfig:
    """Test suite for EPIIConfig class."""

    def test_default_configuration(self):
        """Test that default configuration is loaded correctly."""
        config = EPIIConfig()

        assert config.get("setup_type") == "simulation"
        assert config.get("setup_name") == "default_setup"
        assert config.get("port") == 50051
        assert config.get("log_level") == "INFO"
        assert config.get("max_workers") == 10
        assert config.get("num_qubits") == 2
        assert config.get("simulation_backend") == "numpy"

    def test_load_from_file(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_data = {
            "setup_type": "simulation",
            "setup_name": "test_setup",
            "port": 50052,
            "num_qubits": 3,
            "parameters": {
                "simulation_backend": "high_level",
                "qubits": {
                    "q0": {"f01": 5.0e9, "t1": 30e-6}
                }
            }
        }

        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        config = EPIIConfig(str(config_file))

        assert config.get("setup_name") == "test_setup"
        assert config.get("port") == 50052
        assert config.get("num_qubits") == 3
        assert config.get("parameters")["simulation_backend"] == "high_level"

    def test_environment_override(self, monkeypatch):
        """Test that environment variables override configuration."""
        monkeypatch.setenv("LEEQ_EPII_PORT", "50055")
        monkeypatch.setenv("LEEQ_EPII_LOG_LEVEL", "DEBUG")

        config = EPIIConfig()

        assert config.get("port") == 50055
        assert config.get("log_level") == "DEBUG"

    def test_validation_success(self):
        """Test successful configuration validation."""
        config = EPIIConfig()
        assert config.validate() is True

    def test_validation_missing_field(self):
        """Test validation fails with missing required field."""
        config = EPIIConfig()
        del config.config["port"]

        with pytest.raises(ValueError, match="Missing required configuration field: port"):
            config.validate()

    def test_validation_invalid_port(self):
        """Test validation fails with invalid port number."""
        config = EPIIConfig()
        config.config["port"] = 40000

        with pytest.raises(ValueError, match="Port must be between 50051 and 50099"):
            config.validate()

    def test_validation_invalid_setup_type(self):
        """Test validation fails with invalid setup type."""
        config = EPIIConfig()
        config.config["setup_type"] = "invalid"

        with pytest.raises(ValueError, match="Invalid setup type: invalid"):
            config.validate()

    def test_validation_invalid_num_qubits(self):
        """Test validation fails with invalid num_qubits."""
        config = EPIIConfig()
        config.config["num_qubits"] = 0

        with pytest.raises(ValueError, match="num_qubits must be a positive integer"):
            config.validate()

    def test_validation_invalid_backend(self):
        """Test validation fails with invalid simulation backend."""
        config = EPIIConfig()
        config.config["simulation_backend"] = "invalid_backend"

        with pytest.raises(ValueError, match="Invalid simulation backend: invalid_backend"):
            config.validate()

    def test_backward_compatibility(self):
        """Test backward compatibility with old field names."""
        config = EPIIConfig()
        config.config = {
            "type": "simulation",  # Old field name
            "name": "old_setup",    # Old field name
            "port": 50051
        }

        assert config.validate() is True

    def test_save_to_file(self, tmp_path):
        """Test saving configuration to file."""
        config = EPIIConfig()
        config.config["setup_name"] = "saved_setup"

        save_path = tmp_path / "saved_config.json"
        config.save_to_file(str(save_path))

        assert save_path.exists()

        with open(save_path, 'r') as f:
            loaded = json.load(f)

        assert loaded["setup_name"] == "saved_setup"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "setup_type": "simulation",
            "setup_name": "dict_setup",
            "port": 50060
        }

        config = EPIIConfig.from_dict(config_dict)

        assert config.get("setup_name") == "dict_setup"
        assert config.get("port") == 50060

    def test_create_numpy_simulation_setup(self):
        """Test creating a numpy simulation setup."""
        config = EPIIConfig()
        config.config["simulation_backend"] = "numpy"

        setup = config.create_setup()

        assert isinstance(setup, ExperimentalSetup)
        assert hasattr(setup, "_compiler")
        assert hasattr(setup, "_engine")

    def test_create_high_level_simulation_setup(self):
        """Test creating a high-level simulation setup."""
        config = EPIIConfig()
        config.config["simulation_backend"] = "high_level"
        config.config["num_qubits"] = 2
        config.config["parameters"] = {
            "qubits": {
                "q0": {
                    "f01": 5.0e9,
                    "anharmonicity": -0.33e9,
                    "t1": 20e-6,
                    "t2": 15e-6
                },
                "q1": {
                    "f01": 5.1e9,
                    "anharmonicity": -0.32e9,
                    "t1": 25e-6,
                    "t2": 18e-6
                }
            },
            "couplings": {
                "q0-q1": {
                    "strength": 5e6,
                    "type": "capacitive"
                }
            }
        }

        setup = config.create_setup()

        assert isinstance(setup, ExperimentalSetup)
        assert hasattr(setup, "_virtual_qubits")
        assert hasattr(setup, "_coupling_strength_map")

    def test_create_hardware_setup_not_implemented(self):
        """Test that hardware setup creation raises NotImplementedError."""
        config = EPIIConfig()
        config.config["setup_type"] = "hardware"
        config.config["parameters"] = {
            "hardware_type": "qubic"
        }

        with pytest.raises(NotImplementedError, match="Hardware setup creation requires"):
            config.create_setup()

    def test_load_fixture_files(self):
        """Test loading actual fixture files."""
        fixtures_dir = Path(__file__).parent / "fixtures"

        # Test minimal.json
        minimal_config = EPIIConfig(str(fixtures_dir / "minimal.json"))
        assert minimal_config.get("setup_type") == "simulation"
        assert minimal_config.get("num_qubits") == 2

        # Test simulation_2q.json
        sim_2q_config = EPIIConfig(str(fixtures_dir / "simulation_2q.json"))
        assert sim_2q_config.get("setup_type") == "simulation"
        assert sim_2q_config.get("port") == 50051
        assert sim_2q_config.get("parameters")["num_qubits"] == 2
