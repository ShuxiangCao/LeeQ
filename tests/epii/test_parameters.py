"""
Unit tests for EPII parameter management - Simplified implementation
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from leeq.epii.parameters import ParameterManager


class TestParameterManager:
    """Test suite for simplified ParameterManager class"""

    @pytest.fixture
    def mock_setup(self):
        """Create a mock LeeQ setup with status parameters"""
        setup = Mock()

        # Mock status parameters
        status = Mock()
        status._internal_dict = {
            "shot_number": 2000,
            "shot_period": 500.0,
            "acquisition_type": "IQ",
            "debug_plotter": False,
            "measurement_basis": "<z>"
        }
        status.get_parameters = Mock(side_effect=lambda key=None:
            status._internal_dict[key] if key else status._internal_dict.copy())
        status.set_parameter = Mock(side_effect=lambda key, value:
            status._internal_dict.update({key: value}))

        setup.status = status
        setup._name = "test_setup"
        setup._active = True

        # Mock elements (qubits)
        q0 = Mock()
        q0._parameters = {
            "f01": 5.0e9,
            "anharmonicity": -0.33e9,
            "characterizations.SimpleT1": 20e-6,
            "t2": 15e-6,
            "pi_amp": 0.5,
            "pi_len": 40
        }

        q1 = Mock()
        q1._parameters = {
            "f01": 5.1e9,
            "anharmonicity": -0.32e9,
            "characterizations.SimpleT1": 25e-6,
            "t2": 18e-6,
            "pi_amp": 0.45,
            "pi_len": 38
        }

        setup._elements = {"q0": q0, "q1": q1}

        # Also add qubits list for testing
        setup.qubits = [q0, q1]

        return setup

    def test_init_without_setup(self):
        """Test initialization without a setup"""
        pm = ParameterManager()
        assert pm.setup is None
        assert pm._cache == {}

    def test_init_with_setup(self, mock_setup):
        """Test initialization with a setup"""
        pm = ParameterManager(mock_setup)
        assert pm.setup == mock_setup
        assert pm._cache == {}

    def test_get_status_parameter(self, mock_setup):
        """Test retrieval of setup status parameters"""
        pm = ParameterManager(mock_setup)

        # Get existing parameter
        value = pm.get_parameter("status.shot_number")
        assert value == 2000

        # Get another parameter
        value = pm.get_parameter("status.shot_period")
        assert value == 500.0

        # Parameter names are case-sensitive now
        value = pm.get_parameter("status.shot_number")
        assert value == 2000

    def test_get_element_parameter(self, mock_setup):
        """Test retrieval of element parameters"""
        pm = ParameterManager(mock_setup)

        # Get q0 frequency
        value = pm.get_parameter("q0.f01")
        assert value == 5.0e9

        # Get q1 pi amplitude
        value = pm.get_parameter("q1.pi_amp")
        assert value == 0.45

    def test_get_nonexistent_parameter(self, mock_setup):
        """Test retrieval of non-existent parameter"""
        pm = ParameterManager(mock_setup)

        value = pm.get_parameter("status.nonexistent")
        assert value is None

        value = pm.get_parameter("q0.nonexistent")
        assert value is None

    def test_set_status_parameter(self, mock_setup):
        """Test setting setup status parameters"""
        pm = ParameterManager(mock_setup)

        # Set shot number
        success = pm.set_parameter("status.shot_number", 5000)
        assert success is True
        assert pm.get_parameter("status.shot_number") == 5000

        # Set with string value (should convert)
        success = pm.set_parameter("status.shot_period", "750.5")
        assert success is True
        assert pm.get_parameter("status.shot_period") == 750.5

    def test_set_element_parameter(self, mock_setup):
        """Test setting element parameters"""
        pm = ParameterManager(mock_setup)

        # Set q0 pi amplitude
        success = pm.set_parameter("q0.pi_amp", 0.6)
        assert success is True
        assert mock_setup._elements["q0"]._parameters["pi_amp"] == 0.6

    def test_set_unknown_parameter_to_cache(self, mock_setup):
        """Test that unknown parameters go to cache"""
        pm = ParameterManager(mock_setup)

        # Set a parameter that doesn't exist in setup
        success = pm.set_parameter("custom.param", "value")
        assert success is True
        assert pm._cache["custom.param"] == "value"
        assert pm.get_parameter("custom.param") == "value"

    def test_get_all_parameters(self, mock_setup):
        """Test getting all parameters as flat dictionary"""
        pm = ParameterManager(mock_setup)

        all_params = pm.get_all_parameters()

        # Check status parameters are included
        assert "status.shot_number" in all_params
        assert all_params["status.shot_number"] == 2000

        # Check element parameters are included
        assert "q0.f01" in all_params
        assert all_params["q0.f01"] == 5.0e9
        assert "q1.pi_amp" in all_params
        assert all_params["q1.pi_amp"] == 0.45

        # Check qubit parameters from qubits list
        assert "q0.f01" in all_params
        assert "q1.f01" in all_params

    def test_parse_value(self, mock_setup):
        """Test value parsing from strings"""
        pm = ParameterManager(mock_setup)

        # Test boolean parsing
        assert pm._parse_value("true") is True
        assert pm._parse_value("false") is False
        assert pm._parse_value("True") is True
        assert pm._parse_value("FALSE") is False

        # Test null parsing
        assert pm._parse_value("null") is None

        # Test integer parsing
        assert pm._parse_value("123") == 123
        assert pm._parse_value("-456") == -456

        # Test float parsing
        assert pm._parse_value("123.45") == 123.45
        assert pm._parse_value("1.23e5") == 1.23e5

        # Test string passthrough
        assert pm._parse_value("hello") == "hello"

        # Test non-string passthrough
        assert pm._parse_value(123) == 123
        assert pm._parse_value(True) is True
        assert pm._parse_value([1, 2, 3]) == [1, 2, 3]

    def test_serialize_value(self, mock_setup):
        """Test value serialization"""
        pm = ParameterManager(mock_setup)

        assert pm.serialize_value(None) == "null"
        assert pm.serialize_value(True) == "true"
        assert pm.serialize_value(False) == "false"
        assert pm.serialize_value(123) == "123"
        assert pm.serialize_value(45.6) == "45.6"
        assert pm.serialize_value("test") == "test"

        # Test numpy array serialization
        arr = np.array([1, 2, 3])
        assert pm.serialize_value(arr) == [1, 2, 3]

        # Test list and dict passthrough
        assert pm.serialize_value([1, 2, 3]) == [1, 2, 3]
        assert pm.serialize_value({"a": 1}) == {"a": 1}

    def test_flatten_dict(self, mock_setup):
        """Test dictionary flattening with dot notation"""
        pm = ParameterManager(mock_setup)

        nested = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }

        flat = pm._flatten_dict(nested)
        assert flat == {
            "a": 1,
            "b.c": 2,
            "b.d.e": 3
        }

    def test_set_nested_parameter(self, mock_setup):
        """Test setting nested parameters"""
        pm = ParameterManager(mock_setup)

        # Mock a deeper nested structure
        mock_setup._elements["q0"]._parameters["nested"] = {}

        # Set a nested parameter
        success = pm.set_parameter("q0.nested.deep.value", 42)
        assert success is True
        assert mock_setup._elements["q0"]._parameters["nested"]["deep"]["value"] == 42

    def test_cache_fallback(self, mock_setup):
        """Test that cache is used as fallback"""
        pm = ParameterManager(mock_setup)

        # Add something to cache
        pm._cache["cached.value"] = 100

        # Should be retrievable
        assert pm.get_parameter("cached.value") == 100

        # Should be included in all parameters
        all_params = pm.get_all_parameters()
        assert "cached.value" in all_params
        assert all_params["cached.value"] == 100

    def test_qubit_list_parameters(self, mock_setup):
        """Test that parameters from qubits list are retrieved"""
        pm = ParameterManager(mock_setup)

        all_params = pm.get_all_parameters()

        # Check that qubit parameters are indexed correctly
        assert "q0.f01" in all_params
        assert all_params["q0.f01"] == 5.0e9
        assert "q1.f01" in all_params
        assert all_params["q1.f01"] == 5.1e9

    def test_set_qubit_parameter(self, mock_setup):
        """Test setting parameters on qubits from the qubits list"""
        pm = ParameterManager(mock_setup)

        # Set q0 frequency
        success = pm.set_parameter("q0.f01", 4.9e9)
        assert success is True
        assert mock_setup.qubits[0]._parameters["f01"] == 4.9e9

        # Verify it's retrievable
        assert pm.get_parameter("q0.f01") == 4.9e9
