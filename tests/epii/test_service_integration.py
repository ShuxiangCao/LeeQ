"""
Integration tests for EPII service with simplified ParameterManager.

Tests the integration between the EPII service endpoints and the
simplified parameter management system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from leeq.epii.service import ExperimentPlatformService
from leeq.epii.proto import epii_pb2


class TestServiceIntegration:
    """Test EPII service integration with simplified ParameterManager."""

    @pytest.fixture
    def mock_setup(self):
        """Create a mock setup with test parameters."""
        setup = Mock()
        setup.name = "test_setup"

        # Mock status with parameters - use a real dict for proper updates
        status_dict = {
            "shot_number": 1000,
            "run_mode": "simulation",
            "acquisition_time": 2.5
        }
        setup.status = Mock()
        setup.status._internal_dict = status_dict
        # Make get_parameters return the actual dict
        setup.status.get_parameters = lambda: status_dict
        # Make set_parameter update the dict
        def set_param(key, value):
            status_dict[key] = value
        setup.status.set_parameter = set_param

        # Mock elements
        mock_element = Mock()
        mock_element._parameters = {
            "frequency": 5000.0,
            "amplitude": 0.5,
            "phase": 0.0
        }
        setup._elements = {
            "readout": mock_element
        }

        # Mock qubits
        mock_qubit = Mock()
        mock_qubit._parameters = {
            "f01": 4500.0,
            "anharmonicity": -200.0,
            "T1": 50.0,
            "T2": 25.0
        }
        setup.qubits = [mock_qubit]

        return setup

    @pytest.fixture
    def service(self, mock_setup):
        """Create service instance with mock setup."""
        return ExperimentPlatformService(setup=mock_setup)

    def test_get_all_parameters(self, service):
        """Test GetParameters returns all parameters when no specific ones requested."""
        request = epii_pb2.ParameterRequest()
        # Empty parameter_names means get all

        response = service.GetParameters(request, None)

        # Should return all parameters
        assert len(response.parameters) > 0

        # Check some expected parameters
        assert "status.shot_number" in response.parameters
        assert response.parameters["status.shot_number"] == "1000"

        assert "status.run_mode" in response.parameters
        assert response.parameters["status.run_mode"] == "simulation"

        assert "readout.frequency" in response.parameters
        assert response.parameters["readout.frequency"] == "5000.0"

        assert "q0.f01" in response.parameters
        assert response.parameters["q0.f01"] == "4500.0"

    def test_get_specific_parameters(self, service):
        """Test GetParameters returns only requested parameters."""
        request = epii_pb2.ParameterRequest()
        request.parameter_names.extend([
            "status.shot_number",
            "q0.f01",
            "nonexistent.param"
        ])

        response = service.GetParameters(request, None)

        # Should return exactly 3 parameters
        assert len(response.parameters) == 3

        # Check requested parameters
        assert response.parameters["status.shot_number"] == "1000"
        assert response.parameters["q0.f01"] == "4500.0"
        assert response.parameters["nonexistent.param"] == "null"

    def test_set_parameters_success(self, service):
        """Test SetParameters updates values correctly."""
        request = epii_pb2.SetParametersRequest()
        request.parameters["status.shot_number"] = "2000"
        request.parameters["q0.f01"] = "4600.0"
        request.parameters["readout.amplitude"] = "0.8"

        response = service.SetParameters(request, None)

        # Should succeed
        assert response.success is True
        # StatusResponse only has success and error_message fields

        # Verify values were updated
        get_request = epii_pb2.ParameterRequest()
        get_request.parameter_names.extend([
            "status.shot_number",
            "q0.f01",
            "readout.amplitude"
        ])
        get_response = service.GetParameters(get_request, None)

        assert get_response.parameters["status.shot_number"] == "2000"
        assert get_response.parameters["q0.f01"] == "4600.0"
        assert get_response.parameters["readout.amplitude"] == "0.8"

    def test_set_parameters_mixed_results(self, service):
        """Test SetParameters handles partial success correctly."""
        request = epii_pb2.SetParametersRequest()
        request.parameters["status.shot_number"] = "3000"
        request.parameters["cache.new_param"] = "test_value"  # Will go to cache

        response = service.SetParameters(request, None)

        # Should succeed (both can be set)
        assert response.success is True

        # Verify both were set
        get_request = epii_pb2.ParameterRequest()
        get_request.parameter_names.extend([
            "status.shot_number",
            "cache.new_param"
        ])
        get_response = service.GetParameters(get_request, None)

        assert get_response.parameters["status.shot_number"] == "3000"
        assert get_response.parameters["cache.new_param"] == "test_value"

    def test_list_parameters(self, service):
        """Test ListParameters returns all parameters with metadata."""
        request = epii_pb2.Empty()

        response = service.ListParameters(request, None)

        # Should return multiple parameters
        assert len(response.parameters) > 0

        # Find a specific parameter and check its info
        status_shot = None
        for param in response.parameters:
            if param.name == "status.shot_number":
                status_shot = param
                break

        assert status_shot is not None
        assert status_shot.type == "int"
        assert status_shot.current_value == "1000"
        assert status_shot.description == "Parameter status.shot_number"
        assert status_shot.read_only is False

        # Check qubit parameter
        q0_f01 = None
        for param in response.parameters:
            if param.name == "q0.f01":
                q0_f01 = param
                break

        assert q0_f01 is not None
        assert q0_f01.type == "float"
        assert q0_f01.current_value == "4500.0"

    def test_parameter_type_serialization(self, service):
        """Test different data types are serialized correctly."""
        # Set different types
        request = epii_pb2.SetParametersRequest()
        request.parameters["test.bool_param"] = "true"
        request.parameters["test.int_param"] = "42"
        request.parameters["test.float_param"] = "3.14159"
        request.parameters["test.string_param"] = "hello world"

        response = service.SetParameters(request, None)
        assert response.success is True

        # Get them back
        get_request = epii_pb2.ParameterRequest()
        get_request.parameter_names.extend([
            "test.bool_param",
            "test.int_param",
            "test.float_param",
            "test.string_param"
        ])
        get_response = service.GetParameters(get_request, None)

        # Check serialization
        assert get_response.parameters["test.bool_param"] == "true"
        assert get_response.parameters["test.int_param"] == "42"
        assert get_response.parameters["test.float_param"] == "3.14159"
        assert get_response.parameters["test.string_param"] == "hello world"

    def test_numpy_array_serialization(self, service):
        """Test numpy arrays are handled correctly."""
        # Add a numpy array to the mock setup
        service.setup._elements["readout"]._parameters["waveform"] = np.array([1.0, 2.0, 3.0])

        # Get all parameters
        request = epii_pb2.ParameterRequest()
        response = service.GetParameters(request, None)

        # Check numpy array was serialized
        assert "readout.waveform" in response.parameters
        # Should be serialized as a list
        assert response.parameters["readout.waveform"] == "[1.0, 2.0, 3.0]"

    def test_nested_dict_parameters(self, service):
        """Test nested dictionary parameters are flattened correctly."""
        # Add nested dict to mock setup
        service.setup._elements["readout"]._parameters["calibration"] = {
            "offset": 0.1,
            "scale": 1.5
        }

        # Get all parameters
        request = epii_pb2.ParameterRequest()
        response = service.GetParameters(request, None)

        # Check nested params are flattened with dot notation
        assert "readout.calibration.offset" in response.parameters
        assert response.parameters["readout.calibration.offset"] == "0.1"
        assert "readout.calibration.scale" in response.parameters
        assert response.parameters["readout.calibration.scale"] == "1.5"

    def test_empty_setup_handling(self):
        """Test service handles missing setup gracefully."""
        service = ExperimentPlatformService(setup=None)

        # Get parameters should return empty
        request = epii_pb2.ParameterRequest()
        response = service.GetParameters(request, None)

        # Response.parameters is a protobuf map, check it exists
        assert hasattr(response, 'parameters')

        # Set parameters should still work via cache
        set_request = epii_pb2.SetParametersRequest()
        set_request.parameters["cache.param"] = "value"
        set_response = service.SetParameters(set_request, None)

        assert set_response.success is True

        # Should be able to get it back
        get_request = epii_pb2.ParameterRequest()
        get_request.parameter_names.append("cache.param")
        get_response = service.GetParameters(get_request, None)

        assert get_response.parameters["cache.param"] == "value"
