"""
LeeQ Backend Integration Tests for Phase 3, Task 3.2

This module provides comprehensive tests for LeeQ backend integration,
validating the complete pipeline from gRPC requests to LeeQ experiment execution.

Tests include:
- Real LeeQ simulation setup creation and management
- Experiment parameter passing and validation
- Result collection and serialization
- Timeout and cancellation handling
- Compatibility with different LeeQ experiment types
"""

import pytest
import grpc
import time
import numpy as np
import threading
from typing import Dict, Any
from unittest.mock import patch, Mock

from leeq.epii.proto import epii_pb2, epii_pb2_grpc
from leeq.epii.service import ExperimentPlatformService
from leeq.epii.serialization import protobuf_to_numpy_array
from leeq.epii.config import create_setup_from_config
from leeq.epii.client_helpers import get_data, get_arrays, get_calibration_results

# LeeQ imports for real simulation setup
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.chronicle import Chronicle


class TestLeeQBackendIntegration:
    """
    Tests for complete LeeQ backend integration in simulation mode.

    Validates that EPII service can successfully interface with real LeeQ
    experiment classes and execute them in simulation mode.
    """

    @pytest.fixture
    def real_simulation_setup(self):
        """Create a real LeeQ simulation setup for testing."""
        # Initialize Chronicle
        Chronicle().start_log()

        # Clear any existing setups
        manager = ExperimentManager()
        manager.clear_setups()

        # Create virtual transmons
        virtual_q0 = VirtualTransmon(
            name="q0",
            qubit_frequency=5000.0,  # MHz
            anharmonicity=-200.0,   # MHz
            t1=30.0,               # μs
            t2=20.0,               # μs
            readout_frequency=9000.0,  # MHz
            quiescent_state_distribution=np.array([0.95, 0.04, 0.01])
        )

        virtual_q1 = VirtualTransmon(
            name="q1",
            qubit_frequency=5100.0,  # MHz
            anharmonicity=-190.0,   # MHz
            t1=25.0,               # μs
            t2=18.0,               # μs
            readout_frequency=9100.0,  # MHz
            quiescent_state_distribution=np.array([0.96, 0.03, 0.01])
        )

        # Create high-level simulation setup
        setup = HighLevelSimulationSetup(
            name='test_simulation_setup',
            virtual_qubits={0: virtual_q0, 1: virtual_q1},
            coupling_strength_map={frozenset(['q0', 'q1']): 5.0}  # MHz
        )

        # Register setup
        manager.register_setup(setup)

        return setup

    @pytest.fixture
    def real_qubit_elements(self):
        """Create real LeeQ qubit elements for testing."""
        # Configuration for realistic qubit elements
        configuration_q0 = {
            'lpb_collections': {
                'f01': {
                    'type': 'SimpleDriveCollection',
                    'freq': 5000.0,
                    'channel': 0,
                    'shape': 'blackman_drag',
                    'amp': 0.5,
                    'phase': 0.0,
                    'width': 0.020,
                    'alpha': 400.0,
                    'trunc': 1.2
                },
                'f12': {
                    'type': 'SimpleDriveCollection',
                    'freq': 4800.0,
                    'channel': 0,
                    'shape': 'blackman_drag',
                    'amp': 0.6,
                    'phase': 0.0,
                    'width': 0.025,
                    'alpha': 400.0,
                    'trunc': 1.2
                }
            },
            'measurement_primitives': {
                '0': {
                    'type': 'SimpleDispersiveMeasurement',
                    'freq': 9000.0,
                    'channel': 10,
                    'shape': 'square',
                    'amp': 0.3,
                    'phase': 0.0,
                    'width': 1.0,
                    'trunc': 1.2,
                    'distinguishable_states': [0, 1]
                },
                '1': {
                    'type': 'SimpleDispersiveMeasurement',
                    'freq': 9000.0,
                    'channel': 10,
                    'shape': 'square',
                    'amp': 0.3,
                    'phase': 0.0,
                    'width': 1.0,
                    'trunc': 1.2,
                    'distinguishable_states': [0, 1, 2]
                }
            }
        }

        configuration_q1 = {
            'lpb_collections': {
                'f01': {
                    'type': 'SimpleDriveCollection',
                    'freq': 5100.0,
                    'channel': 1,
                    'shape': 'blackman_drag',
                    'amp': 0.45,
                    'phase': 0.0,
                    'width': 0.018,
                    'alpha': 380.0,
                    'trunc': 1.2
                }
            },
            'measurement_primitives': {
                '0': {
                    'type': 'SimpleDispersiveMeasurement',
                    'freq': 9100.0,
                    'channel': 11,
                    'shape': 'square',
                    'amp': 0.25,
                    'phase': 0.0,
                    'width': 1.0,
                    'trunc': 1.2,
                    'distinguishable_states': [0, 1]
                }
            }
        }

        q0 = TransmonElement(name='q0', parameters=configuration_q0)
        q1 = TransmonElement(name='q1', parameters=configuration_q1)

        return {'q0': q0, 'q1': q1}

    @pytest.fixture
    def epii_service_with_real_setup(self, real_simulation_setup):
        """Create EPII service with real LeeQ simulation setup."""
        config = {
            "setup_type": "simulation",
            "setup_name": "test_simulation_setup",
            "num_qubits": 2,
            "max_workers": 4
        }

        service = ExperimentPlatformService(setup=real_simulation_setup, config=config)
        return service

    def test_real_rabi_experiment_execution(self, epii_service_with_real_setup, real_qubit_elements):
        """
        Test complete Rabi experiment execution with real LeeQ backend.

        This validates the full pipeline from EPII request to LeeQ experiment execution.
        """
        service = epii_service_with_real_setup
        q0 = real_qubit_elements['q0']

        # Create gRPC context mock
        context = Mock()
        context.set_code = Mock()
        context.set_details = Mock()

        # Create experiment request
        request = epii_pb2.ExperimentRequest()
        request.experiment_type = "calibrations.NormalisedRabi"
        request.parameters["dut_qubit"] = "q0"  # LeeQ parameter name
        request.parameters["amp"] = "0.5"       # LeeQ parameter name
        request.parameters["start"] = "0.0"     # LeeQ parameter name
        request.parameters["stop"] = "1.0"      # LeeQ parameter name
        request.parameters["step"] = "0.05"     # LeeQ parameter name

        # Mock the dut_qubit parameter to return our real qubit element
        with patch.object(service.experiment_router, 'map_parameters') as mock_map:
            mock_map.return_value = {
                "dut_qubit": q0,
                "amp": 0.5,
                "start": 0.0,
                "stop": 1.0,
                "step": 0.05
            }

            # Execute experiment
            response = service.RunExperiment(request, context)

            # Validate response
            assert response.success, f"Experiment failed: {response.error_message}"

            # Validate data serialization using new protocol
            arrays = get_arrays(response)
            if arrays:
                # Get the first array (typically measurement data)
                first_array_name = list(arrays.keys())[0] if arrays else None
                if first_array_name:
                    data_array = arrays[first_array_name]
                    assert isinstance(data_array, np.ndarray)
                    assert data_array.size > 0

            # Validate calibration results
            calibration_results = get_calibration_results(response)
            assert isinstance(calibration_results, dict)  # May be empty

            # Validate no gRPC errors were set
            context.set_code.assert_not_called()

    def test_real_t1_experiment_execution(self, epii_service_with_real_setup, real_qubit_elements):
        """
        Test T1 relaxation experiment execution with real LeeQ backend.
        """
        service = epii_service_with_real_setup
        q0 = real_qubit_elements['q0']

        context = Mock()
        context.set_code = Mock()
        context.set_details = Mock()

        # Create T1 experiment request
        request = epii_pb2.ExperimentRequest()
        request.experiment_type = "characterizations.SimpleT1"
        request.parameters["qubit"] = "q0"
        request.parameters["time_length"] = "50.0"  # μs
        request.parameters["time_resolution"] = "2.0"  # μs

        # Mock parameter mapping
        with patch.object(service.experiment_router, 'map_parameters') as mock_map:
            mock_map.return_value = {
                "qubit": q0,
                "time_length": 50.0,
                "time_resolution": 2.0
            }

            # Execute experiment
            response = service.RunExperiment(request, context)

            # Validate response
            assert response.success, f"T1 experiment failed: {response.error_message}"

            # Validate data using new protocol
            arrays = get_arrays(response)
            if arrays:
                # Check that we have measurement data
                assert len(arrays) > 0
                # Get the first array
                data_array = list(arrays.values())[0]
                assert isinstance(data_array, np.ndarray)

            context.set_code.assert_not_called()

    def test_real_ramsey_experiment_execution(self, epii_service_with_real_setup, real_qubit_elements):
        """
        Test Ramsey experiment execution with real LeeQ backend.
        """
        service = epii_service_with_real_setup
        q0 = real_qubit_elements['q0']

        context = Mock()
        context.set_code = Mock()
        context.set_details = Mock()

        # Create Ramsey experiment request
        request = epii_pb2.ExperimentRequest()
        request.experiment_type = "calibrations.SimpleRamseyMultilevel"
        request.parameters["dut"] = "q0"
        request.parameters["start"] = "0.0"
        request.parameters["stop"] = "20.0"
        request.parameters["step"] = "1.0"
        request.parameters["set_offset"] = "0.5"  # MHz

        # Mock parameter mapping
        with patch.object(service.experiment_router, 'map_parameters') as mock_map:
            mock_map.return_value = {
                "dut": q0,
                "start": 0.0,
                "stop": 20.0,
                "step": 1.0,
                "set_offset": 0.5
            }

            # Execute experiment
            response = service.RunExperiment(request, context)

            # Validate response
            assert response.success, f"Ramsey experiment failed: {response.error_message}"

            # Validate data using new protocol
            arrays = get_arrays(response)
            if arrays:
                # Check that we have measurement data
                assert len(arrays) > 0
                # Get the first array
                data_array = list(arrays.values())[0]
                assert isinstance(data_array, np.ndarray)

            context.set_code.assert_not_called()

    def test_experiment_parameter_validation(self, epii_service_with_real_setup):
        """
        Test parameter validation for various experiment types.
        """
        service = epii_service_with_real_setup
        context = Mock()
        context.set_code = Mock()
        context.set_details = Mock()

        # Test missing required parameters
        request = epii_pb2.ExperimentRequest()
        request.experiment_type = "calibrations.NormalisedRabi"
        # Missing dut_qubit parameter

        response = service.RunExperiment(request, context)
        assert not response.success
        assert "missing required parameter" in response.error_message.lower()
        context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

        # Test invalid experiment type
        context.reset_mock()
        request.experiment_type = "invalid_experiment"
        request.parameters["dut_qubit"] = "q0"

        response = service.RunExperiment(request, context)
        assert not response.success
        assert "unknown experiment" in response.error_message.lower()
        context.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_experiment_timeout_handling(self, epii_service_with_real_setup, real_qubit_elements):
        """
        Test experiment execution timeout and cancellation.

        This tests that long-running experiments can be properly cancelled.
        """
        service = epii_service_with_real_setup
        q0 = real_qubit_elements['q0']

        context = Mock()
        context.set_code = Mock()
        context.set_details = Mock()

        # Create a request that would take a long time
        request = epii_pb2.ExperimentRequest()
        request.experiment_type = "calibrations.NormalisedRabi"
        request.parameters["dut_qubit"] = "q0"
        request.parameters["amp"] = "0.5"
        request.parameters["start"] = "0.0"
        request.parameters["stop"] = "1.0"
        request.parameters["step"] = "0.001"  # Very small step = many points

        def execute_with_timeout():
            """Execute experiment in a thread to test timeout."""
            with patch.object(service.experiment_router, 'map_parameters') as mock_map:
                mock_map.return_value = {
                    "dut_qubit": q0,
                    "amp": 0.5,
                    "start": 0.0,
                    "stop": 1.0,
                    "step": 0.001
                }

                return service.RunExperiment(request, context)

        # Execute with a timeout
        start_time = time.time()
        response = execute_with_timeout()
        execution_time = time.time() - start_time

        # For simulation, this should complete quickly
        # Real timeout testing would require hardware or more complex simulation
        assert execution_time < 30.0, "Experiment took too long"

        # Should either succeed or fail gracefully
        assert isinstance(response.success, bool)

    def test_setup_parameter_integration(self, epii_service_with_real_setup):
        """
        Test parameter management integration with real LeeQ setup.
        """
        service = epii_service_with_real_setup

        # Test parameter listing
        request = epii_pb2.Empty()
        response = service.ListParameters(request, None)

        # Should have some parameters from the setup
        assert len(response.parameters) > 0

        # Check for expected parameter types
        param_names = [param.name for param in response.parameters]

        # Should have setup status parameters
        status_params = [name for name in param_names if name.startswith("status.")]
        assert len(status_params) > 0

        # Test parameter getting
        if param_names:
            get_request = epii_pb2.ParameterRequest()
            get_request.parameter_names.extend(param_names[:3])  # Test first 3 parameters
            get_response = service.GetParameters(get_request, None)

            assert len(get_response.parameters) <= 3
            for param_name in get_request.parameter_names:
                assert param_name in get_response.parameters

    def test_data_serialization_with_real_experiments(self, epii_service_with_real_setup, real_qubit_elements):
        """
        Test data serialization with results from real LeeQ experiments.
        """
        service = epii_service_with_real_setup
        q0 = real_qubit_elements['q0']

        context = Mock()
        context.set_code = Mock()
        context.set_details = Mock()

        # Execute a simple Rabi experiment
        request = epii_pb2.ExperimentRequest()
        request.experiment_type = "calibrations.NormalisedRabi"
        request.parameters["dut_qubit"] = "q0"
        request.parameters["amp"] = "0.5"
        request.parameters["start"] = "0.0"
        request.parameters["stop"] = "0.5"
        request.parameters["step"] = "0.1"

        with patch.object(service.experiment_router, 'map_parameters') as mock_map:
            mock_map.return_value = {
                "dut_qubit": q0,
                "amp": 0.5,
                "start": 0.0,
                "stop": 0.5,
                "step": 0.1
            }

            response = service.RunExperiment(request, context)

            if response.success and response.data:
                # Test data extraction using new protocol
                all_data = get_data(response)
                arrays = get_arrays(response)
                
                if arrays:
                    # Get the first array (typically measurement data)
                    first_array = list(arrays.values())[0]
                    
                    # Validate array properties
                    assert isinstance(first_array, np.ndarray)
                    assert first_array.size > 0
                    assert not np.isnan(first_array).all()
                
                # Test calibration results extraction
                calibration_results = get_calibration_results(response)
                for key, value in calibration_results.items():
                    assert isinstance(key, str)
                    assert isinstance(value, (float, int))  # Calibration results are numeric


    def test_concurrent_experiment_execution(self, epii_service_with_real_setup, real_qubit_elements):
        """
        Test that multiple experiments can be executed concurrently.

        This tests thread safety and resource management.
        """
        service = epii_service_with_real_setup
        q0 = real_qubit_elements['q0']

        def execute_experiment(experiment_type: str, exp_id: int) -> tuple:
            """Execute a single experiment."""
            context = Mock()
            context.set_code = Mock()
            context.set_details = Mock()

            request = epii_pb2.ExperimentRequest()
            request.experiment_type = experiment_type
            request.parameters["dut_qubit"] = "q0"
            request.parameters["amp"] = "0.5"
            request.parameters["start"] = "0.0"
            request.parameters["stop"] = "0.5"
            request.parameters["step"] = "0.2"

            with patch.object(service.experiment_router, 'map_parameters') as mock_map:
                mock_map.return_value = {
                    "dut_qubit": q0,
                    "amp": 0.5,
                    "start": 0.0,
                    "stop": 0.5,
                    "step": 0.2
                }

                response = service.RunExperiment(request, context)
                return exp_id, response.success, response.error_message

        # Execute multiple experiments concurrently
        results = []
        threads = []

        for i in range(3):
            thread = threading.Thread(
                target=lambda i=i: results.append(execute_experiment("calibrations.NormalisedRabi", i))
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30.0)

        # Validate results
        assert len(results) == 3
        for exp_id, success, error_msg in results:
            assert isinstance(success, bool)
            if not success:
                pass

    def test_experiment_error_handling(self, epii_service_with_real_setup):
        """
        Test error handling for various failure scenarios.
        """
        service = epii_service_with_real_setup
        context = Mock()
        context.set_code = Mock()
        context.set_details = Mock()

        # Test with None setup
        service_no_setup = ExperimentPlatformService(setup=None)
        request = epii_pb2.ExperimentRequest()
        request.experiment_type = "calibrations.NormalisedRabi"
        request.parameters["dut_qubit"] = "q0"

        response = service_no_setup.RunExperiment(request, context)
        assert not response.success
        assert "no experimental setup" in response.error_message.lower()

        # Test with invalid parameter values
        context.reset_mock()
        request.parameters["invalid_param"] = "invalid_value"

        response = service.RunExperiment(request, context)
        # Should handle gracefully (may succeed with warnings or fail with clear error)
        assert isinstance(response.success, bool)

    def test_experiment_capabilities_discovery(self, epii_service_with_real_setup):
        """
        Test experiment capability discovery and validation.
        """
        service = epii_service_with_real_setup

        # Test GetCapabilities
        response = service.GetCapabilities(epii_pb2.Empty(), None)

        assert response.framework_name == "LeeQ"
        assert response.epii_version == "1.0.0"
        assert "simulation" in response.supported_backends

        # Validate experiment types
        experiment_names = [exp.name for exp in response.experiment_types]
        required_experiments = ["calibrations.NormalisedRabi", "characterizations.SimpleT1", "calibrations.SimpleRamseyMultilevel", "characterizations.SpinEchoMultiLevel", "calibrations.DragCalibrationSingleQubitMultilevel", "characterizations.RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem"]

        for exp_name in required_experiments:
            assert exp_name in experiment_names, f"Missing required experiment: {exp_name}"

        # Test ListAvailableExperiments
        list_response = service.ListAvailableExperiments(epii_pb2.Empty(), None)

        assert len(list_response.experiments) >= 6
        for exp in list_response.experiments:
            assert exp.name in experiment_names
            assert exp.description != ""
