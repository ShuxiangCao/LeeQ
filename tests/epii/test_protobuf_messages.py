"""
Tests for EPII Protocol Buffer message validation and serialization
"""

import pytest
from leeq.epii.proto import epii_pb2, epii_pb2_grpc


class TestProtobufMessages:
    """Test suite for protobuf message creation and validation"""
    
    def test_empty_message(self):
        """Test Empty message creation"""
        msg = epii_pb2.Empty()
        assert msg is not None
        # Empty message should serialize to empty bytes
        assert len(msg.SerializeToString()) == 0
    
    def test_ping_response(self):
        """Test PingResponse message"""
        msg = epii_pb2.PingResponse()
        msg.message = "pong"
        msg.timestamp = 1234567890.123
        
        assert msg.message == "pong"
        assert msg.timestamp == 1234567890.123
        
        # Test serialization round-trip
        serialized = msg.SerializeToString()
        msg2 = epii_pb2.PingResponse()
        msg2.ParseFromString(serialized)
        assert msg2.message == msg.message
        assert msg2.timestamp == msg.timestamp
    
    def test_capabilities_response(self):
        """Test CapabilitiesResponse message with all fields"""
        msg = epii_pb2.CapabilitiesResponse()
        msg.framework_name = "LeeQ"
        msg.framework_version = "0.1.0"
        msg.epii_version = "1.0.0"
        msg.supported_backends.extend(["simulation", "hardware"])
        msg.data_formats.extend(["numpy", "json"])
        msg.extensions["custom_feature"] = "enabled"
        
        # Add experiment spec
        exp_spec = msg.experiment_types.add()
        exp_spec.name = "rabi"
        exp_spec.description = "Rabi oscillation experiment"
        exp_spec.output_parameters.extend(["pi_amplitude", "rabi_frequency"])
        
        # Add parameter spec
        param_spec = exp_spec.parameters.add()
        param_spec.name = "amplitude_range"
        param_spec.type = "array"
        param_spec.required = True
        param_spec.description = "Range of amplitudes to sweep"
        
        # Validate fields
        assert msg.framework_name == "LeeQ"
        assert len(msg.supported_backends) == 2
        assert "simulation" in msg.supported_backends
        assert len(msg.experiment_types) == 1
        assert msg.experiment_types[0].name == "rabi"
        assert len(msg.experiment_types[0].parameters) == 1
        assert msg.extensions["custom_feature"] == "enabled"
    
    def test_experiment_request(self):
        """Test ExperimentRequest message"""
        msg = epii_pb2.ExperimentRequest()
        msg.experiment_type = "t1"
        msg.parameters["delay_range"] = "[0, 100e-6, 1e-6]"
        msg.parameters["initial_state"] = "1"
        msg.return_raw_data = True
        msg.return_plots = False
        
        assert msg.experiment_type == "t1"
        assert len(msg.parameters) == 2
        assert msg.parameters["delay_range"] == "[0, 100e-6, 1e-6]"
        assert msg.return_raw_data is True
        assert msg.return_plots is False
    
    def test_experiment_response(self):
        """Test ExperimentResponse message with all data types"""
        msg = epii_pb2.ExperimentResponse()
        msg.success = True
        msg.execution_time_seconds = 2.5
        
        # Add calibration results
        msg.calibration_results["t1_time"] = 35.6e-6
        msg.calibration_results["decay_constant"] = 0.98
        
        # Add numpy array data
        numpy_data = msg.measurement_data.add()
        numpy_data.data = b'\x00\x01\x02\x03'  # Sample binary data
        numpy_data.shape.extend([2, 2])
        numpy_data.dtype = "float64"
        numpy_data.name = "raw_data"
        numpy_data.metadata["units"] = "V"
        
        # Add plot data
        plot = msg.plots.add()
        plot.plot_type = "scatter"
        plot.title = "T1 Decay"
        
        trace = plot.traces.add()
        trace.x.extend([0.0, 1.0, 2.0])
        trace.y.extend([1.0, 0.5, 0.25])
        trace.name = "Data"
        trace.type = "scatter"
        
        # Validate
        assert msg.success is True
        assert len(msg.calibration_results) == 2
        assert msg.calibration_results["t1_time"] == 35.6e-6
        assert len(msg.measurement_data) == 1
        assert msg.measurement_data[0].name == "raw_data"
        assert list(msg.measurement_data[0].shape) == [2, 2]
        assert len(msg.plots) == 1
        assert len(msg.plots[0].traces) == 1
        assert len(msg.plots[0].traces[0].x) == 3
    
    def test_parameter_info(self):
        """Test ParameterInfo message"""
        msg = epii_pb2.ParameterInfo()
        msg.name = "qubit_frequency"
        msg.type = "float"
        msg.current_value = "5.123e9"
        msg.description = "Qubit transition frequency"
        msg.read_only = False
        
        assert msg.name == "qubit_frequency"
        assert msg.type == "float"
        assert msg.current_value == "5.123e9"
        assert msg.read_only is False
    
    def test_parameters_list_response(self):
        """Test ParametersListResponse with multiple parameters"""
        msg = epii_pb2.ParametersListResponse()
        
        # Add first parameter
        param1 = msg.parameters.add()
        param1.name = "frequency"
        param1.type = "float"
        param1.current_value = "5.0e9"
        param1.description = "Frequency in Hz"
        param1.read_only = False
        
        # Add second parameter
        param2 = msg.parameters.add()
        param2.name = "power"
        param2.type = "float"
        param2.current_value = "-10"
        param2.description = "Power in dBm"
        param2.read_only = True
        
        assert len(msg.parameters) == 2
        assert msg.parameters[0].name == "frequency"
        assert msg.parameters[1].read_only is True
    
    def test_set_parameters_request(self):
        """Test SetParametersRequest message"""
        msg = epii_pb2.SetParametersRequest()
        msg.parameters["frequency"] = "5.1e9"
        msg.parameters["amplitude"] = "0.5"
        
        assert len(msg.parameters) == 2
        assert msg.parameters["frequency"] == "5.1e9"
        assert msg.parameters["amplitude"] == "0.5"
    
    def test_status_response(self):
        """Test StatusResponse for both success and error cases"""
        # Success case
        msg_success = epii_pb2.StatusResponse()
        msg_success.success = True
        assert msg_success.success is True
        assert msg_success.error_message == ""  # Default empty
        
        # Error case
        msg_error = epii_pb2.StatusResponse()
        msg_error.success = False
        msg_error.error_message = "Parameter out of range"
        assert msg_error.success is False
        assert msg_error.error_message == "Parameter out of range"
    
    def test_numpy_array_message(self):
        """Test NumpyArray message with metadata"""
        msg = epii_pb2.NumpyArray()
        msg.data = b'\x00\x00\x00\x00\x00\x00\xf0?'  # 1.0 in float64
        msg.shape.extend([1])
        msg.dtype = "float64"
        msg.name = "measurement"
        msg.metadata["experiment"] = "rabi"
        msg.metadata["timestamp"] = "2024-01-01T00:00:00"
        
        assert msg.data == b'\x00\x00\x00\x00\x00\x00\xf0?'
        assert list(msg.shape) == [1]
        assert msg.dtype == "float64"
        assert len(msg.metadata) == 2
        assert msg.metadata["experiment"] == "rabi"
    
    def test_plot_data_message(self):
        """Test PlotData message with traces"""
        msg = epii_pb2.PlotData()
        msg.plot_type = "line"
        msg.title = "Rabi Oscillation"
        msg.layout["xaxis_title"] = "Amplitude"
        msg.layout["yaxis_title"] = "Population"
        
        # Add trace
        trace = msg.traces.add()
        trace.x.extend([0.0, 0.1, 0.2, 0.3])
        trace.y.extend([0.0, 0.5, 1.0, 0.5])
        trace.name = "Qubit 0"
        trace.type = "scatter"
        
        assert msg.plot_type == "line"
        assert msg.title == "Rabi Oscillation"
        assert len(msg.layout) == 2
        assert len(msg.traces) == 1
        assert len(msg.traces[0].x) == 4
        assert msg.traces[0].name == "Qubit 0"
    
    def test_message_serialization_size(self):
        """Test that messages serialize to reasonable sizes"""
        # Small message
        small = epii_pb2.StatusResponse()
        small.success = True
        small_size = len(small.SerializeToString())
        assert small_size < 100  # Should be very small
        
        # Medium message with some data
        medium = epii_pb2.ExperimentRequest()
        medium.experiment_type = "ramsey"
        for i in range(10):
            medium.parameters[f"param_{i}"] = str(i * 0.1)
        medium_size = len(medium.SerializeToString())
        assert medium_size < 1000  # Should be under 1KB
        
        # Larger message with array data
        large = epii_pb2.ExperimentResponse()
        large.success = True
        for i in range(5):
            large.calibration_results[f"result_{i}"] = i * 1.5
        
        # Add some array data
        numpy_data = large.measurement_data.add()
        numpy_data.data = b'\x00' * 1000  # 1KB of data
        numpy_data.shape.extend([125, 8])
        large_size = len(large.SerializeToString())
        assert large_size > 1000  # Should be over 1KB
        assert large_size < 10000  # But under 10KB for this test data


class TestServiceStub:
    """Test that gRPC service stub is properly generated"""
    
    def test_service_stub_exists(self):
        """Test that the service stub class exists"""
        assert hasattr(epii_pb2_grpc, 'ExperimentPlatformServiceStub')
        assert hasattr(epii_pb2_grpc, 'ExperimentPlatformServiceServicer')
        assert hasattr(epii_pb2_grpc, 'add_ExperimentPlatformServiceServicer_to_server')
    
    def test_service_methods_defined(self):
        """Test that all required service methods are defined"""
        servicer = epii_pb2_grpc.ExperimentPlatformServiceServicer
        
        # Check all required methods exist
        assert hasattr(servicer, 'GetCapabilities')
        assert hasattr(servicer, 'Ping')
        assert hasattr(servicer, 'RunExperiment')
        assert hasattr(servicer, 'ListAvailableExperiments')
        assert hasattr(servicer, 'ListParameters')
        assert hasattr(servicer, 'GetParameters')
        assert hasattr(servicer, 'SetParameters')