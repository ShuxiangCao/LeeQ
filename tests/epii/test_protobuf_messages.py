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
        exp_spec.name = "calibrations.NormalisedRabi"
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
        assert msg.experiment_types[0].name == "calibrations.NormalisedRabi"
        assert len(msg.experiment_types[0].parameters) == 1
        assert msg.extensions["custom_feature"] == "enabled"

    def test_experiment_request(self):
        """Test ExperimentRequest message"""
        msg = epii_pb2.ExperimentRequest()
        msg.experiment_type = "characterizations.SimpleT1"
        msg.parameters["delay_range"] = "[0, 100e-6, 1e-6]"
        msg.parameters["initial_state"] = "1"
        msg.return_raw_data = True
        msg.return_plots = False

        assert msg.experiment_type == "characterizations.SimpleT1"
        assert len(msg.parameters) == 2
        assert msg.parameters["delay_range"] == "[0, 100e-6, 1e-6]"
        assert msg.return_raw_data is True
        assert msg.return_plots is False

    def test_experiment_response(self):
        """Test ExperimentResponse message with all data types"""
        msg = epii_pb2.ExperimentResponse()
        msg.success = True
        msg.execution_time_seconds = 2.5

        # Add documentation via nested message
        msg.docs.run = "Test run documentation"
        msg.docs.data = "Test data documentation"
        
        # Add metadata
        msg.metadata["purpose"] = "testing"
        msg.metadata["category"] = "unit_test"
        
        # Add data items
        # Add calibration results as DataItems
        item1 = msg.data.add()
        item1.name = "t1_time"
        item1.description = "T1 decay time"
        item1.number = 35.6e-6
        
        item2 = msg.data.add()
        item2.name = "decay_constant"
        item2.description = "Decay constant"
        item2.number = 0.98
        
        # Add numpy array data as DataItem
        item3 = msg.data.add()
        item3.name = "raw_data"
        item3.description = "Raw measurement data"
        item3.array.data = b'\x00\x01\x02\x03'  # Sample binary data
        item3.array.shape.extend([2, 2])
        item3.array.dtype = "float64"
        item3.array.name = "raw_data"
        item3.array.metadata["units"] = "V"

        # Add plot component
        component = msg.plots.add()
        component.description = "T1 Decay (from plot)"
        component.plotly_json = ""
        component.image_png = b""

        # Validate
        assert msg.success is True
        assert len(msg.data) == 3  # 2 calibration results + 1 array
        assert msg.data[0].name == "t1_time"
        assert msg.data[0].description == "T1 decay time"
        assert msg.data[0].number == 35.6e-6
        assert msg.data[0].WhichOneof('value') == 'number'
        assert msg.data[1].name == "decay_constant"
        assert msg.data[1].number == 0.98
        assert msg.data[2].name == "raw_data"
        assert msg.data[2].HasField('array')
        assert list(msg.data[2].array.shape) == [2, 2]
        assert msg.docs.run == "Test run documentation"
        assert msg.docs.data == "Test data documentation"
        assert msg.metadata["purpose"] == "testing"
        assert msg.metadata["category"] == "unit_test"
        assert len(msg.plots) == 1
        assert msg.plots[0].description == "T1 Decay (from plot)"
        assert msg.plots[0].plotly_json == ""
        assert msg.plots[0].image_png == b""

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
        msg.metadata["experiment"] = "calibrations.NormalisedRabi"
        msg.metadata["timestamp"] = "2024-01-01T00:00:00"

        assert msg.data == b'\x00\x00\x00\x00\x00\x00\xf0?'
        assert list(msg.shape) == [1]
        assert msg.dtype == "float64"
        assert len(msg.metadata) == 2
        assert msg.metadata["experiment"] == "calibrations.NormalisedRabi"

    def test_plot_component_message(self):
        """Test PlotComponent message creation and serialization"""
        component = epii_pb2.PlotComponent()
        component.description = "Time Rabi (from plot)"
        component.plotly_json = ""
        component.image_png = b""
        
        # Test serialization roundtrip
        serialized = component.SerializeToString()
        component_copy = epii_pb2.PlotComponent()
        component_copy.ParseFromString(serialized)
        
        assert component_copy.description == component.description
        assert component_copy.plotly_json == ""
        assert component_copy.image_png == b""

    def test_experiment_response_with_plot_components(self):
        """Test ExperimentResponse with PlotComponent array"""
        response = epii_pb2.ExperimentResponse()
        
        # Add plot component
        component = response.plots.add()
        component.description = "Test Plot (from plot_function)"
        component.plotly_json = ""
        component.image_png = b""
        
        assert len(response.plots) == 1
        assert response.plots[0].description == "Test Plot (from plot_function)"

    def test_documentation_message(self):
        """Test Documentation message creation and serialization"""
        msg = epii_pb2.Documentation()
        msg.run = "This is how to use the experiment"
        msg.data = "This describes what the data means"
        
        assert msg.run == "This is how to use the experiment"
        assert msg.data == "This describes what the data means"
        
        # Test serialization round-trip
        serialized = msg.SerializeToString()
        msg2 = epii_pb2.Documentation()
        msg2.ParseFromString(serialized)
        assert msg2.run == msg.run
        assert msg2.data == msg.data
    
    def test_data_item_with_number(self):
        """Test DataItem message with number value"""
        msg = epii_pb2.DataItem()
        msg.name = "pi_amplitude"
        msg.description = "Calibrated pi pulse amplitude"
        msg.number = 0.456
        
        assert msg.name == "pi_amplitude"
        assert msg.description == "Calibrated pi pulse amplitude"
        assert msg.number == 0.456
        assert msg.WhichOneof('value') == 'number'
    
    def test_data_item_with_text(self):
        """Test DataItem message with text value"""
        msg = epii_pb2.DataItem()
        msg.name = "experiment_id"
        msg.description = "Unique experiment identifier"
        msg.text = "exp_20240101_001"
        
        assert msg.name == "experiment_id"
        assert msg.description == "Unique experiment identifier"
        assert msg.text == "exp_20240101_001"
        assert msg.WhichOneof('value') == 'text'
    
    def test_data_item_with_boolean(self):
        """Test DataItem message with boolean value"""
        msg = epii_pb2.DataItem()
        msg.name = "is_calibrated"
        msg.description = "Whether the system is calibrated"
        msg.boolean = True
        
        assert msg.name == "is_calibrated"
        assert msg.description == "Whether the system is calibrated"
        assert msg.boolean is True
        assert msg.WhichOneof('value') == 'boolean'
    
    def test_data_item_with_array(self):
        """Test DataItem message with numpy array value"""
        msg = epii_pb2.DataItem()
        msg.name = "measurement_data"
        msg.description = "Raw measurement data array"
        
        # Set up the array
        msg.array.data = b'\x00\x00\x00\x00\x00\x00\xf0?'  # 1.0 in float64
        msg.array.shape.extend([1])
        msg.array.dtype = "float64"
        msg.array.name = "raw_data"
        
        assert msg.name == "measurement_data"
        assert msg.description == "Raw measurement data array"
        assert msg.HasField('array')
        assert msg.WhichOneof('value') == 'array'
        assert list(msg.array.shape) == [1]
        assert msg.array.dtype == "float64"
    
    def test_data_item_serialization(self):
        """Test DataItem serialization with different value types"""
        # Test with number
        msg1 = epii_pb2.DataItem()
        msg1.name = "test_number"
        msg1.description = "A test number"
        msg1.number = 3.14159
        
        serialized1 = msg1.SerializeToString()
        msg1_copy = epii_pb2.DataItem()
        msg1_copy.ParseFromString(serialized1)
        assert msg1_copy.name == msg1.name
        assert msg1_copy.number == msg1.number
        
        # Test with text
        msg2 = epii_pb2.DataItem()
        msg2.name = "test_text"
        msg2.description = "A test string"
        msg2.text = "Hello, EPII!"
        
        serialized2 = msg2.SerializeToString()
        msg2_copy = epii_pb2.DataItem()
        msg2_copy.ParseFromString(serialized2)
        assert msg2_copy.text == msg2.text
    
    def test_message_serialization_size(self):
        """Test that messages serialize to reasonable sizes"""
        # Small message
        small = epii_pb2.StatusResponse()
        small.success = True
        small_size = len(small.SerializeToString())
        assert small_size < 100  # Should be very small

        # Medium message with some data
        medium = epii_pb2.ExperimentRequest()
        medium.experiment_type = "calibrations.SimpleRamseyMultilevel"
        for i in range(10):
            medium.parameters[f"param_{i}"] = str(i * 0.1)
        medium_size = len(medium.SerializeToString())
        assert medium_size < 1000  # Should be under 1KB

        # Larger message with array data
        large = epii_pb2.ExperimentResponse()
        large.success = True
        
        # Add data items instead of calibration_results
        for i in range(5):
            item = large.data.add()
            item.name = f"result_{i}"
            item.description = f"Result {i}"
            item.number = i * 1.5

        # Add some array data as DataItem
        array_item = large.data.add()
        array_item.name = "large_array"
        array_item.description = "Large test array"
        array_item.array.data = b'\x00' * 1000  # 1KB of data
        array_item.array.shape.extend([125, 8])
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
