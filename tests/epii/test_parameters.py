"""
Unit tests for EPII parameter management
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from leeq.epii.parameters import ParameterManager


class TestParameterManager:
    """Test suite for ParameterManager class"""
    
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
            "t1": 20e-6,
            "t2": 15e-6,
            "pi_amp": 0.5,
            "pi_len": 40
        }
        
        q1 = Mock()
        q1._parameters = {
            "f01": 5.1e9,
            "anharmonicity": -0.32e9,
            "t1": 25e-6,
            "t2": 18e-6,
            "pi_amp": 0.45,
            "pi_len": 38
        }
        
        setup._elements = {"q0": q0, "q1": q1}
        
        return setup
    
    def test_init_without_setup(self):
        """Test initialization without a setup"""
        pm = ParameterManager()
        assert pm.setup is None
        assert pm.parameter_cache == {}
        assert len(pm._readonly_params) > 0
    
    def test_init_with_setup(self, mock_setup):
        """Test initialization with a setup"""
        pm = ParameterManager(mock_setup)
        assert pm.setup == mock_setup
        assert pm.parameter_cache == {}
    
    def test_get_status_parameter(self, mock_setup):
        """Test retrieval of setup status parameters"""
        pm = ParameterManager(mock_setup)
        
        # Get existing parameter
        value = pm.get_parameter("status.shot_number")
        assert value == 2000
        
        # Get another parameter
        value = pm.get_parameter("status.shot_period")
        assert value == 500.0
        
        # Test case insensitive
        value = pm.get_parameter("Status.SHOT_NUMBER")
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
    
    def test_set_readonly_parameter(self, mock_setup):
        """Test that read-only parameters cannot be set"""
        pm = ParameterManager(mock_setup)
        
        success = pm.set_parameter("setup.name", "new_name")
        assert success is False
        
        success = pm.set_parameter("engine.status", "new_status")
        assert success is False
    
    def test_list_parameters(self, mock_setup):
        """Test listing all available parameters"""
        pm = ParameterManager(mock_setup)
        
        params = pm.list_parameters()
        assert len(params) > 0
        
        # Check that status parameters are included
        param_names = [p["name"] for p in params]
        assert "status.shot_number" in param_names
        assert "status.shot_period" in param_names
        
        # Check that element parameters are included
        assert "q0.f01" in param_names
        assert "q1.pi_amp" in param_names
        
        # Check parameter structure
        shot_param = next(p for p in params if p["name"] == "status.shot_number")
        assert "type" in shot_param
        assert "current_value" in shot_param
        assert "description" in shot_param
        assert "read_only" in shot_param
    
    def test_validate_shot_number(self, mock_setup):
        """Test validation of shot_number parameter"""
        pm = ParameterManager(mock_setup)
        
        # Valid values
        assert pm.validate_parameter("status.shot_number", 1000) is True
        assert pm.validate_parameter("status.shot_number", "5000") is True
        
        # Invalid values
        assert pm.validate_parameter("status.shot_number", -100) is False
        assert pm.validate_parameter("status.shot_number", 0) is False
        assert pm.validate_parameter("status.shot_number", 2000000) is False
        assert pm.validate_parameter("status.shot_number", "abc") is False
    
    def test_validate_frequency_parameter(self, mock_setup):
        """Test validation of frequency parameters"""
        pm = ParameterManager(mock_setup)
        
        # Valid values
        assert pm.validate_parameter("q0.f01", 5.5e9) is True
        assert pm.validate_parameter("q0.f01", "4.8e9") is True
        
        # Invalid values (negative frequency)
        assert pm.validate_parameter("q0.f01", -1e9) is False
        assert pm.validate_parameter("q0.f01", 0) is False
    
    def test_validate_amplitude_parameter(self, mock_setup):
        """Test validation of amplitude parameters"""
        pm = ParameterManager(mock_setup)
        
        # Valid values
        assert pm.validate_parameter("q0.pi_amp", 0.5) is True
        assert pm.validate_parameter("q0.pi_amp", "0.75") is True
        assert pm.validate_parameter("q0.pi_amp", 0) is True
        assert pm.validate_parameter("q0.pi_amp", 1.0) is True
        
        # Invalid values (outside 0-1 range)
        assert pm.validate_parameter("q0.pi_amp", -0.1) is False
        assert pm.validate_parameter("q0.pi_amp", 1.5) is False
    
    def test_type_conversion(self, mock_setup):
        """Test automatic type conversion"""
        pm = ParameterManager(mock_setup)
        
        # Test _convert_value method
        assert pm._convert_value("true", "debug_plotter") is True
        assert pm._convert_value("false", "debug_plotter") is False
        assert pm._convert_value("123", "shot_number") == 123
        assert pm._convert_value("45.6", "shot_period") == 45.6
        assert pm._convert_value("null", "some_param") is None
        assert pm._convert_value("[1, 2, 3]", "array_param") == [1, 2, 3]
    
    def test_serialize_value(self, mock_setup):
        """Test value serialization to string"""
        pm = ParameterManager(mock_setup)
        
        assert pm._serialize_value(None) == "null"
        assert pm._serialize_value(True) == "true"
        assert pm._serialize_value(False) == "false"
        assert pm._serialize_value(123) == "123"
        assert pm._serialize_value(45.6) == "45.6"
        assert pm._serialize_value("test") == "test"
        assert pm._serialize_value([1, 2, 3]) == "[1, 2, 3]"
        assert pm._serialize_value({"a": 1}) == '{"a": 1}'
    
    def test_get_type_string(self, mock_setup):
        """Test type string mapping"""
        pm = ParameterManager(mock_setup)
        
        assert pm._get_type_string(True) == "bool"
        assert pm._get_type_string(123) == "int"
        assert pm._get_type_string(45.6) == "float"
        assert pm._get_type_string("test") == "string"
        assert pm._get_type_string([1, 2]) == "array"
        assert pm._get_type_string({"a": 1}) == "object"
        assert pm._get_type_string(None) == "null"
    
    def test_parameter_cache_fallback(self):
        """Test that parameter cache is used when no setup"""
        pm = ParameterManager()
        
        # Set parameter in cache
        pm.set_parameter("custom.param", 42)
        assert pm.parameter_cache["custom.param"] == 42
        
        # Get from cache
        value = pm.get_parameter("custom.param")
        assert value == 42
        
        # List includes cached parameters
        params = pm.list_parameters()
        assert len(params) == 1
        assert params[0]["name"] == "custom.param"
        assert params[0]["current_value"] == "42"
    
    def test_get_param_description(self, mock_setup):
        """Test parameter description generation"""
        pm = ParameterManager(mock_setup)
        
        desc = pm._get_param_description("shot_number")
        assert "measurement shots" in desc.lower()
        
        desc = pm._get_param_description("shot_period")
        assert "μs" in desc or "time" in desc.lower()
        
        desc = pm._get_param_description("unknown_param")
        assert "Setup parameter" in desc

    def test_set_parameter_exception_handling(self, mock_setup):
        """Test exception handling in set_parameter"""
        pm = ParameterManager(mock_setup)
        
        # Mock status.set_parameter to raise exception
        mock_setup.status.set_parameter.side_effect = Exception("Database error")
        
        # Should return False on exception
        result = pm.set_parameter("status.shot_number", 1000)
        assert result is False
    
    def test_get_parameter_setup_attributes(self, mock_setup):
        """Test getting setup attributes directly"""
        pm = ParameterManager(mock_setup)
        
        # Test setup attribute access
        value = pm.get_parameter("setup.name")
        assert value == "test_setup"
        
        value = pm.get_parameter("setup.active")
        assert value is True
        
        # Test non-existent setup attribute - Mock will return a Mock object
        # but ParameterManager checks hasattr first, so it should return None
        # Let's test with a setup that doesn't have the attribute
        delattr(mock_setup, '_nonexistent')  # Ensure attribute doesn't exist
        value = pm.get_parameter("setup.nonexistent")
        # This will actually fall back to cache, so it returns None
        assert value is None
    
    def test_convert_value_error_handling(self, mock_setup):
        """Test error handling in _convert_value method"""
        pm = ParameterManager(mock_setup)
        
        # Test with invalid conversion
        with patch.object(pm, '_convert_value') as mock_convert:
            mock_convert.side_effect = ValueError("Invalid conversion")
            
            result = pm.set_parameter("status.shot_number", "invalid")
            assert result is False
    
    def test_validate_parameter_edge_cases(self, mock_setup):
        """Test parameter validation edge cases"""
        pm = ParameterManager(mock_setup)
        
        # Test shot number validation with edge cases (needs status prefix)
        assert pm.validate_parameter("status.shot_number", 1) is True  # Minimum valid
        assert pm.validate_parameter("status.shot_number", 0) is False  # Below minimum
        assert pm.validate_parameter("status.shot_number", -1) is False  # Negative
        assert pm.validate_parameter("status.shot_number", 1000001) is False  # Above maximum
        
        # Test shot period validation 
        assert pm.validate_parameter("status.shot_period", 100.0) is True  # Valid
        assert pm.validate_parameter("status.shot_period", 0) is False  # Zero
        assert pm.validate_parameter("status.shot_period", -1) is False  # Negative
        
        # Test boolean parameter validation
        assert pm.validate_parameter("status.debug_plotter", True) is True
        assert pm.validate_parameter("status.debug_plotter", False) is True
        assert pm.validate_parameter("status.debug_plotter", "true") is True  # String ok
        
        # Test unknown parameter (should default to True)
        assert pm.validate_parameter("status.unknown_param", "any_value") is True
    
    def test_type_conversion_errors(self, mock_setup):
        """Test type conversion error handling"""
        pm = ParameterManager(mock_setup)
        
        # Test conversion of incompatible types
        result = pm._convert_value("not_a_number", "f01")
        assert result == "not_a_number"  # Should return original if conversion fails
        
        # Test None handling
        result = pm._convert_value(None, "shot_number")
        assert result is None
    
    def test_list_parameters_no_setup(self):
        """Test listing parameters when no setup is available"""
        pm = ParameterManager(None)
        
        # Add some cached parameters
        pm.parameter_cache["test.param1"] = 42
        pm.parameter_cache["test.param2"] = "hello"
        
        params = pm.list_parameters()
        assert len(params) == 2
        
        # Check parameter details
        param_names = [p["name"] for p in params]
        assert "test.param1" in param_names
        assert "test.param2" in param_names
    
    def test_get_type_string_edge_cases(self, mock_setup):
        """Test _get_type_string with various types"""
        pm = ParameterManager(mock_setup)
        
        # Test various types
        assert pm._get_type_string(True) == "bool"
        assert pm._get_type_string(42) == "int"
        assert pm._get_type_string(3.14) == "float"
        assert pm._get_type_string("hello") == "string"
        assert pm._get_type_string([1, 2, 3]) == "array"
        assert pm._get_type_string({"key": "value"}) == "object"
        assert pm._get_type_string(None) == "null"
        
        # Test unknown type
        class CustomType:
            pass
        
        custom_obj = CustomType()
        assert pm._get_type_string(custom_obj) == "object"
    
    def test_serialize_value_complex_types(self, mock_setup):
        """Test serialization of complex types"""
        pm = ParameterManager(mock_setup)
        
        # Test complex number (should be converted to string)
        result = pm._serialize_value(1+2j)
        assert result == "(1+2j)"
        
        # Test custom object (should use string representation)
        class CustomClass:
            def __str__(self):
                return "custom_object"
        
        result = pm._serialize_value(CustomClass())
        assert result == "custom_object"
        
        # Test nested dictionary
        nested_dict = {"level1": {"level2": {"value": 42}}}
        result = pm._serialize_value(nested_dict)
        assert isinstance(result, str)
        assert "level1" in result
    
    def test_parameter_cache_operations(self, mock_setup):
        """Test parameter cache edge cases"""
        pm = ParameterManager(mock_setup)
        
        # Test setting parameter that doesn't exist in setup
        result = pm.set_parameter("custom.deeply.nested.param", "value")
        assert result is True
        assert pm.parameter_cache["custom.deeply.nested.param"] == "value"
        
        # Test getting cached parameter
        value = pm.get_parameter("custom.deeply.nested.param")
        assert value == "value"
        
        # Test overwriting cached parameter
        result = pm.set_parameter("custom.deeply.nested.param", "new_value")
        assert result is True
        assert pm.parameter_cache["custom.deeply.nested.param"] == "new_value"
    
    def test_element_parameter_edge_cases(self, mock_setup):
        """Test element parameter access edge cases"""
        pm = ParameterManager(mock_setup)
        
        # Test getting parameter from element that doesn't have _parameters
        mock_element_no_params = Mock()
        del mock_element_no_params._parameters  # Ensure no _parameters attribute
        mock_setup._elements["no_params_element"] = mock_element_no_params
        
        value = pm.get_parameter("no_params_element.f01")
        assert value is None
        
        # Test setting parameter on element without _parameters
        result = pm.set_parameter("no_params_element.f01", 5e9)
        assert result is True  # Should fall back to cache
        assert pm.parameter_cache["no_params_element.f01"] == 5e9