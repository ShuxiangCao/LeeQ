"""Mock testing framework for LeeQ experiments."""
from unittest.mock import Mock, MagicMock


class MockTestingFramework:
    """Framework for creating consistent mocks across tests."""
    
    @staticmethod
    def create_qubit():
        """Create a standardized mock qubit."""
        qubit = Mock()
        qubit.__name__ = "framework_test_qubit"
        
        # Mock get_c1 and get_default_c1 methods
        c1_mock = MockTestingFramework.create_pulse_collection()
        qubit.get_c1 = Mock(return_value=c1_mock)
        qubit.get_default_c1 = Mock(return_value=c1_mock)
        
        # Mock measurement primitives
        mp_mock = MockTestingFramework.create_measurement_primitive()
        qubit.get_measurement_prim_intlist = Mock(return_value=mp_mock)
        qubit.get_default_measurement_prim_int = Mock(return_value=mp_mock)
        qubit.get_default_measurement_prim_intlist = Mock(return_value=mp_mock)
        
        return qubit
    
    @staticmethod
    def create_measurement_primitive():
        """Create a mock measurement primitive with proper attributes."""
        mp_mock = Mock()
        mp_mock.__name__ = "measurement_primitive"
        
        # Create update_freq with __name__ attribute
        update_freq_mock = Mock()
        update_freq_mock.__name__ = "update_freq"
        mp_mock.update_freq = update_freq_mock
        
        mp_mock.result = Mock(return_value=[[0.5]])
        return mp_mock
    
    @staticmethod
    def create_pulse_collection():
        """Create a mock pulse collection."""
        c1_mock = Mock()
        c1_mock.__name__ = "c1_collection"
        
        def create_pulse_mock(x):
            pulse = Mock()
            pulse.__name__ = f"pulse_{x}"
            pulse.freq = 5000
            pulse.amp = 0.1
            
            cloned_pulse = Mock()
            cloned_pulse.__name__ = f"pulse_{x}_cloned"
            
            # Create update_pulse_args with __name__ attribute
            update_pulse_args_mock = Mock()
            update_pulse_args_mock.__name__ = "update_pulse_args"
            cloned_pulse.update_pulse_args = update_pulse_args_mock
            cloned_pulse.freq = 5000
            
            pulse.clone = Mock(return_value=cloned_pulse)
            return pulse
        
        c1_mock.__getitem__ = Mock(side_effect=create_pulse_mock)
        return c1_mock
    
    @staticmethod
    def validate_setup(mock_setup):
        """Validate a complete mock setup."""
        required_keys = ['qubit', 'measurement_primitive', 'pulse_collection']
        for key in required_keys:
            if key not in mock_setup:
                return False
        
        # Basic validation - all objects should have __name__ attribute
        for obj in mock_setup.values():
            if not hasattr(obj, '__name__'):
                return False
                
        return True


def create_complete_mock_qubit():
    """Create a complete mock qubit for testing."""
    return MockTestingFramework.create_qubit()