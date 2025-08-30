"""Mock utilities for LeeQ testing."""
from unittest.mock import Mock, MagicMock


def create_mock_function(name):
    """Create a mock function with proper name attribute."""
    mock_func = Mock()
    mock_func.__name__ = name
    return mock_func


def ensure_mock_attributes(mock_obj, obj_type):
    """Ensure mock object has required attributes."""
    if not hasattr(mock_obj, '__name__'):
        mock_obj.__name__ = f"mock_{obj_type}"
    return mock_obj


def create_qubit_mock():
    """Create a mock qubit with required methods and proper attributes."""
    qubit = Mock()
    qubit.__name__ = "test_qubit"
    
    # Mock get_c1 method
    c1_mock = Mock()
    c1_mock.__name__ = "c1_collection"
    
    def create_pulse_mock(x):
        pulse = Mock()
        pulse.__name__ = f"pulse_{x}"
        pulse.freq = 5000
        pulse.amp = 0.1
        
        cloned_pulse = Mock()
        cloned_pulse.__name__ = f"pulse_{x}_cloned"
        cloned_pulse.update_pulse_args = create_mock_function('update_pulse_args')
        cloned_pulse.freq = 5000
        
        pulse.clone = Mock(return_value=cloned_pulse)
        return pulse
    
    c1_mock.__getitem__ = Mock(side_effect=create_pulse_mock)
    qubit.get_c1 = Mock(return_value=c1_mock)
    
    # Mock measurement primitive
    mp_mock = Mock()
    mp_mock.__name__ = "measurement_primitive"
    mp_mock.update_freq = create_mock_function('update_freq')
    mp_mock.result = Mock(return_value=[[0.5]])
    qubit.get_measurement_prim_intlist = Mock(return_value=mp_mock)
    
    return ensure_mock_attributes(qubit, 'qubit')