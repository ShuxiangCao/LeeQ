import pytest
from unittest.mock import Mock, MagicMock
from .mock_utilities import create_qubit_mock, create_mock_function, ensure_mock_attributes


@pytest.fixture
def mock_qubit():
    """Create a mock qubit with required methods and proper attributes."""
    # Use the new utility function for consistent mock creation
    return create_qubit_mock()


@pytest.fixture 
def legacy_mock_qubit():
    """Legacy mock qubit fixture - updated to include proper attributes."""
    qubit = Mock()
    qubit.__name__ = "legacy_test_qubit"
    
    # Mock get_c1 method with proper attributes
    c1_mock = Mock()
    c1_mock.__name__ = "c1_collection"
    
    def create_pulse_mock(x):
        pulse = Mock()
        pulse.__name__ = f"pulse_{x}"
        pulse.freq = 5000
        pulse.amp = 0.1
        
        # Create cloned pulse with proper function attributes
        cloned_pulse = Mock()
        cloned_pulse.__name__ = f"pulse_{x}_cloned"
        cloned_pulse.update_pulse_args = create_mock_function('update_pulse_args')
        cloned_pulse.freq = 5000
        
        pulse.clone = Mock(return_value=cloned_pulse)
        return pulse
    
    c1_mock.__getitem__ = Mock(side_effect=create_pulse_mock)
    qubit.get_c1 = Mock(return_value=c1_mock)
    
    # Mock measurement primitive with proper attributes
    mp_mock = Mock()
    mp_mock.__name__ = "measurement_primitive"
    mp_mock.update_freq = create_mock_function('update_freq')
    mp_mock.result = Mock(return_value=[[0.5]])
    qubit.get_measurement_prim_intlist = Mock(return_value=mp_mock)
    
    # Ensure all required attributes are present
    return ensure_mock_attributes(qubit, 'qubit')