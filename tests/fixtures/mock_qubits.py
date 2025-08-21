import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_qubit():
    """Create a mock qubit with required methods."""
    qubit = Mock()
    
    # Mock get_c1 method
    c1_mock = Mock()
    c1_mock.__getitem__ = Mock(side_effect=lambda x: Mock(
        freq=5000,
        amp=0.1,
        clone=Mock(return_value=Mock(
            update_pulse_args=Mock(),
            freq=5000
        ))
    ))
    qubit.get_c1 = Mock(return_value=c1_mock)
    
    # Mock measurement primitive
    mp_mock = Mock()
    mp_mock.result = Mock(return_value=[[0.5]])
    qubit.get_measurement_prim_intlist = Mock(return_value=mp_mock)
    
    return qubit