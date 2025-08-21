"""
Tests for AC stark shift experiments.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import contextlib

# Mock external dependencies
mock_modules = {
    'k_agents.inspection.decorator': MagicMock(),
}

# Apply mocks before importing
for module_name, mock_module in mock_modules.items():
    import sys
    sys.modules[module_name] = mock_module

# Mock the setup function before importing experiments
from unittest.mock import patch

# Create a mock setup that returns the expected status parameters
class MockSetupStatusParameters:
    def get_parameters(self, key):
        return False  # Default to False for simulation mode and other flags
    
    def with_parameters(self, **kwargs):
        return contextlib.nullcontext()

class MockSetup:
    def __init__(self):
        self.status = MockSetupStatusParameters()

    def get_default_setup(self):
        return self

# Patch setup function before importing
with patch('leeq.experiments.experiments.setup', return_value=MockSetup()):
    from leeq.experiments.builtin.multi_qubit_gates.ac_stark.ac_stark_shift import (
        StarkSingleQubitT1,
        StarkTwoQubitsSWAP,
        StarkTwoQubitsSWAPTwoDrives,
        StarkRamseyMultilevel,
        StarkDriveRamseyTwoQubits,
        StarkDriveRamseyTwoQubitsTwoStarkDrives,
        StarkDriveRamseyMultiQubits,
        StarkZZShiftTwoQubitMultilevel,
        StarkRepeatedGateRabi,
        StarkContinuesRabi,
        StarkRepeatedGateDRAGLeakageCalibration
    )

from tests.fixtures.mock_qubits import mock_qubit


class TestStarkSingleQubitT1:
    """Test the StarkSingleQubitT1 experiment."""
    
    @pytest.fixture
    def mock_qubit(self):
        """Create a mock qubit element."""
        qubit = Mock()
        qubit.get_gate.return_value = Mock()
        qubit.get_measurement_primitive.return_value = Mock()
        qubit.name = "test_qubit"
        return qubit
    
    @patch.object(StarkSingleQubitT1, 'run')
    @patch('leeq.experiments.experiments.setup')
    def test_initialization(self, mock_setup_func, mock_run, mock_qubit):
        """Test that StarkSingleQubitT1 can be instantiated."""
        # Mock the setup status
        mock_status = Mock()
        mock_status.get_parameters.return_value = False
        mock_status.with_parameters.return_value = contextlib.nullcontext()
        
        mock_setup = Mock()
        mock_setup.status.return_value = mock_status
        mock_setup.get_default_setup.return_value = mock_setup
        mock_setup_func.return_value = mock_setup
        
        # Mock run method to prevent execution during initialization
        mock_run.return_value = None
        
        exp = StarkSingleQubitT1(qubit=mock_qubit)
        assert exp is not None
        assert mock_run.called  # Ensure run was called during init
    
    def test_class_exists(self):
        """Test that the StarkSingleQubitT1 class exists and is importable."""
        assert StarkSingleQubitT1 is not None
        assert hasattr(StarkSingleQubitT1, '__name__')
        assert StarkSingleQubitT1.__name__ == 'StarkSingleQubitT1'


class TestStarkTwoQubitsSWAP:
    """Test the StarkTwoQubitsSWAP experiment."""
    
    def test_class_exists(self):
        """Test that the StarkTwoQubitsSWAP class exists and is importable."""
        assert StarkTwoQubitsSWAP is not None
        assert hasattr(StarkTwoQubitsSWAP, '__name__')
        assert StarkTwoQubitsSWAP.__name__ == 'StarkTwoQubitsSWAP'


class TestStarkTwoQubitsSWAPTwoDrives:
    """Test the StarkTwoQubitsSWAPTwoDrives experiment."""
    
    def test_class_exists(self):
        """Test that the StarkTwoQubitsSWAPTwoDrives class exists and is importable."""
        assert StarkTwoQubitsSWAPTwoDrives is not None
        assert hasattr(StarkTwoQubitsSWAPTwoDrives, '__name__')
        assert StarkTwoQubitsSWAPTwoDrives.__name__ == 'StarkTwoQubitsSWAPTwoDrives'


class TestStarkRamseyMultilevel:
    """Test the StarkRamseyMultilevel experiment."""
    
    def test_class_exists(self):
        """Test that the StarkRamseyMultilevel class exists and is importable."""
        assert StarkRamseyMultilevel is not None
        assert hasattr(StarkRamseyMultilevel, '__name__')
        assert StarkRamseyMultilevel.__name__ == 'StarkRamseyMultilevel'


class TestStarkDriveRamsey:
    """Test the various Stark drive Ramsey experiments."""
    
    def test_stark_drive_ramsey_two_qubits_exists(self):
        """Test StarkDriveRamseyTwoQubits class exists."""
        assert StarkDriveRamseyTwoQubits is not None
        assert hasattr(StarkDriveRamseyTwoQubits, '__name__')
        assert StarkDriveRamseyTwoQubits.__name__ == 'StarkDriveRamseyTwoQubits'
    
    def test_stark_drive_ramsey_two_qubits_two_drives_exists(self):
        """Test StarkDriveRamseyTwoQubitsTwoStarkDrives class exists."""
        assert StarkDriveRamseyTwoQubitsTwoStarkDrives is not None
        assert hasattr(StarkDriveRamseyTwoQubitsTwoStarkDrives, '__name__')
        assert StarkDriveRamseyTwoQubitsTwoStarkDrives.__name__ == 'StarkDriveRamseyTwoQubitsTwoStarkDrives'
    
    def test_stark_drive_ramsey_multi_qubits_exists(self):
        """Test StarkDriveRamseyMultiQubits class exists."""
        assert StarkDriveRamseyMultiQubits is not None
        assert hasattr(StarkDriveRamseyMultiQubits, '__name__')
        assert StarkDriveRamseyMultiQubits.__name__ == 'StarkDriveRamseyMultiQubits'


class TestStarkZZShift:
    """Test the StarkZZShiftTwoQubitMultilevel experiment."""
    
    def test_class_exists(self):
        """Test StarkZZShiftTwoQubitMultilevel class exists."""
        assert StarkZZShiftTwoQubitMultilevel is not None
        assert hasattr(StarkZZShiftTwoQubitMultilevel, '__name__')
        assert StarkZZShiftTwoQubitMultilevel.__name__ == 'StarkZZShiftTwoQubitMultilevel'


class TestStarkRabi:
    """Test the Stark Rabi experiments."""
    
    def test_stark_repeated_gate_rabi_exists(self):
        """Test StarkRepeatedGateRabi class exists."""
        assert StarkRepeatedGateRabi is not None
        assert hasattr(StarkRepeatedGateRabi, '__name__')
        assert StarkRepeatedGateRabi.__name__ == 'StarkRepeatedGateRabi'
    
    def test_stark_continues_rabi_exists(self):
        """Test StarkContinuesRabi class exists."""
        assert StarkContinuesRabi is not None
        assert hasattr(StarkContinuesRabi, '__name__')
        assert StarkContinuesRabi.__name__ == 'StarkContinuesRabi'
    
    def test_stark_repeated_gate_drag_leakage_calibration_exists(self):
        """Test StarkRepeatedGateDRAGLeakageCalibration class exists."""
        assert StarkRepeatedGateDRAGLeakageCalibration is not None
        assert hasattr(StarkRepeatedGateDRAGLeakageCalibration, '__name__')
        assert StarkRepeatedGateDRAGLeakageCalibration.__name__ == 'StarkRepeatedGateDRAGLeakageCalibration'


class TestParameterValidation:
    """Test parameter validation across all experiments."""
    
    def test_numpy_imports_work(self):
        """Test that numpy is properly imported for sweep parameters."""
        sweep_param = np.linspace(0, 100, 10)
        assert len(sweep_param) == 10
        assert np.all(sweep_param >= 0)


@pytest.mark.integration
class TestExperimentIntegration:
    """Integration tests for AC Stark experiments."""
    
    def test_all_experiments_importable(self):
        """Test that all experiments can be imported successfully."""
        experiments = [
            StarkSingleQubitT1,
            StarkTwoQubitsSWAP,
            StarkTwoQubitsSWAPTwoDrives,
            StarkRamseyMultilevel,
            StarkDriveRamseyTwoQubits,
            StarkDriveRamseyTwoQubitsTwoStarkDrives,
            StarkDriveRamseyMultiQubits,
            StarkZZShiftTwoQubitMultilevel,
            StarkRepeatedGateRabi,
            StarkContinuesRabi,
            StarkRepeatedGateDRAGLeakageCalibration
        ]
        
        for exp_class in experiments:
            assert exp_class is not None
            assert hasattr(exp_class, '__name__')
            # Verify it's a proper class
            assert callable(exp_class)
    
    def test_experiment_inheritance(self):
        """Test that experiments inherit from expected base classes."""
        # These are basic existence and callable tests
        experiments = [StarkSingleQubitT1, StarkTwoQubitsSWAP, StarkRamseyMultilevel]
        
        for exp_class in experiments:
            # Check that the class has typical experiment methods
            # This is a basic structural test without instantiation
            assert hasattr(exp_class, '__init__')
            assert callable(getattr(exp_class, '__init__'))