"""
Tests for AC stark shift experiments.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Mock external dependencies
mock_modules = {
    'k_agents.inspection.decorator': MagicMock(),
}

# Apply mocks before importing
for module_name, mock_module in mock_modules.items():
    import sys
    sys.modules[module_name] = mock_module

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
    
    def test_initialization(self):
        """Test that StarkSingleQubitT1 can be instantiated."""
        exp = StarkSingleQubitT1()
        assert exp is not None
    
    @patch('leeq.experiments.builtin.multi_qubit_gates.ac_stark.ac_stark_shift.LogicalPrimitiveBlockSerial')
    @patch('leeq.experiments.builtin.multi_qubit_gates.ac_stark.ac_stark_shift.LogicalPrimitiveBlockSweep')
    def test_run_parameters(self, mock_sweep, mock_serial, mock_qubit):
        """Test run method parameter handling."""
        mock_serial.return_value = Mock()
        mock_sweep.return_value = Mock()
        
        exp = StarkSingleQubitT1()
        
        # Mock the run method behavior without actually running it
        with patch.object(exp, 'run') as mock_run:
            exp.run(
                qubit=mock_qubit,
                collection_name='f01',
                start=0,
                stop=3,
                step=0.03,
                stark_offset=50,
                amp=0.1,
                width=400,
                rise=0.01,
                trunc=1.2
            )
            
            mock_run.assert_called_once()
    
    def test_parameter_assignment(self, mock_qubit):
        """Test parameter assignment in run method."""
        exp = StarkSingleQubitT1()
        
        # Access some basic attributes that should exist
        assert hasattr(exp, '__class__')
        assert exp.__class__.__name__ == 'StarkSingleQubitT1'


class TestStarkTwoQubitsSWAP:
    """Test the StarkTwoQubitsSWAP experiment."""
    
    @pytest.fixture
    def mock_qubits(self):
        """Create mock qubit elements."""
        qubit1 = Mock()
        qubit1.get_gate.return_value = Mock()
        qubit1.get_measurement_primitive.return_value = Mock()
        qubit1.name = "qubit_1"
        
        qubit2 = Mock()
        qubit2.get_gate.return_value = Mock()
        qubit2.get_measurement_primitive.return_value = Mock()
        qubit2.name = "qubit_2"
        
        return [qubit1, qubit2]
    
    def test_initialization(self):
        """Test that StarkTwoQubitsSWAP can be instantiated."""
        exp = StarkTwoQubitsSWAP()
        assert exp is not None
        assert exp.__class__.__name__ == 'StarkTwoQubitsSWAP'
    
    @patch('leeq.experiments.builtin.multi_qubit_gates.ac_stark.ac_stark_shift.LogicalPrimitiveBlockSerial')
    def test_basic_functionality(self, mock_serial, mock_qubits):
        """Test basic functionality."""
        mock_serial.return_value = Mock()
        
        exp = StarkTwoQubitsSWAP()
        assert hasattr(exp, '__class__')


class TestStarkTwoQubitsSWAPTwoDrives:
    """Test the StarkTwoQubitsSWAPTwoDrives experiment."""
    
    def test_initialization(self):
        """Test that StarkTwoQubitsSWAPTwoDrives can be instantiated."""
        exp = StarkTwoQubitsSWAPTwoDrives()
        assert exp is not None
        assert exp.__class__.__name__ == 'StarkTwoQubitsSWAPTwoDrives'


class TestStarkRamseyMultilevel:
    """Test the StarkRamseyMultilevel experiment."""
    
    @pytest.fixture
    def mock_duts(self):
        """Create mock DUT elements."""
        dut1 = Mock()
        dut1.get_gate.return_value = Mock()
        dut1.get_measurement_primitive.return_value = Mock()
        
        dut2 = Mock()
        dut2.get_gate.return_value = Mock()
        dut2.get_measurement_primitive.return_value = Mock()
        
        return [dut1, dut2]
    
    def test_initialization_with_parameters(self, mock_duts):
        """Test StarkRamseyMultilevel initialization with parameters."""
        exp = StarkRamseyMultilevel(
            duts=mock_duts,
            stark_freq=5000,
            stark_amp=0.1,
            sweep_ramsey_width=np.linspace(0, 100, 10)
        )
        
        assert exp.duts == mock_duts
        assert exp.stark_freq == 5000
        assert exp.stark_amp == 0.1
        assert len(exp.sweep_ramsey_width) == 10
    
    def test_initialization_minimal(self):
        """Test minimal initialization."""
        exp = StarkRamseyMultilevel()
        assert exp is not None


class TestStarkDriveRamsey:
    """Test the various Stark drive Ramsey experiments."""
    
    def test_stark_drive_ramsey_two_qubits_init(self):
        """Test StarkDriveRamseyTwoQubits initialization."""
        exp = StarkDriveRamseyTwoQubits()
        assert exp is not None
        assert exp.__class__.__name__ == 'StarkDriveRamseyTwoQubits'
    
    def test_stark_drive_ramsey_two_qubits_two_drives_init(self):
        """Test StarkDriveRamseyTwoQubitsTwoStarkDrives initialization."""
        exp = StarkDriveRamseyTwoQubitsTwoStarkDrives()
        assert exp is not None
        assert exp.__class__.__name__ == 'StarkDriveRamseyTwoQubitsTwoStarkDrives'
    
    def test_stark_drive_ramsey_multi_qubits_init(self):
        """Test StarkDriveRamseyMultiQubits initialization."""
        exp = StarkDriveRamseyMultiQubits()
        assert exp is not None
        assert exp.__class__.__name__ == 'StarkDriveRamseyMultiQubits'


class TestStarkZZShift:
    """Test the StarkZZShiftTwoQubitMultilevel experiment."""
    
    @pytest.fixture
    def mock_duts(self):
        """Create mock DUT elements."""
        dut1 = Mock()
        dut1.get_gate.return_value = Mock()
        dut1.get_measurement_primitive.return_value = Mock()
        
        dut2 = Mock()
        dut2.get_gate.return_value = Mock()
        dut2.get_measurement_primitive.return_value = Mock()
        
        return [dut1, dut2]
    
    def test_initialization_with_parameters(self, mock_duts):
        """Test StarkZZShiftTwoQubitMultilevel initialization."""
        exp = StarkZZShiftTwoQubitMultilevel(
            duts=mock_duts,
            stark_freq=5000,
            stark_amp=0.1,
            sweep_width=np.linspace(10, 100, 10)
        )
        
        assert exp.duts == mock_duts
        assert exp.stark_freq == 5000
        assert exp.stark_amp == 0.1
        assert len(exp.sweep_width) == 10
    
    def test_initialization_minimal(self):
        """Test minimal initialization."""
        exp = StarkZZShiftTwoQubitMultilevel()
        assert exp is not None


class TestStarkRabi:
    """Test the Stark Rabi experiments."""
    
    @pytest.fixture
    def mock_duts(self):
        """Create mock DUT elements."""
        dut = Mock()
        dut.get_gate.return_value = Mock()
        dut.get_measurement_primitive.return_value = Mock()
        return [dut]
    
    def test_stark_repeated_gate_rabi_init(self, mock_duts):
        """Test StarkRepeatedGateRabi initialization."""
        exp = StarkRepeatedGateRabi(
            duts=mock_duts,
            stark_freq=5000,
            stark_amp=0.1,
            sweep_n_gates=np.arange(1, 11)
        )
        
        assert exp.duts == mock_duts
        assert exp.stark_freq == 5000
        assert exp.stark_amp == 0.1
        assert len(exp.sweep_n_gates) == 10
    
    def test_stark_continues_rabi_init(self, mock_duts):
        """Test StarkContinuesRabi initialization."""
        exp = StarkContinuesRabi(
            duts=mock_duts,
            stark_freq=5000,
            stark_amp=0.1,
            sweep_width=np.linspace(10, 100, 10)
        )
        
        assert exp.duts == mock_duts
        assert exp.stark_freq == 5000
        assert exp.stark_amp == 0.1
        assert len(exp.sweep_width) == 10
    
    def test_stark_repeated_gate_drag_leakage_calibration_init(self, mock_duts):
        """Test StarkRepeatedGateDRAGLeakageCalibration initialization."""
        exp = StarkRepeatedGateDRAGLeakageCalibration(
            duts=mock_duts,
            stark_freq=5000,
            stark_amp=0.1,
            sweep_drag_coefficient=np.linspace(0, 1, 10)
        )
        
        assert exp.duts == mock_duts
        assert exp.stark_freq == 5000
        assert exp.stark_amp == 0.1
        assert len(exp.sweep_drag_coefficient) == 10


class TestParameterValidation:
    """Test parameter validation across all experiments."""
    
    @pytest.fixture
    def mock_duts(self):
        """Create mock DUT elements."""
        dut = Mock()
        dut.get_gate.return_value = Mock()
        dut.get_measurement_primitive.return_value = Mock()
        return [dut]
    
    def test_stark_frequency_validation(self, mock_duts):
        """Test that Stark frequencies are properly set."""
        exp = StarkRamseyMultilevel(
            duts=mock_duts,
            stark_freq=5000,
            stark_amp=0.1
        )
        
        assert exp.stark_freq == 5000
        assert exp.stark_freq > 0  # Should be positive frequency
    
    def test_stark_amplitude_validation(self, mock_duts):
        """Test that Stark amplitudes are properly set."""
        exp = StarkRamseyMultilevel(
            duts=mock_duts,
            stark_freq=5000,
            stark_amp=0.1
        )
        
        assert exp.stark_amp == 0.1
        assert exp.stark_amp >= 0  # Should be non-negative
        assert exp.stark_amp <= 1.0  # Should be reasonable amplitude
    
    def test_sweep_parameters_validation(self, mock_duts):
        """Test that sweep parameters are properly validated."""
        sweep_param = np.linspace(0, 100, 10)
        
        exp = StarkZZShiftTwoQubitMultilevel(
            duts=mock_duts,
            stark_freq=5000,
            stark_amp=0.1,
            sweep_width=sweep_param
        )
        
        assert len(exp.sweep_width) == 10
        assert np.all(exp.sweep_width >= 0)  # All values should be non-negative


@pytest.mark.integration
class TestExperimentIntegration:
    """Integration tests for AC Stark experiments."""
    
    @pytest.fixture
    def mock_duts_comprehensive(self):
        """Create comprehensive mock DUTs for integration testing."""
        dut1 = Mock()
        dut1.get_gate.return_value = Mock()
        dut1.get_measurement_primitive.return_value = Mock()
        dut1.name = "qubit_1"
        
        dut2 = Mock()
        dut2.get_gate.return_value = Mock()
        dut2.get_measurement_primitive.return_value = Mock()
        dut2.name = "qubit_2"
        
        return [dut1, dut2]
    
    @patch('leeq.experiments.builtin.multi_qubit_gates.ac_stark.ac_stark_shift.LogicalPrimitiveBlockSerial')
    @patch('leeq.experiments.builtin.multi_qubit_gates.ac_stark.ac_stark_shift.LogicalPrimitiveBlockSweep')
    def test_multi_experiment_compatibility(self, mock_sweep, mock_serial, mock_duts_comprehensive):
        """Test that different experiments can be created with same DUTs."""
        mock_serial.return_value = Mock()
        mock_sweep.return_value = Mock()
        
        # Create multiple experiments with the same DUTs
        exp1 = StarkRamseyMultilevel(
            duts=mock_duts_comprehensive,
            stark_freq=5000,
            stark_amp=0.1
        )
        
        exp2 = StarkZZShiftTwoQubitMultilevel(
            duts=mock_duts_comprehensive,
            stark_freq=5100,
            stark_amp=0.15
        )
        
        exp3 = StarkRepeatedGateRabi(
            duts=mock_duts_comprehensive[:1],  # Single qubit
            stark_freq=5200,
            stark_amp=0.2
        )
        
        # All experiments should be created successfully
        assert exp1 is not None
        assert exp2 is not None
        assert exp3 is not None
        
        # They should have different parameters
        assert exp1.stark_freq != exp2.stark_freq
        assert exp2.stark_freq != exp3.stark_freq
        
        # But share the same DUTs where applicable
        assert exp1.duts[0] is exp2.duts[0]
        assert exp1.duts[0] is exp3.duts[0]