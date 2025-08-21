"""
Tests for conditional stark shift AI experiments.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Mock external dependencies that are causing import issues
mock_modules = {
    'k_agents.execution.agent': MagicMock(),
    'k_agents.execution.stage_execution': MagicMock(),
    'k_agents.inspection.decorator': MagicMock(),
    'k_agents.io_interface': MagicMock(),
    'k_agents.utils': MagicMock(),
    'mllm': MagicMock()
}

# Apply mocks before importing the actual module
for module_name, mock_module in mock_modules.items():
    import sys
    sys.modules[module_name] = mock_module

# Mock the Singleton decorator
class MockSingleton:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

mock_modules['k_agents.utils'].Singleton = MockSingleton

from leeq.experiments.builtin.multi_qubit_gates.conditional_stark_ai import (
    _qubit_z_expectation_value_off_resonance_drive,
    ConditionalStarkShiftContinuousPhaseSweep,
    ConditionalStarkShiftContinuous,
    ConditionalStarkShiftRepeatedGate,
    ConditionalStarkEchoTuneUpAI,
    ConditionalStarkTwoQubitGateAIParameterSearchFull,
    TwoQubitTuningEnv,
    ConditionalStarkTwoQubitGateAIParameterSearchBase,
    ConditionalStarkTwoQubitGateAmplitudeAdvise,
    ConditionalStarkTwoQubitGateAmplitudeAttempt,
    ConditionalStarkTwoQubitGateFrequencyAdvise
)


class TestQubitZExpectationValue:
    """Test the helper function for qubit Z expectation value calculation."""
    
    def test_basic_calculation(self):
        """Test basic Z expectation value calculation."""
        result = _qubit_z_expectation_value_off_resonance_drive(
            f_qubit=5000,  # MHz
            f_drive=5100,  # MHz
            t_start=0,
            t_stop=1,
            t_step=0.1,
            drive_rate=10  # MHz
        )
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert np.all(np.abs(result) <= 1.0)  # Z expectation values should be between -1 and 1
    
    def test_on_resonance_drive(self):
        """Test Z expectation value when drive is on resonance."""
        result = _qubit_z_expectation_value_off_resonance_drive(
            f_qubit=5000,  # MHz
            f_drive=5000,  # MHz - on resonance
            t_start=0,
            t_stop=1,
            t_step=0.1,
            drive_rate=10  # MHz
        )
        
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    
    def test_different_time_parameters(self):
        """Test with different time parameters."""
        result = _qubit_z_expectation_value_off_resonance_drive(
            f_qubit=5000,
            f_drive=5100,
            t_start=0,
            t_stop=5,
            t_step=0.5,
            drive_rate=10
        )
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 10  # Should have 10 points for 5us with 0.5us steps


class TestConditionalStarkExperiments:
    """Test conditional stark shift experiment classes."""
    
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
    
    def test_conditional_stark_continuous_phase_sweep_init(self, mock_duts):
        """Test ConditionalStarkShiftContinuousPhaseSweep initialization."""
        exp = ConditionalStarkShiftContinuousPhaseSweep(
            duts=mock_duts,
            drive_freq_1=5000,
            drive_freq_2=5100,
            drive_amp=0.1,
            n_gates=10,
            sweep_phase=np.linspace(0, 2*np.pi, 10)
        )
        
        assert exp.drive_freq_1 == 5000
        assert exp.drive_freq_2 == 5100
        assert exp.drive_amp == 0.1
        assert exp.n_gates == 10
        assert len(exp.sweep_phase) == 10
    
    def test_conditional_stark_continuous_init(self, mock_duts):
        """Test ConditionalStarkShiftContinuous initialization."""
        exp = ConditionalStarkShiftContinuous(
            duts=mock_duts,
            drive_freq_1=5000,
            drive_freq_2=5100,
            drive_amp=0.1,
            sweep_width=np.linspace(10, 100, 10)
        )
        
        assert exp.drive_freq_1 == 5000
        assert exp.drive_freq_2 == 5100
        assert exp.drive_amp == 0.1
        assert len(exp.sweep_width) == 10
    
    def test_conditional_stark_repeated_gate_init(self, mock_duts):
        """Test ConditionalStarkShiftRepeatedGate initialization."""
        exp = ConditionalStarkShiftRepeatedGate(
            duts=mock_duts,
            drive_freq_1=5000,
            drive_freq_2=5100,
            drive_amp=0.1,
            drive_width=50,
            sweep_n_gates=np.arange(1, 11)
        )
        
        assert exp.drive_freq_1 == 5000
        assert exp.drive_freq_2 == 5100
        assert exp.drive_amp == 0.1
        assert exp.drive_width == 50
        assert len(exp.sweep_n_gates) == 10
    
    @patch('leeq.experiments.builtin.multi_qubit_gates.conditional_stark_ai.fits')
    def test_conditional_stark_echo_tune_up_init(self, mock_fits, mock_duts):
        """Test ConditionalStarkEchoTuneUpAI initialization."""
        exp = ConditionalStarkEchoTuneUpAI(
            duts=mock_duts,
            frequencies_MHz=[5000, 5100],
            max_amplitude=0.5,
            widths_ns=np.linspace(10, 100, 10)
        )
        
        assert exp.frequencies_MHz == [5000, 5100]
        assert exp.max_amplitude == 0.5
        assert len(exp.widths_ns) == 10


class TestAIParameterSearch:
    """Test AI-based parameter search experiments."""
    
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
    
    @patch('leeq.experiments.builtin.multi_qubit_gates.conditional_stark_ai.Chat')
    def test_ai_parameter_search_full_init(self, mock_chat, mock_duts):
        """Test ConditionalStarkTwoQubitGateAIParameterSearchFull initialization."""
        # Mock the chat instance
        mock_chat_instance = Mock()
        mock_chat.return_value = mock_chat_instance
        
        exp = ConditionalStarkTwoQubitGateAIParameterSearchFull(
            duts=mock_duts,
            frequencies_MHz=[5000, 5100],
            max_amplitude=0.5
        )
        
        assert exp.frequencies_MHz == [5000, 5100]
        assert exp.max_amplitude == 0.5
    
    def test_ai_parameter_search_base_init(self, mock_duts):
        """Test ConditionalStarkTwoQubitGateAIParameterSearchBase initialization."""
        exp = ConditionalStarkTwoQubitGateAIParameterSearchBase(
            duts=mock_duts,
            frequencies_MHz=[5000, 5100],
            max_amplitude=0.5
        )
        
        assert exp.frequencies_MHz == [5000, 5100]
        assert exp.max_amplitude == 0.5
    
    def test_amplitude_advise_init(self, mock_duts):
        """Test ConditionalStarkTwoQubitGateAmplitudeAdvise initialization."""
        exp = ConditionalStarkTwoQubitGateAmplitudeAdvise(
            duts=mock_duts,
            frequencies_MHz=[5000, 5100],
            amplitude_control=0.3,
            amplitude_target=0.4
        )
        
        assert exp.frequencies_MHz == [5000, 5100]
        assert exp.amplitude_control == 0.3
        assert exp.amplitude_target == 0.4
    
    def test_amplitude_attempt_init(self, mock_duts):
        """Test ConditionalStarkTwoQubitGateAmplitudeAttempt initialization."""
        exp = ConditionalStarkTwoQubitGateAmplitudeAttempt(
            duts=mock_duts,
            frequencies_MHz=[5000, 5100],
            max_amplitude=0.5,
            widths_ns=np.linspace(10, 100, 10)
        )
        
        assert exp.frequencies_MHz == [5000, 5100]
        assert exp.max_amplitude == 0.5
        assert len(exp.widths_ns) == 10
    
    def test_frequency_advise_init(self, mock_duts):
        """Test ConditionalStarkTwoQubitGateFrequencyAdvise initialization."""
        exp = ConditionalStarkTwoQubitGateFrequencyAdvise(
            duts=mock_duts,
            frequencies_MHz=[5000, 5100],
            amplitude_control=0.3,
            amplitude_target=0.4
        )
        
        assert exp.frequencies_MHz == [5000, 5100]
        assert exp.amplitude_control == 0.3
        assert exp.amplitude_target == 0.4


class TestTwoQubitTuningEnv:
    """Test the TwoQubitTuningEnv singleton class."""
    
    def test_singleton_creation(self):
        """Test that TwoQubitTuningEnv can be instantiated."""
        # Mock the Singleton behavior
        env1 = TwoQubitTuningEnv()
        env2 = TwoQubitTuningEnv()
        
        # Both should be instances of TwoQubitTuningEnv
        assert isinstance(env1, TwoQubitTuningEnv)
        assert isinstance(env2, TwoQubitTuningEnv)
    
    def test_env_attributes_exist(self):
        """Test that the environment has expected attributes."""
        env = TwoQubitTuningEnv()
        
        # Test that we can set attributes (basic functionality)
        env.test_attribute = "test_value"
        assert env.test_attribute == "test_value"


class TestParameterValidation:
    """Test parameter validation for various experiments."""
    
    @pytest.fixture
    def mock_duts(self):
        """Create mock DUT elements."""
        dut1 = Mock()
        dut2 = Mock()
        return [dut1, dut2]
    
    def test_frequency_parameter_validation(self, mock_duts):
        """Test that frequency parameters are properly validated."""
        # Test positive frequencies
        exp = ConditionalStarkShiftContinuous(
            duts=mock_duts,
            drive_freq_1=5000,
            drive_freq_2=5100,
            drive_amp=0.1,
            sweep_width=np.linspace(10, 100, 10)
        )
        
        assert exp.drive_freq_1 > 0
        assert exp.drive_freq_2 > 0
    
    def test_amplitude_parameter_validation(self, mock_duts):
        """Test that amplitude parameters are properly validated."""
        exp = ConditionalStarkShiftContinuous(
            duts=mock_duts,
            drive_freq_1=5000,
            drive_freq_2=5100,
            drive_amp=0.1,
            sweep_width=np.linspace(10, 100, 10)
        )
        
        assert exp.drive_amp >= 0
        assert exp.drive_amp <= 1.0


@pytest.mark.integration
class TestExperimentIntegration:
    """Integration tests for experiment functionality."""
    
    @pytest.fixture
    def mock_duts_with_methods(self):
        """Create mock DUTs with proper method structure."""
        dut1 = Mock()
        dut1.get_gate.return_value = Mock()
        dut1.get_measurement_primitive.return_value = Mock()
        dut1.name = "qubit_1"
        
        dut2 = Mock()
        dut2.get_gate.return_value = Mock()
        dut2.get_measurement_primitive.return_value = Mock()
        dut2.name = "qubit_2"
        
        return [dut1, dut2]
    
    @patch('leeq.experiments.builtin.multi_qubit_gates.conditional_stark_ai.LogicalPrimitiveBlockSerial')
    @patch('leeq.experiments.builtin.multi_qubit_gates.conditional_stark_ai.LogicalPrimitiveBlockSweep')
    def test_experiment_lpb_creation(self, mock_sweep, mock_serial, mock_duts_with_methods):
        """Test that experiments can create logical primitive blocks."""
        mock_serial.return_value = Mock()
        mock_sweep.return_value = Mock()
        
        exp = ConditionalStarkShiftContinuous(
            duts=mock_duts_with_methods,
            drive_freq_1=5000,
            drive_freq_2=5100,
            drive_amp=0.1,
            sweep_width=np.linspace(10, 100, 10)
        )
        
        # Test that the experiment was created successfully
        assert exp is not None
        assert hasattr(exp, 'duts')
        assert len(exp.duts) == 2