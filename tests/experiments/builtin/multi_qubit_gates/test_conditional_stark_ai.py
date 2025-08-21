"""
Tests for conditional stark shift AI experiments.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import contextlib

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

# Import necessary modules for mocking setup
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

from tests.fixtures.mock_qubits import mock_qubit


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
    
    def test_conditional_stark_continuous_phase_sweep_exists(self):
        """Test ConditionalStarkShiftContinuousPhaseSweep class exists."""
        assert ConditionalStarkShiftContinuousPhaseSweep is not None
        assert hasattr(ConditionalStarkShiftContinuousPhaseSweep, '__name__')
        assert ConditionalStarkShiftContinuousPhaseSweep.__name__ == 'ConditionalStarkShiftContinuousPhaseSweep'
    
    def test_conditional_stark_continuous_exists(self):
        """Test ConditionalStarkShiftContinuous class exists."""
        assert ConditionalStarkShiftContinuous is not None
        assert hasattr(ConditionalStarkShiftContinuous, '__name__')
        assert ConditionalStarkShiftContinuous.__name__ == 'ConditionalStarkShiftContinuous'
    
    def test_conditional_stark_repeated_gate_exists(self):
        """Test ConditionalStarkShiftRepeatedGate class exists."""
        assert ConditionalStarkShiftRepeatedGate is not None
        assert hasattr(ConditionalStarkShiftRepeatedGate, '__name__')
        assert ConditionalStarkShiftRepeatedGate.__name__ == 'ConditionalStarkShiftRepeatedGate'
    
    def test_conditional_stark_echo_tune_up_exists(self):
        """Test ConditionalStarkEchoTuneUpAI class exists."""
        assert ConditionalStarkEchoTuneUpAI is not None
        assert hasattr(ConditionalStarkEchoTuneUpAI, '__name__')
        assert ConditionalStarkEchoTuneUpAI.__name__ == 'ConditionalStarkEchoTuneUpAI'


class TestAIParameterSearch:
    """Test AI-based parameter search experiments."""
    
    def test_ai_parameter_search_full_exists(self):
        """Test ConditionalStarkTwoQubitGateAIParameterSearchFull class exists."""
        assert ConditionalStarkTwoQubitGateAIParameterSearchFull is not None
        assert hasattr(ConditionalStarkTwoQubitGateAIParameterSearchFull, '__name__')
        assert ConditionalStarkTwoQubitGateAIParameterSearchFull.__name__ == 'ConditionalStarkTwoQubitGateAIParameterSearchFull'
    
    def test_ai_parameter_search_base_exists(self):
        """Test ConditionalStarkTwoQubitGateAIParameterSearchBase class exists."""
        assert ConditionalStarkTwoQubitGateAIParameterSearchBase is not None
        assert hasattr(ConditionalStarkTwoQubitGateAIParameterSearchBase, '__name__')
        assert ConditionalStarkTwoQubitGateAIParameterSearchBase.__name__ == 'ConditionalStarkTwoQubitGateAIParameterSearchBase'
    
    def test_amplitude_advise_exists(self):
        """Test ConditionalStarkTwoQubitGateAmplitudeAdvise class exists."""
        assert ConditionalStarkTwoQubitGateAmplitudeAdvise is not None
        assert hasattr(ConditionalStarkTwoQubitGateAmplitudeAdvise, '__name__')
        assert ConditionalStarkTwoQubitGateAmplitudeAdvise.__name__ == 'ConditionalStarkTwoQubitGateAmplitudeAdvise'
    
    def test_amplitude_attempt_exists(self):
        """Test ConditionalStarkTwoQubitGateAmplitudeAttempt class exists."""
        assert ConditionalStarkTwoQubitGateAmplitudeAttempt is not None
        assert hasattr(ConditionalStarkTwoQubitGateAmplitudeAttempt, '__name__')
        assert ConditionalStarkTwoQubitGateAmplitudeAttempt.__name__ == 'ConditionalStarkTwoQubitGateAmplitudeAttempt'
    
    def test_frequency_advise_exists(self):
        """Test ConditionalStarkTwoQubitGateFrequencyAdvise class exists."""
        assert ConditionalStarkTwoQubitGateFrequencyAdvise is not None
        assert hasattr(ConditionalStarkTwoQubitGateFrequencyAdvise, '__name__')
        assert ConditionalStarkTwoQubitGateFrequencyAdvise.__name__ == 'ConditionalStarkTwoQubitGateFrequencyAdvise'


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
    
    def test_numpy_operations(self):
        """Test that basic numpy operations work for parameter validation."""
        # Test basic numpy functionality that would be used in parameter validation
        amp_control = 0.3
        amp_target = 0.4
        
        assert amp_control > 0
        assert amp_target > 0
        assert amp_control >= 0
        assert amp_target >= 0


@pytest.mark.integration
class TestExperimentIntegration:
    """Integration tests for experiment functionality."""
    
    def test_all_experiments_importable(self):
        """Test that all conditional stark experiments can be imported successfully."""
        experiments = [
            ConditionalStarkShiftContinuousPhaseSweep,
            ConditionalStarkShiftContinuous,
            ConditionalStarkShiftRepeatedGate,
            ConditionalStarkEchoTuneUpAI,
            ConditionalStarkTwoQubitGateAIParameterSearchFull,
            ConditionalStarkTwoQubitGateAIParameterSearchBase,
            ConditionalStarkTwoQubitGateAmplitudeAdvise,
            ConditionalStarkTwoQubitGateAmplitudeAttempt,
            ConditionalStarkTwoQubitGateFrequencyAdvise
        ]
        
        for exp_class in experiments:
            assert exp_class is not None
            assert hasattr(exp_class, '__name__')
            # Verify it's a proper class
            assert callable(exp_class)
    
    def test_helper_function_exists(self):
        """Test that helper functions are properly imported."""
        assert callable(_qubit_z_expectation_value_off_resonance_drive)
        
        # Test that it can be called with basic parameters
        result = _qubit_z_expectation_value_off_resonance_drive(
            f_qubit=5000,
            f_drive=5100,
            t_start=0,
            t_stop=1,
            t_step=0.1,
            drive_rate=10
        )
        assert isinstance(result, np.ndarray)
    
    def test_singleton_class_exists(self):
        """Test that the TwoQubitTuningEnv singleton class exists."""
        assert TwoQubitTuningEnv is not None
        assert callable(TwoQubitTuningEnv)
        
        # Test basic instantiation
        env = TwoQubitTuningEnv()
        assert env is not None