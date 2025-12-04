"""
Unit tests for noise disable functionality in qubit spectroscopy experiments.

Tests the disable_noise parameter functionality in QubitSpectroscopyFrequency
and QubitSpectroscopyAmplitudeFrequency classes, validating:
- Clean deterministic output when disable_noise=True
- Existing noisy behavior when disable_noise=False  
- Parameter passing through constructor
- Reproducibility and determinism
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from leeq.experiments.builtin.basic.calibrations.qubit_spectroscopy import (
    QubitSpectroscopyFrequency, 
    QubitSpectroscopyAmplitudeFrequency
)
from leeq.core.elements.built_in.qudit_transmon import TransmonElement


@pytest.fixture
def test_qubit():
    """Create a test qubit element for spectroscopy tests."""
    configuration = {
        'lpb_collections': {
            'f01': {
                'type': 'SimpleDriveCollection',
                'freq': 5000.0,
                'channel': 2,
                'shape': 'blackman_drag',
                'amp': 0.1,
                'phase': 0.,
                'width': 0.025,
                'alpha': 425.0,
                'trunc': 1.2
            },
            'f12': {
                'type': 'SimpleDriveCollection',
                'freq': 4800.0,
                'channel': 2,
                'shape': 'blackman_drag',
                'amp': 0.05,
                'phase': 0.,
                'width': 0.025,
                'alpha': 425.0,
                'trunc': 1.2
            }
        },
        'measurement_primitives': {
            '0': {
                'type': 'SimpleDispersiveMeasurement',
                'freq': 9141.21,
                'channel': 1,
                'shape': 'square',
                'amp': 0.1,
                'phase': 0.,
                'width': 1,
                'trunc': 1.2,
                'distinguishable_states': [0, 1]
            },
            '1': {
                'type': 'SimpleDispersiveMeasurement',
                'freq': 9141.21,
                'channel': 1,
                'shape': 'square',
                'amp': 0.1,
                'phase': 0.,
                'width': 1,
                'trunc': 1.2,
                'distinguishable_states': [0, 1, 2]
            }
        }
    }
    
    return TransmonElement(
        name='test_qubit',
        parameters=configuration
    )


class TestQubitSpectroscopyNoiseDisable:
    """Test noise disable functionality across spectroscopy experiments."""
    
    def test_frequency_spectroscopy_clean_output(self, simulation_setup, test_qubit):
        """Test QubitSpectroscopyFrequency produces clean output when disable_noise=True."""
        # Run with noise disabled
        exp_clean = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=4900.0,
            stop=5100.0,
            step=10.0,
            num_avs=1000,
            disable_noise=True
        )
        
        # Verify results exist and have expected shape
        assert exp_clean.trace is not None
        assert exp_clean.result is not None
        assert 'Magnitude' in exp_clean.result
        assert 'Phase' in exp_clean.result
        assert len(exp_clean.trace) == 20  # (5100-4900)/10
        
        # Verify trace is complex
        assert np.iscomplexobj(exp_clean.trace)
        
        # Verify magnitude and phase are computed correctly
        expected_magnitude = np.absolute(exp_clean.trace)
        expected_phase = np.angle(exp_clean.trace)
        np.testing.assert_array_equal(exp_clean.result['Magnitude'], expected_magnitude)
        np.testing.assert_array_equal(exp_clean.result['Phase'], expected_phase)
    
    def test_frequency_spectroscopy_noisy_output(self, simulation_setup, test_qubit):
        """Test QubitSpectroscopyFrequency produces noisy output when disable_noise=False."""
        # Run with noise enabled (default)
        exp_noisy = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=4900.0,
            stop=5100.0,
            step=10.0,
            num_avs=1000,
            disable_noise=False
        )
        
        # Verify results exist and have expected shape
        assert exp_noisy.trace is not None
        assert exp_noisy.result is not None
        assert len(exp_noisy.trace) == 20
        assert np.iscomplexobj(exp_noisy.trace)
        
        # Run again with same parameters to verify randomness
        exp_noisy2 = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=4900.0,
            stop=5100.0,
            step=10.0,
            num_avs=1000,
            disable_noise=False
        )
        
        # Traces should be different due to noise (with high probability)
        assert not np.allclose(exp_noisy.trace, exp_noisy2.trace, atol=1e-10)
    
    def test_frequency_spectroscopy_deterministic_clean(self, simulation_setup, test_qubit):
        """Test QubitSpectroscopyFrequency produces identical clean output on multiple runs."""
        # Run multiple times with noise disabled
        exp1 = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=4950.0,
            stop=5050.0,
            step=20.0,
            num_avs=500,
            disable_noise=True
        )
        
        exp2 = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=4950.0,
            stop=5050.0,
            step=20.0,
            num_avs=500,
            disable_noise=True
        )
        
        # Results should be identical (deterministic)
        np.testing.assert_array_equal(exp1.trace, exp2.trace)
        np.testing.assert_array_equal(exp1.result['Magnitude'], exp2.result['Magnitude'])
        np.testing.assert_array_equal(exp1.result['Phase'], exp2.result['Phase'])
    
    def test_frequency_spectroscopy_noise_vs_clean_comparison(self, simulation_setup, test_qubit):
        """Test QubitSpectroscopyFrequency clean vs noisy data comparison."""
        # Same parameters for both experiments
        params = {
            'dut_qubit': test_qubit,
            'res_freq': 9141.21,
            'start': 4980.0,
            'stop': 5020.0,
            'step': 5.0,
            'num_avs': 1000,
            'mp_width': 0.5
        }
        
        # Run clean version
        exp_clean = QubitSpectroscopyFrequency(**params, disable_noise=True)
        
        # Run noisy version
        exp_noisy = QubitSpectroscopyFrequency(**params, disable_noise=False)
        
        # Both should have same shape
        assert exp_clean.trace.shape == exp_noisy.trace.shape
        
        # Clean version should have no added noise artifacts
        # The clean data should be "smoother" than noisy data in general
        clean_magnitude = exp_clean.result['Magnitude']
        noisy_magnitude = exp_noisy.result['Magnitude']
        
        # Check that both have reasonable values
        assert np.all(clean_magnitude >= 0)
        assert np.all(noisy_magnitude >= 0)
        assert np.all(np.isfinite(clean_magnitude))
        assert np.all(np.isfinite(noisy_magnitude))
    
    def test_amplitude_frequency_spectroscopy_clean_output(self, simulation_setup, test_qubit):
        """Test QubitSpectroscopyAmplitudeFrequency produces clean 2D output when disable_noise=True."""
        # Run 2D spectroscopy with noise disabled
        exp_clean = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=test_qubit,
            start=4900.0,
            stop=5100.0,
            step=50.0,
            qubit_amp_start=0.1,
            qubit_amp_stop=0.3,
            qubit_amp_step=0.1,
            num_avs=500,
            disable_noise=True
        )
        
        # Verify 2D results
        import numpy as np
        expected_freq_points = len(np.arange(4900.0, 5100.0, 50.0))  # 4 points
        expected_amp_points = len(np.arange(0.1, 0.3, 0.1))  # 2 points
        
        assert exp_clean.trace is not None
        assert exp_clean.trace.shape == (expected_amp_points, expected_freq_points)  # (2, 4)
        assert np.iscomplexobj(exp_clean.trace)
        
        # Verify results dictionary
        assert exp_clean.result['Magnitude'].shape == (expected_amp_points, expected_freq_points)
        assert exp_clean.result['Phase'].shape == (expected_amp_points, expected_freq_points)
    
    def test_amplitude_frequency_spectroscopy_noisy_output(self, simulation_setup, test_qubit):
        """Test QubitSpectroscopyAmplitudeFrequency produces noisy 2D output when disable_noise=False."""
        # Run with noise enabled
        exp_noisy = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=test_qubit,
            start=4900.0,
            stop=5000.0,
            step=50.0,
            qubit_amp_start=0.1,
            qubit_amp_stop=0.2,
            qubit_amp_step=0.1,
            num_avs=500,
            disable_noise=False
        )
        
        # Verify 2D results with noise
        expected_shape = (1, 2)  # 1 amplitude point, 2 frequency points
        assert exp_noisy.trace.shape == expected_shape
        assert np.iscomplexobj(exp_noisy.trace)
        
        # Run again to verify randomness
        exp_noisy2 = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=test_qubit,
            start=4900.0,
            stop=5000.0,
            step=50.0,
            qubit_amp_start=0.1,
            qubit_amp_stop=0.2,
            qubit_amp_step=0.1,
            num_avs=500,
            disable_noise=False
        )
        
        # Should be different due to noise
        assert not np.allclose(exp_noisy.trace, exp_noisy2.trace, atol=1e-10)
    
    def test_amplitude_frequency_spectroscopy_deterministic_clean(self, simulation_setup, test_qubit):
        """Test QubitSpectroscopyAmplitudeFrequency produces identical clean 2D output on multiple runs."""
        params = {
            'dut_qubit': test_qubit,
            'start': 4950.0,
            'stop': 5050.0,
            'step': 50.0,
            'qubit_amp_start': 0.1,
            'qubit_amp_stop': 0.2,
            'qubit_amp_step': 0.1,
            'num_avs': 300,
            'disable_noise': True
        }
        
        # Run multiple times
        exp1 = QubitSpectroscopyAmplitudeFrequency(**params)
        exp2 = QubitSpectroscopyAmplitudeFrequency(**params)
        
        # Should be identical
        np.testing.assert_array_equal(exp1.trace, exp2.trace)
        np.testing.assert_array_equal(exp1.result['Magnitude'], exp2.result['Magnitude'])
        np.testing.assert_array_equal(exp1.result['Phase'], exp2.result['Phase'])
    
    def test_amplitude_frequency_noise_vs_clean_comparison(self, simulation_setup, test_qubit):
        """Test QubitSpectroscopyAmplitudeFrequency clean vs noisy 2D data comparison."""
        params = {
            'dut_qubit': test_qubit,
            'start': 4900.0,
            'stop': 5000.0,
            'step': 100.0,
            'qubit_amp_start': 0.05,
            'qubit_amp_stop': 0.15,
            'qubit_amp_step': 0.05,
            'num_avs': 400
        }
        
        # Clean version
        exp_clean = QubitSpectroscopyAmplitudeFrequency(**params, disable_noise=True)
        
        # Noisy version
        exp_noisy = QubitSpectroscopyAmplitudeFrequency(**params, disable_noise=False)
        
        # Same shapes
        assert exp_clean.trace.shape == exp_noisy.trace.shape
        assert exp_clean.result['Magnitude'].shape == exp_noisy.result['Magnitude'].shape
        
        # Both should have reasonable 2D data
        clean_mag = exp_clean.result['Magnitude']
        noisy_mag = exp_noisy.result['Magnitude']
        
        assert np.all(clean_mag >= 0)
        assert np.all(noisy_mag >= 0)
        assert clean_mag.ndim == 2
        assert noisy_mag.ndim == 2
    
    def test_backward_compatibility_default_behavior(self, simulation_setup, test_qubit):
        """Test that default behavior (no disable_noise parameter) maintains existing noise behavior."""
        # Default behavior (no disable_noise specified) should be equivalent to disable_noise=False
        exp_default = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=4950.0,
            stop=5050.0,
            step=25.0,
            num_avs=800
            # No disable_noise parameter - should default to False
        )
        
        exp_explicit_false = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=4950.0,
            stop=5050.0,
            step=25.0,
            num_avs=800,
            disable_noise=False
        )
        
        # Both should have noise (results will be different but structure same)
        assert exp_default.trace.shape == exp_explicit_false.trace.shape
        assert exp_default.result['Magnitude'].shape == exp_explicit_false.result['Magnitude'].shape
        
        # Verify both are complex and have reasonable values
        assert np.iscomplexobj(exp_default.trace)
        assert np.iscomplexobj(exp_explicit_false.trace)
        assert np.all(np.isfinite(exp_default.trace))
        assert np.all(np.isfinite(exp_explicit_false.trace))
    
    def test_parameter_passing_through_constructor(self, simulation_setup, test_qubit):
        """Test that disable_noise parameter is correctly passed through LeeQ constructor framework."""
        # Test that the parameter is accepted and processed without errors
        
        # Test with disable_noise=True
        exp_clean = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=4950.0,
            stop=5050.0,
            step=50.0,
            num_avs=200,
            disable_noise=True,
            mp_width=0.8,
            rep_rate=0.001
        )
        
        # Should run successfully and produce results
        assert exp_clean.trace is not None
        assert len(exp_clean.trace) == 2  # 2 frequency points
        
        # Test with disable_noise=False
        exp_noisy = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=4950.0,
            stop=5050.0,
            step=50.0,
            num_avs=200,
            disable_noise=False,
            mp_width=0.8,
            rep_rate=0.001
        )
        
        # Should also run successfully
        assert exp_noisy.trace is not None
        assert len(exp_noisy.trace) == 2
    
    def test_edge_cases_and_error_conditions(self, simulation_setup, test_qubit):
        """Test edge cases and error conditions for noise disable functionality."""
        
        # Test with very small sweep (single point)
        exp_single = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=5000.0,
            stop=5010.0,
            step=10.0,
            num_avs=100,
            disable_noise=True
        )
        
        assert len(exp_single.trace) == 1
        assert np.iscomplexobj(exp_single.trace)
        
        # Test with minimal parameters for 2D spectroscopy
        exp_2d_minimal = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=test_qubit,
            start=5000.0,
            stop=5010.0,
            step=10.0,
            qubit_amp_start=0.1,
            qubit_amp_stop=0.11,
            qubit_amp_step=0.01,
            num_avs=50,
            disable_noise=True
        )
        
        assert exp_2d_minimal.trace.shape == (1, 1)
        assert np.iscomplexobj(exp_2d_minimal.trace)
    
    def test_noise_scaling_independence(self, simulation_setup, test_qubit):
        """Test that clean data is independent of noise scaling parameters."""
        # Clean data should be the same regardless of num_avs when disable_noise=True
        
        exp_low_avs = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=4950.0,
            stop=5050.0,
            step=25.0,
            num_avs=100,  # Low averages
            disable_noise=True
        )
        
        exp_high_avs = QubitSpectroscopyFrequency(
            dut_qubit=test_qubit,
            res_freq=9141.21,
            start=4950.0,
            stop=5050.0,
            step=25.0,
            num_avs=10000,  # High averages
            disable_noise=True
        )
        
        # Clean data should be identical regardless of num_avs
        np.testing.assert_array_equal(exp_low_avs.trace, exp_high_avs.trace)
    
    def test_complex_data_integrity(self, simulation_setup, test_qubit):
        """Test that complex data integrity is maintained in both clean and noisy modes."""
        
        params = {
            'dut_qubit': test_qubit,
            'res_freq': 9141.21,
            'start': 4900.0,
            'stop': 5100.0,
            'step': 25.0,
            'num_avs': 500
        }
        
        # Test clean data
        exp_clean = QubitSpectroscopyFrequency(**params, disable_noise=True)
        
        # Verify complex data properties
        assert np.iscomplexobj(exp_clean.trace)
        assert np.all(np.isfinite(exp_clean.trace))
        assert np.all(np.isfinite(exp_clean.result['Magnitude']))
        assert np.all(np.isfinite(exp_clean.result['Phase']))
        
        # Test noisy data
        exp_noisy = QubitSpectroscopyFrequency(**params, disable_noise=False)
        
        # Verify complex data properties for noisy data too
        assert np.iscomplexobj(exp_noisy.trace)
        assert np.all(np.isfinite(exp_noisy.trace))
        assert np.all(np.isfinite(exp_noisy.result['Magnitude']))
        assert np.all(np.isfinite(exp_noisy.result['Phase']))
        
        # Magnitudes should be non-negative in both cases
        assert np.all(exp_clean.result['Magnitude'] >= 0)
        assert np.all(exp_noisy.result['Magnitude'] >= 0)