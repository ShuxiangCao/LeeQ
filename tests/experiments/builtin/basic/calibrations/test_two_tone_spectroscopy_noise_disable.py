"""
Unit tests for noise disable functionality in TwoToneQubitSpectroscopy experiment.

Tests the disable_noise parameter functionality in TwoToneQubitSpectroscopy class, validating:
- Clean deterministic output when disable_noise=True
- Existing noisy behavior when disable_noise=False
- Parameter passing through constructor
- 2D spectroscopy data integrity
"""

import pytest
import numpy as np

from leeq.experiments.builtin.basic.calibrations.two_tone_spectroscopy import TwoToneQubitSpectroscopy
from leeq.core.elements.built_in.qudit_transmon import TransmonElement


@pytest.fixture
def test_qubit():
    """Create a test qubit element for two-tone spectroscopy tests."""
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
                'freq': 9000.0,
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
                'freq': 9000.0,
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
        name='test_qubit_two_tone',
        parameters=configuration
    )


class TestTwoToneSpectroscopyNoiseDisable:
    """Test noise disable functionality for TwoToneQubitSpectroscopy."""
    
    def test_two_tone_spectroscopy_clean_output(self, simulation_setup, test_qubit):
        """Test TwoToneQubitSpectroscopy produces clean 2D output when disable_noise=True."""
        # Run two-tone spectroscopy with noise disabled
        exp_clean = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4950.0,
            tone1_stop=5050.0,
            tone1_step=50.0,
            tone1_amp=0.1,
            tone2_start=4750.0,
            tone2_stop=4850.0,
            tone2_step=50.0,
            tone2_amp=0.05,
            num_avs=500,
            disable_noise=True
        )
        
        # Verify 2D results exist
        assert exp_clean.trace is not None
        assert exp_clean.result is not None
        assert 'Magnitude' in exp_clean.result
        assert 'Phase' in exp_clean.result
        
        # Verify correct 2D shape: (tone1 points, tone2 points)
        expected_tone1_points = len(np.arange(4950.0, 5050.0 + 50.0/2, 50.0))  # 3 points
        expected_tone2_points = len(np.arange(4750.0, 4850.0 + 50.0/2, 50.0))  # 3 points
        assert exp_clean.trace.shape == (expected_tone1_points, expected_tone2_points)
        
        # Verify complex data
        assert np.iscomplexobj(exp_clean.trace)
        
        # Verify magnitude and phase computations
        expected_magnitude = np.abs(exp_clean.trace)
        expected_phase = np.angle(exp_clean.trace)
        np.testing.assert_array_equal(exp_clean.result['Magnitude'], expected_magnitude)
        np.testing.assert_array_equal(exp_clean.result['Phase'], expected_phase)
        
        # Verify all values are finite
        assert np.all(np.isfinite(exp_clean.trace))
        assert np.all(np.isfinite(exp_clean.result['Magnitude']))
        assert np.all(np.isfinite(exp_clean.result['Phase']))
    
    def test_two_tone_spectroscopy_noisy_output(self, simulation_setup, test_qubit):
        """Test TwoToneQubitSpectroscopy produces noisy 2D output when disable_noise=False."""
        # Run with noise enabled (default behavior)
        exp_noisy = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4900.0,
            tone1_stop=5000.0,
            tone1_step=50.0,
            tone1_amp=0.1,
            tone2_start=4700.0,
            tone2_stop=4800.0,
            tone2_step=50.0,
            tone2_amp=0.05,
            num_avs=1000,
            disable_noise=False
        )
        
        # Verify 2D results exist and have correct shape
        expected_shape = (3, 3)  # 3 x 3 grid
        assert exp_noisy.trace.shape == expected_shape
        assert np.iscomplexobj(exp_noisy.trace)
        assert exp_noisy.result['Magnitude'].shape == expected_shape
        assert exp_noisy.result['Phase'].shape == expected_shape
        
        # Run again with same parameters to verify noise randomness
        exp_noisy2 = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4900.0,
            tone1_stop=5000.0,
            tone1_step=50.0,
            tone1_amp=0.1,
            tone2_start=4700.0,
            tone2_stop=4800.0,
            tone2_step=50.0,
            tone2_amp=0.05,
            num_avs=1000,
            disable_noise=False
        )
        
        # Results should be different due to noise (with high probability)
        assert not np.allclose(exp_noisy.trace, exp_noisy2.trace, atol=1e-10)
    
    def test_two_tone_spectroscopy_deterministic_clean(self, simulation_setup, test_qubit):
        """Test TwoToneQubitSpectroscopy produces identical clean output on multiple runs."""
        params = {
            'dut_qubit': test_qubit,
            'tone1_start': 4950.0,
            'tone1_stop': 5050.0,
            'tone1_step': 100.0,
            'tone1_amp': 0.08,
            'tone2_start': 4750.0,
            'tone2_stop': 4850.0,
            'tone2_step': 100.0,
            'tone2_amp': 0.04,
            'num_avs': 800,
            'disable_noise': True
        }
        
        # Run multiple times
        exp1 = TwoToneQubitSpectroscopy(**params)
        exp2 = TwoToneQubitSpectroscopy(**params)
        
        # Results should be identical (deterministic)
        np.testing.assert_array_equal(exp1.trace, exp2.trace)
        np.testing.assert_array_equal(exp1.result['Magnitude'], exp2.result['Magnitude'])
        np.testing.assert_array_equal(exp1.result['Phase'], exp2.result['Phase'])
    
    def test_two_tone_spectroscopy_noise_vs_clean_comparison(self, simulation_setup, test_qubit):
        """Test TwoToneQubitSpectroscopy clean vs noisy 2D data comparison."""
        params = {
            'dut_qubit': test_qubit,
            'tone1_start': 4950.0,
            'tone1_stop': 5050.0,
            'tone1_step': 50.0,
            'tone1_amp': 0.1,
            'tone2_start': 4750.0,
            'tone2_stop': 4850.0,
            'tone2_step': 50.0,
            'tone2_amp': 0.05,
            'num_avs': 600
        }
        
        # Clean version
        exp_clean = TwoToneQubitSpectroscopy(**params, disable_noise=True)
        
        # Noisy version
        exp_noisy = TwoToneQubitSpectroscopy(**params, disable_noise=False)
        
        # Both should have same 2D shape
        assert exp_clean.trace.shape == exp_noisy.trace.shape
        assert exp_clean.result['Magnitude'].shape == exp_noisy.result['Magnitude'].shape
        
        # Both should have valid 2D complex data
        clean_mag = exp_clean.result['Magnitude']
        noisy_mag = exp_noisy.result['Magnitude']
        
        assert np.all(clean_mag >= 0)
        assert np.all(noisy_mag >= 0)
        assert clean_mag.ndim == 2
        assert noisy_mag.ndim == 2
        assert np.all(np.isfinite(clean_mag))
        assert np.all(np.isfinite(noisy_mag))
    
    def test_two_tone_same_channel_functionality(self, simulation_setup, test_qubit):
        """Test two-tone spectroscopy with same_channel=True option."""
        # Test same channel mode with clean data
        exp_same_channel = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4950.0,
            tone1_stop=5050.0,
            tone1_step=100.0,
            tone1_amp=0.1,
            tone2_start=4750.0,
            tone2_stop=4850.0,
            tone2_step=100.0,
            tone2_amp=0.05,
            same_channel=True,  # Both tones on same channel
            num_avs=500,
            disable_noise=True
        )
        
        # Should produce valid 2D results
        assert exp_same_channel.trace is not None
        assert exp_same_channel.trace.shape == (2, 2)  # 2x2 grid
        assert np.iscomplexobj(exp_same_channel.trace)
        assert np.all(np.isfinite(exp_same_channel.trace))
        
        # Test different channel mode for comparison
        exp_diff_channel = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4950.0,
            tone1_stop=5050.0,
            tone1_step=100.0,
            tone1_amp=0.1,
            tone2_start=4750.0,
            tone2_stop=4850.0,
            tone2_step=100.0,
            tone2_amp=0.05,
            same_channel=False,  # Different channels
            num_avs=500,
            disable_noise=True
        )
        
        # Both should have same shape and valid data
        assert exp_same_channel.trace.shape == exp_diff_channel.trace.shape
        assert np.all(np.isfinite(exp_diff_channel.trace))
    
    def test_two_tone_parameter_passing_through_constructor(self, simulation_setup, test_qubit):
        """Test that disable_noise parameter is correctly passed through LeeQ constructor framework."""
        # Test with all parameters including disable_noise=True
        exp_clean = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4950.0,
            tone1_stop=5050.0,
            tone1_step=50.0,
            tone1_amp=0.12,
            tone2_start=4750.0,
            tone2_stop=4850.0,
            tone2_step=50.0,
            tone2_amp=0.06,
            same_channel=False,
            num_avs=400,
            rep_rate=0.001,
            mp_width=0.8,
            set_qubit='f01',
            disable_noise=True  # Key parameter
        )
        
        # Should run successfully and produce clean results
        assert exp_clean.trace is not None
        assert exp_clean.trace.shape == (3, 3)
        assert np.iscomplexobj(exp_clean.trace)
        
        # Test with disable_noise=False
        exp_noisy = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4950.0,
            tone1_stop=5050.0,
            tone1_step=50.0,
            tone1_amp=0.12,
            tone2_start=4750.0,
            tone2_stop=4850.0,
            tone2_step=50.0,
            tone2_amp=0.06,
            same_channel=False,
            num_avs=400,
            rep_rate=0.001,
            mp_width=0.8,
            set_qubit='f01',
            disable_noise=False  # Explicitly false
        )
        
        # Should also run successfully
        assert exp_noisy.trace is not None
        assert exp_noisy.trace.shape == (3, 3)
        assert np.iscomplexobj(exp_noisy.trace)
    
    def test_backward_compatibility_default_behavior(self, simulation_setup, test_qubit):
        """Test that default behavior maintains existing noise behavior."""
        # Default behavior (no disable_noise specified) should be equivalent to disable_noise=False
        exp_default = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4950.0,
            tone1_stop=5050.0,
            tone1_step=100.0,
            tone1_amp=0.1,
            tone2_start=4750.0,
            tone2_stop=4850.0,
            tone2_step=100.0,
            tone2_amp=0.05,
            num_avs=500
            # No disable_noise parameter - should default to False
        )
        
        exp_explicit_false = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4950.0,
            tone1_stop=5050.0,
            tone1_step=100.0,
            tone1_amp=0.1,
            tone2_start=4750.0,
            tone2_stop=4850.0,
            tone2_step=100.0,
            tone2_amp=0.05,
            num_avs=500,
            disable_noise=False  # Explicitly false
        )
        
        # Both should have same structure (noise makes them different but same shape)
        assert exp_default.trace.shape == exp_explicit_false.trace.shape
        assert exp_default.result['Magnitude'].shape == exp_explicit_false.result['Magnitude'].shape
        
        # Both should be complex and finite
        assert np.iscomplexobj(exp_default.trace)
        assert np.iscomplexobj(exp_explicit_false.trace)
        assert np.all(np.isfinite(exp_default.trace))
        assert np.all(np.isfinite(exp_explicit_false.trace))
    
    def test_noise_scaling_independence_2d(self, simulation_setup, test_qubit):
        """Test that clean 2D data is independent of noise scaling parameters."""
        # Clean data should be identical regardless of num_avs when disable_noise=True
        
        exp_low_avs = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4950.0,
            tone1_stop=5050.0,
            tone1_step=100.0,
            tone1_amp=0.1,
            tone2_start=4750.0,
            tone2_stop=4850.0,
            tone2_step=100.0,
            tone2_amp=0.05,
            num_avs=100,  # Low averages
            disable_noise=True
        )
        
        exp_high_avs = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4950.0,
            tone1_stop=5050.0,
            tone1_step=100.0,
            tone1_amp=0.1,
            tone2_start=4750.0,
            tone2_stop=4850.0,
            tone2_step=100.0,
            tone2_amp=0.05,
            num_avs=10000,  # High averages
            disable_noise=True
        )
        
        # Clean data should be identical regardless of num_avs
        np.testing.assert_array_equal(exp_low_avs.trace, exp_high_avs.trace)
        np.testing.assert_array_equal(exp_low_avs.result['Magnitude'], exp_high_avs.result['Magnitude'])
    
    def test_edge_cases_and_error_conditions_2d(self, simulation_setup, test_qubit):
        """Test edge cases for two-tone spectroscopy noise disable functionality."""
        
        # Test with minimal 2D sweep (1x1 grid)
        exp_minimal = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=5000.0,
            tone1_stop=5010.0,
            tone1_step=20.0,  # Only 1 point
            tone1_amp=0.1,
            tone2_start=4800.0,
            tone2_stop=4810.0,
            tone2_step=20.0,  # Only 1 point
            tone2_amp=0.05,
            num_avs=200,
            disable_noise=True
        )
        
        # Should produce 1x1 2D array
        assert exp_minimal.trace.shape == (1, 1)
        assert np.iscomplexobj(exp_minimal.trace)
        assert np.all(np.isfinite(exp_minimal.trace))
        
        # Test with larger asymmetric grid
        exp_asymmetric = TwoToneQubitSpectroscopy(
            dut_qubit=test_qubit,
            tone1_start=4900.0,
            tone1_stop=5100.0,
            tone1_step=100.0,  # 3 points
            tone1_amp=0.1,
            tone2_start=4700.0,
            tone2_stop=4800.0,
            tone2_step=25.0,   # 5 points
            tone2_amp=0.05,
            num_avs=300,
            disable_noise=True
        )
        
        # Should produce 3x5 2D array
        assert exp_asymmetric.trace.shape == (3, 5)
        assert np.iscomplexobj(exp_asymmetric.trace)
        assert np.all(np.isfinite(exp_asymmetric.trace))
    
    def test_complex_data_integrity_2d(self, simulation_setup, test_qubit):
        """Test that 2D complex data integrity is maintained in both clean and noisy modes."""
        
        params = {
            'dut_qubit': test_qubit,
            'tone1_start': 4900.0,
            'tone1_stop': 5000.0,
            'tone1_step': 50.0,
            'tone1_amp': 0.1,
            'tone2_start': 4700.0,
            'tone2_stop': 4800.0,
            'tone2_step': 50.0,
            'tone2_amp': 0.05,
            'num_avs': 500
        }
        
        # Test clean data
        exp_clean = TwoToneQubitSpectroscopy(**params, disable_noise=True)
        
        # Verify 2D complex data properties
        assert exp_clean.trace.ndim == 2
        assert np.iscomplexobj(exp_clean.trace)
        assert np.all(np.isfinite(exp_clean.trace))
        assert np.all(np.isfinite(exp_clean.result['Magnitude']))
        assert np.all(np.isfinite(exp_clean.result['Phase']))
        
        # Test noisy data
        exp_noisy = TwoToneQubitSpectroscopy(**params, disable_noise=False)
        
        # Verify 2D complex data properties for noisy data too
        assert exp_noisy.trace.ndim == 2
        assert np.iscomplexobj(exp_noisy.trace)
        assert np.all(np.isfinite(exp_noisy.trace))
        assert np.all(np.isfinite(exp_noisy.result['Magnitude']))
        assert np.all(np.isfinite(exp_noisy.result['Phase']))
        
        # 2D magnitudes should be non-negative in both cases
        assert np.all(exp_clean.result['Magnitude'] >= 0)
        assert np.all(exp_noisy.result['Magnitude'] >= 0)
        
        # Both should have same 2D shape
        assert exp_clean.trace.shape == exp_noisy.trace.shape
        assert exp_clean.result['Magnitude'].shape == exp_noisy.result['Magnitude'].shape