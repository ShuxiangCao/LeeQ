"""
Tests for leeq.theory.fits.fit_exp
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from leeq.theory.fits.fit_exp import (
    fit_1d_freq, _fit_exp_decay, fit_exp_decay_with_cov,
    _fit_1d_freq_exp, fit_1d_freq_exp_with_cov,
    _fft_based_initial_estimation, _fit_2d_freq, fit_2d_freq_with_cov,
    _fit_2d_freq_curvefit
)


class TestFit1DFreq:
    """Test suite for fit_1d_freq function."""

    def test_fit_1d_freq_sinusoidal_data(self):
        """Test fitting known sinusoidal data."""
        # Generate known sinusoidal data
        dt = 0.01
        t = np.linspace(0, 10, 1000)
        freq_true = 2.5  # Hz
        amp_true = 1.0
        phase_true = np.pi / 4
        offset_true = 0.5

        z = amp_true * np.sin(2 * np.pi * freq_true * t + phase_true) + offset_true

        # Fit the data
        result = fit_1d_freq(z, dt)

        # Check results are within reasonable tolerance
        assert abs(result['Frequency'] - freq_true) < 0.1
        assert abs(result['Amplitude'] - amp_true) < 0.1
        assert abs(result['Offset'] - offset_true) < 0.1

    def test_fit_1d_freq_with_fixed_frequency(self):
        """Test fitting with fixed frequency."""
        dt = 0.01
        t = np.linspace(0, 5, 500)
        freq_true = 1.5
        z = np.sin(2 * np.pi * freq_true * t) + 0.1 * np.random.randn(len(t))

        result = fit_1d_freq(z, dt, fix_frequency=True, freq_guess=freq_true)

        # Frequency should be exactly the guess
        assert abs(result['Frequency'] - freq_true) < 1e-6
        assert 'Amplitude' in result
        assert 'Phase' in result

    def test_fit_1d_freq_no_freq_bound(self):
        """Test fitting without frequency bounds."""
        dt = 0.01
        t = np.linspace(0, 2, 200)
        z = np.sin(2 * np.pi * 3.0 * t)

        result = fit_1d_freq(z, dt, use_freq_bound=False)

        assert 'Frequency' in result
        assert result['Residual'] >= 0

    def test_fit_1d_freq_assertion_error(self):
        """Test assertion error when fix_frequency=True but no freq_guess."""
        dt = 0.01
        z = np.sin(np.linspace(0, 10, 100))

        with pytest.raises(AssertionError):
            fit_1d_freq(z, dt, fix_frequency=True, freq_guess=None)

    def test_fit_1d_freq_with_tstart(self):
        """Test fitting with non-zero start time."""
        dt = 0.01
        tstart = 2.0
        t = np.linspace(tstart, tstart + 5, 500)
        z = np.sin(2 * np.pi * 1.0 * t)

        result = fit_1d_freq(z, dt, tstart=tstart)

        assert abs(result['Frequency'] - 1.0) < 0.1


class TestFitExpDecay:
    """Test suite for exponential decay fitting functions."""

    def test_fit_exp_decay_with_dt(self):
        """Test exponential decay fitting with time step."""
        dt = 0.1
        t_max = 10.0
        n_points = int(t_max / dt) + 1
        t = np.linspace(0, t_max, n_points)

        # Known parameters
        amp_true = 2.0
        decay_true = 3.0
        offset_true = 0.2

        z = amp_true * np.exp(-t / decay_true) + offset_true

        result = _fit_exp_decay(z, dt=dt)

        assert abs(result['Amplitude'] - amp_true) < 0.1
        assert abs(result['Decay'] - decay_true) < 0.3
        assert abs(result['Offset'] - offset_true) < 0.1

    def test_fit_exp_decay_with_t_array(self):
        """Test exponential decay fitting with explicit time array."""
        t = np.linspace(0, 8, 80)

        amp_true = 1.5
        decay_true = 2.5
        offset_true = 0.1

        z = amp_true * np.exp(-t / decay_true) + offset_true

        result = _fit_exp_decay(z, t=t)

        assert abs(result['Amplitude'] - amp_true) < 0.1
        assert abs(result['Decay'] - decay_true) < 0.3

    def test_fit_exp_decay_assertion_error(self):
        """Test assertion when neither dt nor t is provided."""
        z = np.array([1, 2, 3, 4])

        with pytest.raises(AssertionError):
            _fit_exp_decay(z)

    def test_fit_exp_decay_with_cov(self):
        """Test exponential decay with covariance calculation."""
        dt = 0.1
        t = np.linspace(0, 5, 50)
        z = 2.0 * np.exp(-t / 2.0) + 0.1

        result = fit_exp_decay_with_cov(z, dt=dt)

        # Check that uncertainties are returned
        assert hasattr(result['Amplitude'], 'nominal_value')
        assert hasattr(result['Decay'], 'nominal_value')
        assert hasattr(result['Offset'], 'nominal_value')
        assert 'Cov' in result


class TestFit1DFreqExp:
    """Test suite for 1D frequency exponential fitting."""

    def test_fit_1d_freq_exp_basic(self):
        """Test basic 1D frequency exponential fitting."""
        dt = 0.05
        t = np.linspace(0, 10, 200)
        freq_true = 1.0
        amp_true = 1.0
        decay_true = 2.0

        z = amp_true * np.exp(-t / decay_true) * np.sin(2 * np.pi * freq_true * t)

        result = _fit_1d_freq_exp(z, dt)

        assert abs(result['Frequency'] - freq_true) < 0.2
        assert abs(result['Amplitude'] - amp_true) < 0.2
        assert result['Decay'] > 0

    def test_fit_1d_freq_exp_with_cov(self):
        """Test 1D frequency exponential fitting with covariance."""
        dt = 0.05
        t = np.linspace(0, 8, 160)
        z = np.exp(-t / 3.0) * np.sin(2 * np.pi * 0.8 * t) + 0.1

        result = fit_1d_freq_exp_with_cov(z, dt)

        # Check uncertainties are returned
        assert hasattr(result['Frequency'], 'nominal_value')
        assert hasattr(result['Amplitude'], 'nominal_value')
        assert 'Cov' in result

    def test_fit_1d_freq_exp_no_bounds(self):
        """Test fitting without frequency bounds."""
        dt = 0.02
        t = np.linspace(0, 5, 250)
        z = np.exp(-t / 4.0) * np.sin(2 * np.pi * 2.0 * t)

        result = _fit_1d_freq_exp(z, dt, use_freq_bound=False)

        assert 'Frequency' in result
        assert 'Decay' in result


class TestFFTBasedEstimation:
    """Test suite for FFT-based frequency estimation."""

    def test_fft_based_initial_estimation(self):
        """Test FFT-based frequency estimation."""
        dt = 0.01
        freq_true = 5.0
        t = np.linspace(0, 2, 200)
        z = np.sin(2 * np.pi * freq_true * t)

        estimated_freq, estimated_amp = _fft_based_initial_estimation(z, dt)

        assert abs(estimated_freq - freq_true) < 0.5
        assert estimated_amp > 0


class TestFit2DFreq:
    """Test suite for 2D frequency fitting."""

    def test_fit_2d_freq_basic(self):
        """Test basic 2D frequency fitting."""
        dt = 0.01
        t = np.linspace(0, 5, 500)
        freq_true = 2.0
        amp_true = 1.0

        # Create complex exponential signal
        z = amp_true * np.exp(1j * 2 * np.pi * freq_true * t)

        result = _fit_2d_freq(z, dt)

        assert abs(result['Frequency'] - freq_true) < 0.2
        assert abs(result['Amplitude'] - amp_true) < 0.2

    def test_fit_2d_freq_with_fixed_frequency(self):
        """Test 2D frequency fitting with fixed frequency."""
        dt = 0.01
        freq_guess = 3.0
        t = np.linspace(0, 3, 300)
        z = np.exp(1j * 2 * np.pi * freq_guess * t)

        result = _fit_2d_freq(z, dt, fix_frequency=True, freq_guess=freq_guess)

        assert 'Frequency' in result
        assert 'Amplitude' in result

    def test_fit_2d_freq_with_cov(self):
        """Test 2D frequency fitting with covariance."""
        dt = 0.02
        t = np.linspace(0, 4, 200)
        z = np.exp(1j * 2 * np.pi * 1.5 * t) * np.exp(-t / 5.0)

        result = fit_2d_freq_with_cov(z, dt)

        # Should return ufloats
        assert hasattr(result['Frequency'], 'nominal_value')
        assert hasattr(result['Amplitude'], 'nominal_value')

    def test_fit_2d_freq_curvefit(self):
        """Test 2D frequency curve fitting function."""
        dt = 0.01
        freq_guess = 2.0
        amp_guess = 1.0
        phi_guess = 0.0
        offset_real_guess = 0.1
        offset_imag_guess = 0.0
        t = np.linspace(0, 3, 300)

        z = amp_guess * np.exp(1j * (2 * np.pi * freq_guess * t + phi_guess)) + complex(offset_real_guess, offset_imag_guess)

        result = _fit_2d_freq_curvefit(z, dt, freq_guess, amp_guess, phi_guess,
                                      offset_real_guess, offset_imag_guess, t=t)

        assert hasattr(result['Frequency'], 'nominal_value')
        assert 'Residual' in result


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_array(self):
        """Test behavior with empty arrays."""
        z = np.array([])
        dt = 0.1

        # Should handle gracefully or raise appropriate error
        with pytest.raises((IndexError, ValueError)):
            fit_1d_freq(z, dt)

    def test_single_point(self):
        """Test behavior with single data point."""
        z = np.array([1.0])
        dt = 0.1

        with pytest.raises((IndexError, ValueError)):
            fit_1d_freq(z, dt)

    def test_constant_data(self):
        """Test fitting constant (no oscillation) data."""
        z = np.ones(100) * 2.0
        dt = 0.01

        result = fit_1d_freq(z, dt)

        # Should handle gracefully - frequency might be near zero
        assert 'Frequency' in result
        assert abs(result['Offset'] - 2.0) < 0.1

    def test_noisy_data(self):
        """Test fitting very noisy data."""
        dt = 0.01
        t = np.linspace(0, 5, 500)
        signal = np.sin(2 * np.pi * 1.0 * t)
        noise = 0.5 * np.random.randn(len(t))
        z = signal + noise

        result = fit_1d_freq(z, dt)

        # Should still find approximate frequency
        assert abs(result['Frequency'] - 1.0) < 0.5


class TestParameterValidation:
    """Test parameter validation and bounds."""

    def test_negative_amplitude_correction(self):
        """Test correction of negative amplitude."""
        dt = 0.01
        t = np.linspace(0, 2, 200)
        # Create signal that might result in negative amplitude estimate
        z = -np.sin(2 * np.pi * 2.0 * t) + 1.0

        result = fit_1d_freq(z, dt)

        # Amplitude should be positive after correction
        assert result['Amplitude'] >= 0

    def test_phase_normalization(self):
        """Test phase normalization to [-pi, pi]."""
        dt = 0.01
        t = np.linspace(0, 3, 300)
        z = np.sin(2 * np.pi * 1.0 * t + 3 * np.pi)  # Phase > pi

        result = fit_1d_freq(z, dt)

        # Phase should be normalized to [-pi, pi]
        assert -np.pi <= result['Phase'] <= np.pi
