import numpy as np
from scipy.optimize import minimize
from typing import List, Optional, Dict, Any
from .fit_exp import *

import numpy as np
from scipy.optimize import minimize, curve_fit
from typing import Optional, Union, Tuple, Dict, List


def fit_sinusoidal(
        data: np.ndarray,
        time_step: float,
        use_freq_bound: bool = True,
        fix_frequency: bool = False,
        start_time: float = 0,
        freq_guess: Optional[float] = None,
        **kwargs: Any
) -> Dict[str, float]:
    """
    Fit a sinusoidal model to 1D data.

    Parameters:
    data (np.ndarray): The 1D data array to fit.
    time_step (float): The time interval between data points.
    use_freq_bound (bool): Whether to bound the frequency during optimization.
    fix_frequency (bool): Whether to keep the frequency fixed during optimization.
    start_time (float): The start time of the data.
    freq_guess (Optional[float]): An initial guess for the frequency.
    **kwargs (Any): Additional keyword arguments for the optimizer.

    Returns:
    dict: A dictionary containing the optimized parameters and the residual of the fit.
    """

    # Ensure frequency is provided if it's fixed
    if fix_frequency:
        assert freq_guess is not None, "Initial frequency guess must be provided if frequency is fixed."

    # Estimate initial frequency if not provided
    if freq_guess is None:
        rfft = np.abs(np.fft.rfft(data))
        frequencies = np.fft.rfftfreq(len(data), time_step)
        # Skip the first element which is the zero-frequency component
        max_index = np.argmax(rfft[1:]) + 1
        dominant_freq = frequencies[max_index]

        omega = dominant_freq

        freq_resolution = frequencies[1] - frequencies[0]
        min_omega = dominant_freq - freq_resolution
        max_omega = dominant_freq + freq_resolution
    else:
        omega = freq_guess
        max_omega = omega * 1.5
        min_omega = omega * 0.5

    # Generate time data
    time = np.linspace(start_time, start_time +
                       time_step * (len(data) - 1), len(data))

    # Initial parameter guesses based on data properties
    offset = np.mean(data)
    amplitude = 0.5 * (np.max(data) - np.min(data))
    phase = np.arcsin(np.clip((data[0] - offset) / amplitude, -1, 1))
    if data[1] - data[0] < 0:
        phase = np.pi - phase

    # Objective functions for optimization
    def leastsq(params: List[float], t: np.ndarray, y: np.ndarray) -> float:
        omega, amplitude, phase, offset = params
        fit = amplitude * np.sin(2. * np.pi * omega * t + phase) + offset
        return np.mean((fit - y) ** 2) * 1e5

    def leastsq_without_omega(
            params: List[float],
            omega: float,
            t: np.ndarray,
            y: np.ndarray) -> float:
        amplitude, phase, offset = params
        fit = amplitude * np.sin(2. * np.pi * omega * t + phase) + offset
        return np.mean((fit - y) ** 2) * 1e5

    # Optimization process
    optimization_args = dict()
    optimization_args.update(kwargs)

    if not fix_frequency:
        if use_freq_bound:
            optimization_args['bounds'] = (
                (min_omega, max_omega), (None, None), (None, None), (None, None))
        result = minimize(leastsq, np.array([omega, amplitude, phase, offset]), args=(
            time, data), **optimization_args)
        omega, amplitude, phase, offset = result.x
    else:
        result = minimize(leastsq_without_omega, np.array(
            [amplitude, phase, offset]), args=(omega, time, data), **optimization_args)
        amplitude, phase, offset = result.x

    residual = result.fun

    # Ensure amplitude is positive and phase is within [-pi, pi]
    if amplitude < 0:
        phase += np.pi
        amplitude = -amplitude

    phase = phase % (2 * np.pi)
    if phase > np.pi:
        phase -= 2 * np.pi

    return {
        'Frequency': omega,
        'Amplitude': amplitude,
        'Phase': phase,
        'Offset': offset,
        'Residual': residual}


