import numpy as np
from scipy.optimize import minimize, curve_fit
from typing import Dict, Tuple, Any


def fit_1d_freq_exp(z: np.ndarray, dt: float, use_freq_bound: bool = True) -> Dict[str, float]:
    """
    Fits a 1D frequency exponential to a dataset.

    Parameters:
    z (np.ndarray): The dataset to fit.
    dt (float): The time step.
    use_freq_bound (bool): Whether to bound the frequency during fitting. Defaults to True.

    Returns:
    dict: A dictionary containing the parameters of the fitted function.
    """
    rfft = np.abs(np.fft.rfft(z))
    frequencies = np.fft.rfftfreq(len(z), dt)

    # Ignore the zero frequency component when finding the maximum
    imax = np.argmax(rfft[1:]) + 1
    fmax = frequencies[imax]

    t = np.linspace(0., dt * (len(z) - 1), len(z))

    omega = fmax

    df = frequencies[1] - frequencies[0]
    min_omega = fmax - df
    max_omega = fmax + df

    # Pre-compute sine and cosine terms
    cosz = z * np.cos(2.0 * np.pi * omega * t)
    sinz = z * np.sin(2.0 * np.pi * omega * t)

    offset = np.mean(z)
    amp = 0.5 * (np.max(z) - np.min(z))
    T = t[-1]

    # Estimate the phase
    phi = np.arcsin(np.clip((z[0] - offset) / amp, -1, 1))
    if z[1] - z[0] < 0:
        phi = np.pi - phi

    def leastsq_phi_only(phi: float, x: np.ndarray, t: np.ndarray, z: np.ndarray) -> float:
        omega, amp, offset, T = x
        fit = amp * np.exp(-t / T) * np.sin(2. * np.pi * omega * t + phi) + offset
        return np.sum((fit - z) ** 2) * 1e3

    def leastsq(x: np.ndarray, t: np.ndarray, z: np.ndarray) -> float:
        omega, amp, phi, offset, T = x
        fit = amp * np.exp(-t / T) * np.sin(2. * np.pi * omega * t + phi) + offset
        return np.sum((fit - z) ** 2) * 1e3

    args = {}
    if use_freq_bound:
        args['bounds'] = ((min_omega, max_omega), (None, None), (None, None), (None, None), (None, None))

    result = minimize(leastsq_phi_only, np.array([phi]), args=(np.array([omega, amp, offset, T]), t, z))

    phi = result.x[0]

    result = minimize(leastsq, np.array([omega, amp, phi, offset, T]), args=(t, z), tol=1e-6, **args)

    omega, amp, phi, offset, T = result.x

    # Ensure amplitude is positive
    if amp < 0:
        phi += np.pi
        amp = -amp

    # Normalize the phase to [-pi, pi]
    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    return {'Frequency': omega, 'Amplitude': amp, 'Phase': phi, 'Offset': offset, 'Decay': T}


def fit_1d_freq_exp_with_cov(z: np.ndarray, dt: float, use_freq_bound: bool = True) -> Dict[str, Any]:
    """
    Fits a 1D frequency exponential to a dataset and calculates covariance.

    Parameters:
    z (np.ndarray): The dataset to fit.
    dt (float): The time step.
    use_freq_bound (bool): Whether to bound the frequency during fitting. Defaults to True.

    Returns:
    dict: A dictionary containing the parameters of the fitted function and covariance.
    """
    result = fit_1d_freq_exp(z=z, dt=dt, use_freq_bound=use_freq_bound)

    t = np.linspace(0., dt * (len(z) - 1), len(z))

    omega = result['Frequency']
    amp = result['Amplitude']
    phi = result['Phase']
    offset = result['Offset']
    T = result['Decay']

    def curve(t: np.ndarray, omega: float, amp: float, phi: float, offset: float, T: float) -> np.ndarray:
        return amp * np.exp(-t / T) * np.sin(2. * np.pi * omega * t + phi) + offset

    popt, pcov = curve_fit(curve, t, z, p0=[omega, amp, phi, offset, T])

    omega, amp, phi, offset, T = popt
    omega_std, amp_std, phi_std, offset_std, T_std = np.sqrt(np.diag(pcov)).tolist()

    return {'Frequency': (omega, omega_std),
            'Amplitude': (amp, amp_std),
            'Phase': (phi, phi_std),
            'Offset': (offset, offset_std),
            'Decay': (T, T_std),
            'Cov': pcov}
