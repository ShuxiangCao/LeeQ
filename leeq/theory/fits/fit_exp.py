import numpy as np
from scipy.optimize import minimize, curve_fit
from typing import Dict, Tuple, Any
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize, curve_fit

import numpy as np
from scipy.optimize import minimize, curve_fit
from typing import Dict, Tuple, Any
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize, curve_fit
import uncertainties as unc

def fit_1d_freq(
        z,
        dt,
        use_freq_bound=True,
        fix_frequency=False,
        tstart=0,
        freq_guess=None,
        **kwargs):

    if fix_frequency:
        assert freq_guess is not None

    if freq_guess is None:
        rfft = np.abs(np.fft.rfft(z))
        f = np.fft.rfftfreq(len(z), dt)
        imax = np.argmax(rfft[1:]) + 1
        fmax = f[imax]

        omega = fmax

        df = f[1] - f[0]
        min_omega = fmax - df
        max_omega = fmax + df
    else:
        omega = freq_guess
        max_omega = omega * 1.5
        min_omega = omega * 0.5

    t = np.linspace(tstart, tstart + dt * (len(z) - 1), len(z))

    cosz = z * np.cos(2.0 * np.pi * omega * t)
    sinz = z * np.sin(2.0 * np.pi * omega * t)
    offset = np.mean(z)
    # 2 * np.std(z) ** 0.5  # 2 ** 0.5 * np.mean((z - offset) ** 2) ** 0.5
    amp = 0.5 * (np.max(z) - np.min(z))
    # 3 * np.pi / 2  # np.arctan2(np.mean(cosz),np.mean(sinz))
    phi = np.arcsin(np.max([np.min([(z[0] - offset) / amp, 1]), -1]))
    if z[1] - z[0] < 0:
        phi = np.pi - phi

    def leastsq(x, t, z):
        omega, amp, phi, offset = x
        fit = amp * np.sin(2. * np.pi * omega * t + phi) + offset
        return np.mean((fit - z) ** 2) * 1e5

    def leastsq_without_omega(x, omega, t, z):
        amp, phi, offset = x
        fit = amp * np.sin(2. * np.pi * omega * t + phi) + offset
        return np.mean((fit - z) ** 2) * 1e5

    def leastsq_phi_only(phi, x, t, z):
        omega, amp, offset = x
        fit = amp * np.sin(2. * np.pi * omega * t + phi) + offset
        return np.mean((fit - z) ** 2) * 1e5

    def leastsq_omega_only(omega, x, t, z):
        phi, amp, offset = x
        fit = amp * np.sin(2. * np.pi * omega * t + phi) + offset
        return np.mean((fit - z) ** 2) * 1e5

    result = minimize(leastsq_phi_only, np.array(
        [phi]), args=(np.array([omega, amp, offset]), t, z))
    phi = result.x[0]

    if not fix_frequency:
        result = minimize(leastsq_omega_only, np.array(
            [omega]), args=(np.array([phi, amp, offset]), t, z))
        omega = result.x[0]

    result = minimize(leastsq_phi_only, np.array(
        [phi]), args=(np.array([omega, amp, offset]), t, z))
    phi = result.x[0]

    args = dict()

    args.update(kwargs)

    if not fix_frequency:
        if use_freq_bound:
            args['bounds'] = ((min_omega, max_omega),
                              (None, None), (None, None), (None, None))
        result = minimize(leastsq, np.array(
            [omega, amp, phi, offset]), args=(t, z), **args)
        omega, amp, phi, offset = result.x
    else:
        result = minimize(leastsq_without_omega, np.array(
            [amp, phi, offset]), args=(omega, t, z), **args)
        amp, phi, offset = result.x

    r = result.fun

    # We here make sure amp > 0

    if amp < 0:
        phi = phi + np.pi
        amp = -amp

    while phi > np.pi:
        phi -= np.pi * 2
    while phi < -np.pi:
        phi += np.pi * 2

    return {
        'Frequency': omega,
        'Amplitude': amp,
        'Phase': phi,
        'Offset': offset,
        'Residual': r}


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

def fft_based_initial_estimation(z, dt):
    n = len(z)
    fft_values = np.fft.fft(z)
    frequencies = np.fft.fftfreq(n, dt)
    positive_frequencies = frequencies[:n // 2]
    positive_fft_values = np.abs(fft_values[:n // 2])

    dominant_index = np.argmax(positive_fft_values)
    dominant_frequency = positive_frequencies[dominant_index] if positive_frequencies[dominant_index] >= 0 else - \
    positive_frequencies[dominant_index]

    return dominant_frequency, np.max(positive_fft_values)

def fit_2d_freq(z, dt, use_freq_bound=True, fix_frequency=False, freq_guess=None, t=None, **kwargs):

    # Initial frequency and amplitude estimation
    # estimated_frequency, estimated_amplitude = fft_based_initial_estimation(z, dt)
    # print(f'Initial Frequency Estimate: {estimated_frequency}')
    # print(f'Initial Amplitude Estimate: {estimated_amplitude}')

    if fix_frequency:
        assert freq_guess is not None

    args = kwargs.copy()

    z = np.asarray(z)

    if freq_guess is None:
        fft = np.abs(np.fft.fft(z))
        f = np.fft.fftfreq(len(z), dt)
        imax = np.argmax(fft[1:]) + 1
        fmax = f[imax]

        omega = fmax.real
        df = f[1] - f[0]
        min_omega = abs(fmax) - df
        max_omega = abs(fmax) + df
    else:
        omega = abs(freq_guess)
        max_omega = omega * 2
        min_omega = -omega * 2



    if t is None:
        t = np.linspace(0, dt * (len(z) - 1), len(z))

    offset = np.mean(z)
    offset_real = np.real(offset)
    offset_imag = np.imag(offset)

    amp = np.max(np.abs(z - offset))

    phi = np.angle(z[0] - offset)

    if isinstance(phi, np.ndarray) and phi.size == 1:
        phi = phi.item()

    def leastsq(x, t, z):
        omega, amp, phi, offset_real, offset_imag, decay = x
        fit = amp * np.exp(-decay * t) * np.exp(1.j * (2. * np.pi * omega * t + phi)) + offset_real + 1.j * offset_imag
        return np.mean(np.abs(fit - z) ** 2).real * 1e5

    def leastsq_without_omega(x, omega, t, z):
        amp, phi, offset_real, offset_imag, decay = x
        fit = amp * np.exp(-decay * t) * np.exp(1.j * (2. * np.pi * omega * t + phi)) + offset_real + 1.j * offset_imag
        return np.mean(np.abs(fit - z) ** 2).real * 1e5

    def leastsq_phi_only(phi, x, t, z):
        omega, amp, offset_real, offset_imag, decay = x
        fit = amp * np.exp(-decay * t) * np.exp(1.j * (2. * np.pi * omega * t + phi)) + offset_real + 1.j * offset_imag
        return np.mean(np.abs(fit - z) ** 2).real * 1e5

    def leastsq_omega_only(omega, x, t, z):
        amp, phi, offset_real, offset_imag, decay = x
        fit = amp * np.exp(-decay * t) * np.exp(1.j * (2. * np.pi * omega * t + phi)) + offset_real + 1.j * offset_imag
        return np.mean(np.abs(fit - z) ** 2).real * 1e5

    initial_guess = np.array([omega, amp, phi, offset_real, offset_imag, 0.1])  # Initial decay guess

    args = {}
    if use_freq_bound:
        args['bounds'] = ((min_omega, max_omega), (None, None), (None, None), (None, None), (None, None),(None, None))

    result = minimize(leastsq, initial_guess, args=(t, z),**args)
    omega, amp, phi, offset_real, offset_imag, decay = result.x

    if amp < 0:
        phi = phi + np.pi
        amp = -amp

    return {
        'Frequency': omega,
        'Amplitude': amp,
        'Phase': phi,
        'Offset': complex(offset_real, offset_imag),
        'Decay': decay,
        'Residual': result.fun
    }

import numpy as np
from scipy.optimize import curve_fit
def fit_2d_freq_curvefit(z, dt,
                         freq_guess, amp_guess, phi_guess, offset_real_guess,
                         offset_imag_guess , t=None):
    z = np.asarray(z)

    if t is None:
        t = np.linspace(0, dt * (len(z) - 1), len(z))

    def model(t, omega, amp, phi, offset_real, offset_imag):
        return amp  * np.exp(1.j * (2. * np.pi * omega * t + phi)) + offset_real + 1.j * offset_imag

    p0 = [freq_guess, amp_guess, phi_guess, offset_real_guess, offset_imag_guess]

    # We need to define a custom least squares function as curve_fit expects real-valued functions
    def complex_fit_func(t, omega, amp, phi, offset_real, offset_imag):
        fitted_curve = model(t, omega, amp, phi, offset_real, offset_imag)
        return np.append(np.real(fitted_curve - z), np.imag(fitted_curve - z))

    popt, pcov = curve_fit(complex_fit_func, t, np.zeros(2 * len(z)), p0=np.real(p0))

    # Extract parameters and calculate parameter uncertainties
    perr = np.sqrt(np.diag(pcov))

    return {
        'Frequency': unc.ufloat(popt[0], perr[0]),
        'Amplitude': unc.ufloat(popt[1], perr[1]),
        'Phase': unc.ufloat(popt[2], perr[2]),
        'Offset_real': unc.ufloat(popt[3], perr[3]),
        'Offset_imag': unc.ufloat(popt[4], perr[4]),
        'Residual': np.sum(np.abs(model(t, *popt) - z)**2)
    }

def fit_2d_freq_with_cov(z, dt, use_freq_bound=True, fix_frequency=False, freq_guess=None, t=None, **kwargs):
    direct_estimation = fit_2d_freq(z, dt, use_freq_bound=use_freq_bound, fix_frequency=fix_frequency, freq_guess=freq_guess, t=t, **kwargs)
    curve_fit_estimation = fit_2d_freq_curvefit(z, dt,
                                                freq_guess=direct_estimation['Frequency'],
                                                amp_guess=direct_estimation['Amplitude'],
                                                phi_guess=direct_estimation['Phase'],
                                                offset_real_guess=direct_estimation['Offset'].real,
                                                offset_imag_guess=direct_estimation['Offset'].imag,
                                                t=t)

    return curve_fit_estimation