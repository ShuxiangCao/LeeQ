import numpy as np
from typing import Optional, Tuple

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def root_lorentzian(
    f: float, f0: float, kappa: float, amp: float, baseline: float
) -> float:
    """
    Calculate the root of the Lorentzian function.

    Parameters:
    f (float): The frequency at which the Lorentzian is evaluated.
    f0 (float): The resonant frequency (i.e., the peak position).
    kappa (float): The resonator line width.
    amp (float): Amplitude of the Lorentzian peak.
    baseline (float): Baseline offset of the Lorentzian.

    Returns:
    float: The absolute value of the root Lorentzian function at frequency f.

    Note:
    The Lorentzian function is given by:
        L(f) = amp / [1 + 2jQ(f - f0)/f0] + baseline
    Where:
        - j is the imaginary unit.
        - The root Lorentzian is obtained by taking the absolute value.
    """

    Q = f0 / kappa

    # Compute the Lorentzian function
    lorentzian = amp / (1 + (2j * Q * (f - f0) / f0))

    abs = amp - np.abs(lorentzian) + baseline
    angle = np.angle(lorentzian)
    angle = angle * 2  # TODO: explain why need to double the angle to get the root

    return abs * np.exp(1.0j * angle)


def get_t_list(sampling_rate: int, width: float) -> np.ndarray:
    """
    Get the time list for the pulse shape.

    Parameters:
        sampling_rate (int): The sampling rate of the pulse shape.
        width (float): The width of the pulse shape.

    Returns:
        np.ndarray: The time list for the pulse shape.
    """

    # Get the number of samples
    num_samples = int(width * sampling_rate + 0.5)

    # Get the time list
    t_list = np.linspace(0, width, num_samples, endpoint=False)

    return t_list


def soft_square(
    sampling_rate: int,
    amp: float,
    phase: Optional[float] = None,
    width: Optional[float] = None,
    rise: Optional[float] = None,
    trunc: Optional[float] = None,
    delay: float = 0.0,
    phase_shift: float = 0,
    ex_delay: float = 0,
) -> np.ndarray:
    """
    Generate a soft square wave.

    Parameters:
    - sampling_rate (int): Sampling rate in Megasamples per second.
    - amp (float): Amplitude of the signal.
    - phase (float, optional): Phase of the signal.
    - width (float, optional): Width of the pulse.
    - rise (float, optional): Rise time of the signal.
    - trunc (float, optional): Time to truncate the signal.
    - delay (float, optional): Delay of the signal.
    - phase_shift (float, optional): Phase shift of the signal.
    - ex_delay (float, optional): Extra delay of the signal.

    Returns:
    - np.ndarray: Generated soft square wave.
    """
    full_width = width + 2.0 * rise * trunc + ex_delay
    t = get_t_list(sampling_rate, full_width + delay)

    t -= 0.5 * delay
    y = (
        amp
        * np.exp(1.0j * (phase + phase_shift))
        * 0.5
        * (np.tanh((t + 0.5 * width) / rise) - np.tanh((t - 0.5 * width) / rise))
    )

    if ex_delay > 0:
        extra_delay_y = get_t_list(sampling_rate, delay)
        y = np.concatenate([extra_delay_y, y])

    return y


def estimate_relative_entropy(dist_p, dist_q, kernel="gaussian", search_params={}):
    """
    Estimate the relative entropy between two distributions, using kernel density estimation and
    monte carlo integration.

    Parameters:
    - dist_p (np.ndarray): The first distribution.
    - dist_q (np.ndarray): The second distribution.
    - kernel (str): The kernel to use for kernel density estimation.
    - search_params (dict): The parameters to use for grid search.

    Returns:
    - float: The relative entropy between the two distributions.
    """

    params = {"bandwidth": np.logspace(-1, 2, 25)}
    params.update(search_params)

    # Fit the distributions using kernel density estimation
    def find_kde(dist):
        grid = GridSearchCV(KernelDensity(kernel=kernel), params)
        grid.fit(dist)
        kde = grid.best_estimator_
        print("best bandwidth: {0}".format(kde.bandwidth))
        return kde

    kde_p = find_kde(dist_p)
    kde_q = find_kde(dist_q)

    # Estimate the log probability of each sample
    log_prob_p = kde_p.score_samples(dist_p)
    log_prob_q = kde_q.score_samples(dist_p)

    # Estimate the relative entropy
    rel_entropy = np.mean(log_prob_p - log_prob_q)

    return rel_entropy
