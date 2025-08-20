import numpy as np
import scipy.linalg
from typing import Optional, List, Tuple

__all__ = ["simulate_decay", 'fit_decay', 'show_one_plot']

from matplotlib import pyplot as plt
from scipy.optimize import minimize
from leeq.utils.utils import setup_logging

logger = setup_logging(__name__)


def normalize_gamma(gamma: np.ndarray) -> np.ndarray:
    """
    Normalizes the given gamma matrix to ensure that the sum of each row is zero.

    This is done by setting the diagonal elements to the negative sum of the
    respective row's off-diagonal elements.

    Args:
    gamma (np.ndarray): A square matrix of shape (N, N).

    Returns:
    np.ndarray: The normalized gamma matrix with the same shape.
    """
    # Set diagonal elements to zero
    for i in range(gamma.shape[0]):
        gamma[i, i] = 0

    # Compute the sum of each row and set the diagonal to the negative of this sum
    summation = np.sum(gamma, axis=1)
    for i in range(gamma.shape[0]):
        gamma[i, i] = -summation[i]

    return gamma


def single_photon_gamma_encode(gamma: np.ndarray) -> np.ndarray:
    """
    Encodes a gamma matrix by extracting and concatenating certain elements for
    simplified storage or transmission.

    Specifically, it extracts the first superdiagonal and the first subdiagonal.

    Args:
    gamma (np.ndarray): A square matrix of shape (N, N).

    Returns:
    np.ndarray: A 1D array of length 2*(N-1) containing concatenated superdiagonal and subdiagonal elements.
    """
    gamma_width = gamma.shape[0]

    # Initialize arrays to hold the superdiagonal and subdiagonal elements
    a = np.zeros(gamma_width - 1)
    b = np.zeros(gamma_width - 1)

    # Extract the elements
    for i in range(gamma_width - 1):
        a[i] = gamma[i, i + 1]
        b[i] = gamma[i + 1, i]

    return np.concatenate([a, b])


def single_photon_gamma_decode(encoded: np.ndarray) -> np.ndarray:
    """
    Decodes a previously encoded array back into a gamma matrix and normalizes it.

    Args:
    encoded (np.ndarray): A 1D array containing encoded superdiagonal and subdiagonal elements.

    Returns:
    np.ndarray: A normalized square matrix (gamma matrix) reconstructed from the encoded data.
    """
    gamma_width = len(encoded) // 2 + 1
    gamma = np.zeros((gamma_width, gamma_width))

    a = encoded[:gamma_width - 1]
    b = encoded[gamma_width - 1:]

    # Reconstruct the gamma matrix from the encoded data
    for i in range(gamma_width - 1):
        gamma[i, i + 1] = a[i]
        gamma[i + 1, i] = b[i]

    return normalize_gamma(gamma)


def simulate_decay(initial_distribution: np.ndarray, gamma: np.ndarray, time_resolution: float,
                   time_length: float) -> np.ndarray:
    """
    Simulates the decay of a system over time using a gamma matrix.

    Args:
    initial_distribution (np.ndarray): Initial state vector of the system.
    gamma (np.ndarray): Square matrix representing decay rates between states.
    time_resolution (float): Time interval at which to simulate the decay.
    time_length (float): Total duration of the simulation.

    Returns:
    np.ndarray: A matrix where each row represents the system state at each time step.
    """
    records = []
    current = initial_distribution

    # Compute the decay matrix for each step using fractional matrix power
    decay_per_step = scipy.linalg.fractional_matrix_power(np.eye(gamma.shape[0]) - gamma.T, time_resolution)

    size = time_length / time_resolution

    # Simulate the decay at each time step
    for i in range(int(size)):
        records.append(current)
        current = decay_per_step @ current

    return np.vstack(records)


def loss_function(x: np.ndarray,
                  probs: np.ndarray,
                  time_resolution: float,
                  time_length: float,
                  initial_state: Optional[np.ndarray] = None,
                  ignore_two_photon_transition: bool = True) -> float:
    """
    Calculate the loss based on the difference between simulated decay probabilities and observed probabilities.

    Parameters:
        x (np.ndarray): The flattened array containing both the initial state probabilities and the transition matrix elements.
        probs (np.ndarray): Observed probabilities array for each state over time.
        time_resolution (float): The time resolution of the simulation.
        time_length (float): The total length of time for which the decay is simulated.
        initial_state (Optional[np.ndarray]): Optional parameter to provide an initial state matrix. If None, it is extracted from `x`.
        ignore_two_photon_transition (bool): A flag to consider only single-photon transitions in the simulation.

    Returns:
        float: The calculated loss as the sum of squared differences between simulated and observed probabilities.
    """

    # Determine the number of quantum levels by comparing the size of `x` with expected sizes based on the level count.
    sizes = []  # This list will store the possible sizes of `x` for different level configurations.

    # Loop through possible levels from 2 to 4 (since the range is exclusive of the stop value).
    for n in range(2, 5):
        if ignore_two_photon_transition:
            # Calculate the size assuming a reduced matrix without two-photon transitions.
            # Size = transition elements + initial state probabilities.
            sizes.append(2 * (n - 1) + n * (n - 1))
        else:
            # Calculate the size with a full transition matrix included.
            sizes.append(n * n + n * (n - 1))

    # Find the actual number of levels by matching the size of `x` with calculated sizes.
    levels = np.argmax([x.size == s for s in sizes]) + 2

    # If the initial state is not provided, extract it from `x`.
    if initial_state is None:
        initial_state = x[:levels * (levels - 1)].reshape(levels - 1, levels)

    # Calculate the transition rates `gamma` from `x`, considering whether to ignore two-photon transitions.
    if not ignore_two_photon_transition:
        # Extract and normalize the full transition matrix.
        gamma = x[levels * (levels - 1):].reshape(levels, levels)
        gamma = normalize_gamma(gamma)  # Assuming `normalize_gamma` is defined elsewhere.
    else:
        # Decode only the single-photon transition rates.
        gamma = single_photon_gamma_decode(x[levels * (levels - 1):])  # Assuming this function is defined elsewhere.

    # Calculate the residuals for each state, which is the squared difference between the simulated and observed probabilities.
    residuals = [
        (simulate_decay(initial_distribution=initial_state[i, :], gamma=gamma, time_length=time_length,
                        time_resolution=time_resolution).flatten() - probs[:, i, :].flatten()) ** 2 for i in
        range(initial_state.shape[0])
    ]

    # Return the total loss as the sum of all residuals.
    return np.sum(residuals)


def fit_single_decay(sequence: np.ndarray, time_length: float, time_resolution: float) -> Tuple[float, float]:
    """
    Fit an exponential decay model to a given sequence of decay data.

    Parameters:
        sequence (np.ndarray): The observed decay data.
        time_length (float): The total time duration of the decay data.
        time_resolution (float): The time interval between consecutive data points.

    Returns:
        Tuple[float, float]: Tuple containing the decay constant and pre-exponential factor.
    """
    # Log-transform the sequence where values are positive
    log_sequence = np.log(sequence[sequence > 0])
    # Create time points array corresponding to the log-transformed sequence
    x = np.linspace(0, time_length, int(time_length / time_resolution))[sequence > 0]

    # Perform linear regression on the log-transformed data to find the decay parameters
    fit_params = np.polyfit(x, log_sequence, 1)  # Linear fit to log of the data

    # Define the exponential function model using the fitted parameters
    def exponential_decay_model(z: float, params: np.ndarray) -> float:
        return np.exp(params[1]) * np.exp(params[0] * z)

    # Return the pre-exponential factor and decay constant
    return np.exp(fit_params[1]), fit_params[0]


def fit_decay(probs: np.ndarray, time_length: float, time_resolution: float, verbose: bool = False) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Fit decay models to multiple sequences of decay probabilities.
     probs : [time index, trace index, measured state index]


    Parameters:
        probs (np.ndarray): A 3D array of probabilities indexed by time, trace, and measured state.
        time_length (float): The total time duration for each decay trace.
        time_resolution (float): The time interval between consecutive probability measurements.
        verbose (bool): If True, prints additional information about the fitting process.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the initial state probabilities and the gamma transition matrix.
    """
    # Validate input shape assumptions
    assert probs.shape[1] == probs.shape[2] - 1, "The number of traces must be one less than the number of states."

    # Initialize gamma transition matrix
    gamma_size = probs.shape[2]
    gamma_0 = np.zeros((gamma_size, gamma_size))
    initial_state = probs[0, :, :]  # Initial state probabilities from the first time slice

    # Fit decay rates for transitions from each state to itself
    for i in range(gamma_size - 1):
        _, decay_rate = fit_single_decay(probs[:, i, i], time_length, time_resolution)
        gamma_0[i + 1, i] = decay_rate

    # Normalize the gamma matrix to ensure proper probabilistic properties
    gamma_0 = normalize_gamma(gamma_0)  # Assuming `normalize_gamma` function is defined elsewhere

    # Display fitting results if verbose is True
    if verbose:
        logger.info(f"Fitted gamma0: {decay_rate}")
        logger.info(f"{gamma_0}")

    # Flatten and concatenate the initial state and gamma matrices for optimization
    x_guess = np.concatenate(
        [initial_state.flatten(), single_photon_gamma_encode(gamma_0)])  # Assuming this function is defined

    # Minimize the loss function to refine the estimates
    result = minimize(loss_function,
                      x_guess, args=(probs, time_resolution, time_length),
                      method='BFGS', options={'disp': verbose}, tol=1e-6)

    x = result.x
    initial_state = x[:gamma_size * (gamma_size - 1)].reshape([gamma_size - 1, gamma_size]).T
    gamma = single_photon_gamma_decode(x[gamma_size * (gamma_size - 1):])

    return initial_state, gamma


title_dict = {
    '00': rf"Qubit prepared to |0$\rangle$",
    '01': rf"Qubit prepared to |1$\rangle$",
    '12': rf"Qubit prepared to |2$\rangle$",
    '23': rf"Qubit prepared to |3$\rangle$",
    0: rf"Qubit prepared to |0$\rangle$",
    1: rf"Qubit prepared to |1$\rangle$",
    2: rf"Qubit prepared to |2$\rangle$",
    3: rf"Qubit prepared to |3$\rangle$",
}


def show_one_plot(ax: plt.Axes, probs: np.ndarray, j: int, time_length: float, time_resolution: int,
                  initial_distribution: Optional[np.ndarray] = None, gamma: Optional[float] = None) -> None:
    """
    Plot probability sequences and optional decay simulations on a given Axes object.

    Parameters:
        ax (plt.Axes): The matplotlib Axes object to plot on.
        probs (np.ndarray): 3D array of probabilities with dimensions [time_steps, series_index, state_index].
        j (int): Index of the series in `probs` to plot.
        time_length (float): Total time length of the experiment.
        time_resolution (int): Number of time points.
        initial_distribution (Optional[np.ndarray]): Initial distribution of states for decay simulation, if applicable.
        gamma (Optional[float]): Decay constant for the exponential decay simulation, if applicable.
    """
    if initial_distribution is not None and gamma is not None:
        fitted_distribution = simulate_decay(initial_distribution, gamma=gamma, time_length=time_length,
                                             time_resolution=time_resolution)

    markers = ['^', 's', 'D', 'o']  # Marker styles for different states
    color_set = ['k', 'r', 'g', 'b']  # Colors for different states

    # Iterate over the states and plot each probability sequence
    for i in range(probs.shape[-1]):
        prob_seq = probs[:, j, i]
        ax.scatter(np.linspace(0, time_length, len(prob_seq)), prob_seq, label=rf'$|{i}\rangle$',
                   marker=markers[i], facecolors='none', s=5,
                   color=color_set[i], alpha=0.8)
        if initial_distribution is not None and gamma is not None:
            ax.plot(np.linspace(0, time_length, len(prob_seq)), fitted_distribution[:, i],
                    color=color_set[i], alpha=1)

    ax.set_ylabel('Population')
    ax.set_xlabel(r'Delay time [$\mu$s]')
    ax.set_ylim([-0.1, 1.1])
    ax.set_title(title_dict[j + 1])
    ax.legend()


def plot(probs: np.ndarray, time_length: float, time_resolution: int,
         initial_distribution: Optional[np.ndarray] = None, gamma: Optional[float] = None,
         figsize: Tuple[int, int] = (5, 4)) -> None:
    """
    Generate and display a plot for each series in a set of probability data.

    Parameters:
        probs (np.ndarray): 3D array of probabilities with dimensions [time_steps, series_index, state_index].
        time_length (float): Total time length of the experiment.
        time_resolution (int): Number of time points.
        initial_distribution (Optional[np.ndarray]): Initial distribution for each series, if applicable.
        gamma (Optional[float]): Decay constant for the exponential decay simulation, if applicable.
        figsize (Tuple[int, int]): Tuple specifying the width and height of each subplot.
    """
    fig = plt.figure(figsize=(figsize[0] * probs.shape[1], figsize[1]))

    # Create a subplot for each series and plot it
    for j in range(probs.shape[1]):
        ax = fig.add_subplot(1, probs.shape[1], j + 1)
        show_one_plot(ax, probs, j, time_length, time_resolution,
                      initial_distribution=initial_distribution[:, j] if initial_distribution is not None else None,
                      gamma=gamma)
    plt.tight_layout()
    return fig
