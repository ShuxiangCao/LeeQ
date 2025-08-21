import numpy as np

from leeq import setup
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon


def apply_noise_to_data(readout_qubit: VirtualTransmon, data: np.ndarray):
    """
    Apply noise to the data.

    Parameters
    ----------
    readout_qubit: VirtualTransmon
        The readout qubit that we take the reference for.
    data: np.ndarray
        The data to apply noise to. Usually the data is the expectation value of the readout qubit.
    Returns
    -------
    np.ndarray
        The data with noise applied.
    """

    data = (data + 1) / 2

    # If sampling noise is enabled, simulate the noise
    if setup().status().get_param('Sampling_Noise'):
        # Get the number of shot used in the simulation
        shot_number = setup().status().get_param('Shot_Number')

        # generate binomial distribution of the result to simulate the
        # sampling noise
        data = np.random.binomial(
            shot_number, data) / shot_number

    quiescent_state_distribution = readout_qubit.quiescent_state_distribution
    standard_deviation = np.sum(quiescent_state_distribution[1:])

    random_noise_factor = 1 + np.random.normal(
        0, standard_deviation, data.shape)

    data = (2 * data - 1)

    random_noise_factor = 1 + np.random.normal(
        0, standard_deviation, data.shape)

    random_noise_sum = np.random.normal(
        0, standard_deviation / 2, data.shape)

    data = np.clip(
        data * (0.5 - quiescent_state_distribution[0]) * 2 * random_noise_factor + random_noise_sum, -1, 1)

    return data
