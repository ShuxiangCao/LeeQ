import copy
import numpy as np


def to_dense_probabilities(data_measured, base=2):  # [qubit index,sample index,anything else]
    """
    Convert the measured data to dense probabilities.

    Parameters:
        data_measured (np.ndarray): The data retrieved from the FPGA.
        [qubit index,shot index,anything else...]

        base (int): The base of the data. Qubit is 2, qudit is 3, etc.

    Returns:
        np.ndarray: The dense probabilities.
    """

    data_measured = copy.copy(data_measured)

    qubit_count = data_measured.shape[0]
    aggregated_data = data_measured[0]

    for i in range(1, qubit_count):
        aggregated_data += base ** i * data_measured[i]

    bins = np.asarray([np.sum((aggregated_data == i).astype(int), axis=0) for i in range(base ** qubit_count)])

    return bins / data_measured.shape[1]
