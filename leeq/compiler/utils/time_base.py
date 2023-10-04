import numpy as np


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
