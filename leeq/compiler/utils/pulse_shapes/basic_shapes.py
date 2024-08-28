import inspect

import numpy as np  # Assuming numpy is needed based on the provided function.
from typing import Optional, Tuple, List
from leeq.compiler.utils.time_base import get_t_list

__all__ = [
    "blackman",
    "blackman_drag",
    "clear_square",
    "gaussian",
    "gaussian_drag",
    "soft_square",
    "square",
    "clk",
    "segment_by_time",
    "segment_by_index",
    "blackman_square",
    "arbitrary_pulse"
]


def segment_by_time(sampling_rate: int, env_name: str, seg_t_start: float, seg_t_end: float, **kwargs) -> np.array:
    """
    Generates a complex wave packet that is a segment of another wave packet.
    This function is for bypassing the restriction of the pulse shape function
    to have a limited width.

    Parameters:
    - sampling_rate (int): The sampling rate in Megasamples per second.
    - env_name (str): The name of the pulse shape.
    - seg_t_start (float): The starting time of the segment.
    - seg_t_end (float): The ending time of the segment.
    - kwargs: The parameters of the pulse shape.
    """

    from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory
    pulse_shape_factory = PulseShapeFactory()
    env_func = pulse_shape_factory.get_pulse_shape_function(env_name)

    # Filter the kwargs so that only the required parameters are passed to
    # the function
    required_parameters = inspect.signature(env_func).parameters

    kwargs_call = {key: kwargs[key]
                   for key in required_parameters if key in kwargs}

    print(kwargs_call)
    # Get the array of the entire pulse shape
    env_array = env_func(sampling_rate=sampling_rate, **kwargs_call)

    dt = 1 / sampling_rate
    t = np.arange(0, len(env_array) * dt, dt)

    seg_array = env_array[(t >= seg_t_start) & (t < seg_t_end)]
    print(seg_t_start, seg_t_end)
    print(seg_array.shape)

    return seg_array


def segment_by_index(sampling_rate: int, env_name: str, seg_i_start: float, seg_i_end: float, **kwargs) -> np.array:
    """
    Generates a complex wave packet that is a segment of another wave packet.
    This function is for bypassing the restriction of the pulse shape function
    to have a limited width.

    Parameters:
    - sampling_rate (int): The sampling rate in Megasamples per second.
    - env_name (str): The name of the pulse shape.
    - seg_i_start (float): The starting time of the segment.
    - seg_i_end (float): The ending time of the segment.
    - kwargs: The parameters of the pulse shape.
    """

    from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory
    pulse_shape_factory = PulseShapeFactory()
    env_func = pulse_shape_factory.get_pulse_shape_function(env_name)

    # Filter the kwargs so that only the required parameters are passed to
    # the function
    required_parameters = inspect.signature(env_func).parameters

    kwargs_call = {key: kwargs[key]
                   for key in required_parameters if key in kwargs}

    # Get the array of the entire pulse shape
    env_array = env_func(sampling_rate=sampling_rate, **kwargs_call)

    dt = 1 / sampling_rate
    t = np.arange(0, len(env_array) * dt, dt)

    seg_array = env_array[seg_i_start: seg_i_end]

    return seg_array


def gaussian(
        sampling_rate: int,
        amp: float,
        width: float,
        phase: float = 0,
        trunc: float = 1.2) -> np.array:
    """
    Generates a complex Gaussian wave packet.

    Parameters:
    - sampling_rate (int): The sampling rate in Megasamples per second.
    - amp (float): Amplitude of the Gaussian wave.
    - phase (float): Initial phase of the wave in radians.
    - width (float): Width of the Gaussian envelope in microseconds.
    - trunc (float): A multiplier of the width to truncate the Gaussian.

    Returns:
    np.array: A complex-valued array representing the Gaussian wave packet.
    """

    # Calculate the Gaussian width, which is half of the provided width.
    gauss_width = width / 2.0

    # Get the time base using the sampling_rate and pulse_width (width*trunc)
    t = get_t_list(sampling_rate, width * trunc)

    # Generate and return the Gaussian wave packet.
    # You may extend the function to incorporate it in the future.
    return (
            amp
            * np.exp(1.0j * phase)
            * np.exp(-(((t - gauss_width) / gauss_width) ** 2)).astype("complex64")
    )


def gaussian_drag(
        sampling_rate: int,
        amp: float = 1.0,
        phase: float = 0.0,
        width: float = 1.0,
        alpha: float = 1.0,
        trunc: float = 1.0,
) -> np.ndarray:
    """
    Generate a Gaussian DRAG pulse shape.

    Parameters
    ----------
    sampling_rate : int
        The sampling rate in Megasamples per second.
    amp : float, optional
        The amplitude of the Gaussian pulse, by default 1.0.
    phase : float, optional
        The phase of the Gaussian pulse in radians, by default 0.0.
    width : float, optional
        The width of the Gaussian pulse in microseconds, by default 1.0.
    alpha : float, optional
        The DRAG coefficient, by default 1.0.
    trunc : float, optional
        Truncation of the pulse width (multiplier), by default 1.0.

    Returns
    -------
    np.ndarray
        The generated complex pulse shape.
    """
    # Multiply alpha by 2.0 for usage in derivative calculation
    alpha2 = alpha * 2.0

    # Calculate half of the Gaussian width
    gauss_width = width / 2.0

    # Retrieve timebase
    t = get_t_list(sampling_rate, width * trunc)

    # Compute the real part of the pulse shape using a Gaussian function
    shape = np.exp(-(((t - gauss_width) / gauss_width) ** 2), dtype="cfloat")

    # Compute the imaginary part using the derivative of the Gaussian
    shape.imag = shape.real * 2.0 * t / \
                 (2.0 * np.pi * alpha2) / gauss_width ** 2

    # Return the pulse shape, with applied amplitude and phase modulation
    return amp * np.exp(1.0j * phase) * shape


def blackman(
        sampling_rate: int,
        amp: float = 1.0,
        phase: float = 0.0,
        width: float = 1.0,
        trunc: float = 1.0,
) -> np.ndarray:
    """
    Generate a modified Blackman window function.

    Parameters:
    - sampling_rate (int): The sampling rate in Megasamples per second.
    - amp (float, optional): The amplitude of the window. Defaults to 1.0.
    - phase (float, optional): The phase of the window. Defaults to 0.0.
    - width (float, optional): The width of the window in microseconds. Defaults to 1.0.
    - trunc (float, optional): Truncation of the window. Defaults to 1.0.

    Returns:
    - np.ndarray: The complex-valued modified Blackman window function.
    """

    # Coefficients used in the Blackman window formula
    a0 = 7938.0 / 18608.0
    a1 = 9240.0 / 18608.0
    a2 = 1430.0 / 18608.0

    # Obtain the timebase
    t = get_t_list(sampling_rate, width)
    x = get_t_list(sampling_rate, width * trunc)

    # Calculate the offset for centering the Blackman window
    offset = (len(x) - len(t)) // 2

    # Initialize a complex array for shape
    shape = np.zeros(shape=x.shape, dtype="cfloat")

    # Create a midshape slice for modification
    midshape = shape[offset: offset + len(t)]

    # Update the real part of midshape based on the Blackman window formula
    midshape += 1 - (
            a0
            - a1 * np.cos((2.0 * np.pi * (t + width / 2.0)) / width, dtype="cfloat")
            + a2 * np.cos(4.0 * np.pi * (t + width / 2.0) / width, dtype="cfloat")
    )

    # Return the scaled and phased shape
    return amp * np.exp(1.0j * phase) * shape


def blackman_drag(
        sampling_rate: int,
        width: float,
        amp: float = 1.,
        phase: float = 0.,
        alpha: float = 1e9,  # 1e9 basically means no drag correction
        trunc: float = 1.0,
) -> np.ndarray:
    """
    Generate a Blackman DRAG pulse using the specified parameters.

    Parameters
    ----------
    sampling_rate : int
        The sampling rate in Megasamples per second.
    amp : float
        Amplitude of the pulse.
    phase : float, optional
        Phase of the pulse, default is None.
    width : float, optional
        Width of the pulse in microseconds, default is None.
    alpha : float, optional
        An alpha parameter for the DRAG pulse, default is None.
    trunc : float, optional
        Truncation of the pulse, default is 1.0.

    Returns
    -------
    np.ndarray
        A NumPy array representing the Blackman DRAG pulse.

    Notes
    -----
    Ensure that `get_t_list` function is defined with appropriate logic as per use case.
    """
    # Validation of parameters might be needed, such as ensuring non-null
    # values for width and alpha

    # Coefficients for Blackman function
    a0 = 7938.0 / 18608.0
    a1 = 9240.0 / 18608.0
    a2 = 1430.0 / 18608.0

    # Get timebase array for the given width
    t = get_t_list(sampling_rate, width)

    # Get truncated timebase array for width*trunc
    x = get_t_list(sampling_rate, width * trunc)

    # Prepare shape array of complex float type with zeroed elements
    shape = np.zeros(shape=x.shape, dtype="cfloat")

    # Compute offset for positioning in the middle of shape array
    offset = (len(x) - len(t)) // 2

    # Intermediate shape array for calculations
    midshape = shape[offset: offset + len(t)]

    # Apply the Blackman function
    midshape += 1 - (
            a0
            - a1 * np.cos((2.0 * np.pi * (t + width / 2.0)) / width, dtype="cfloat")
            + a2 * np.cos(4.0 * np.pi * (t + width / 2.0) / width, dtype="cfloat")
    )

    # Handle imaginary part with a DRAG correction term
    alpha2 = alpha * 2.0
    midshape.imag = -(
            a1 * 2.0 * np.pi / width * np.sin((2.0 * np.pi * (t + width / 2.0)) / width)
            - a2 * 4.0 * np.pi / width * np.sin(4.0 * np.pi * (t + width / 2.0) / width)
    ) / (2.0 * np.pi * alpha2)

    # Return the pulse shaped with phase and amplitude
    env = amp * np.exp(1.0j * phase) * shape
    return env


def blackman_square(
        sampling_rate: int,
        amp: float = 1.0,
        phase: float = 0.0,
        width: float = 1.0,
        trunc: float = 1.0,
        rise: float = 0.02,
        alpha: float = 1e9,  # 1e9 basically means no drag correction
) -> np.ndarray:
    """
    Generate a modified Blackman square window function.

    Parameters:
    - sampling_rate (int): The sampling rate in Megasamples per second.
    - amp (float, optional): The amplitude of the window. Defaults to 1.0.
    - phase (float, optional): The phase of the window. Defaults to 0.0.
    - width (float, optional): The width of the window in microseconds. Defaults to 1.0.
    - trunc (float, optional): Truncation of the window. Defaults to 1.0.
    - rise (float, optional): Rise time of the signal. Defaults to 0.02.
    - alpha (float, optional): An alpha parameter for the DRAG pulse. Defaults to 1e9.

    Returns:
    - np.ndarray: The complex-valued modified Blackman window function.
    """

    if rise == 0:
        return square(sampling_rate=sampling_rate, amp=amp, width=width, phase=phase)

    # Coefficients used in the Blackman window formula
    a0 = 7938.0 / 18608.0
    a1 = 9240.0 / 18608.0
    a2 = 1430.0 / 18608.0

    # Obtain the timebase
    t = get_t_list(sampling_rate, width)
    t_rise = get_t_list(sampling_rate, rise * 2)
    x_rise = get_t_list(sampling_rate, rise * trunc * 2)

    # Calculate the offset for centering the Blackman window
    offset = (len(x_rise) - len(t_rise)) // 2

    # Initialize a complex array for shape
    shape = np.zeros(shape=x_rise.shape, dtype="cfloat")

    # Create a midshape slice for modification
    midshape = shape[offset: offset + len(t_rise)]

    shape[offset: offset + len(t_rise)] = blackman_drag(sampling_rate=sampling_rate, amp=1, phase=0, width=rise * 2,
                                                        alpha=alpha, trunc=trunc)

    flat_top = np.ones(shape=t.shape, dtype="cfloat")

    half_rise = int(len(shape) / 2)
    final_shape = np.concatenate([shape[:half_rise], flat_top, shape[half_rise:]])

    # Return the scaled and phased shape
    final_shape = amp * np.exp(1.0j * phase) * final_shape

    return final_shape


def soft_square(
        sampling_rate: int,
        amp: float,
        width: float,
        phase: Optional[float] = 0,
        rise: Optional[float] = 0,
        trunc: Optional[float] = 1,
        delay: float = 0.0,
        phase_shift: float = 0,
        ex_delay: float = 0.0,
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

    assert ex_delay == 0, "Extra delay is not supported for soft square wave."

    if rise == 0:
        return square(sampling_rate=sampling_rate, amp=amp, width=width, phase=phase, delay=delay)

    full_width = width + 2. * rise * trunc + ex_delay
    t = get_t_list(sampling_rate, full_width + delay)

    t -= 0.5 * delay

    part_1 = np.tanh((t + 0.5 * width - full_width / 2) / rise)
    part_2 = np.tanh((t - 0.5 * width - full_width / 2) / rise)

    y = amp * np.exp(1.0j * (phase + phase_shift)) * 0.5 * (part_1 - part_2)

    if ex_delay > 0:
        extra_delay_y = get_t_list(sampling_rate, delay)
        y = np.concatenate([extra_delay_y, y])

    return y


def square(
        sampling_rate: int,
        amp: float,
        width: Optional[float] = None,
        phase: Optional[float] = 0,
        delay: float = 0.0,
        phase_shift: float = 0,
        dc_bias: float = 0,
        ex_delay: float = 0,
) -> np.ndarray:
    """
    Generate a square wave.

    Parameters:
    - sampling_rate (int): Sampling rate in Megasamples per second.
    - amp (float): Amplitude of the signal.
    - phase (float, optional): Phase of the signal.
    - width (float, optional): Width of the pulse.
    - delay (float, optional): Delay of the signal.
    - phase_shift (float, optional): Phase shift of the signal.
    - dc_bias (float, optional): DC bias of the signal.
    - ex_delay (float, optional): Extra delay of the signal.

    Returns:
    - np.ndarray: Generated square wave.
    """
    delay += ex_delay
    x = get_t_list(sampling_rate, width + delay)
    y = np.empty(shape=(len(x),), dtype="cfloat")
    y.fill(amp * np.exp(1.0j * (phase + phase_shift)))
    y[x < (delay - width) / 2.0] = 0.0
    return y + dc_bias


def clear_square(
        sampling_rate: int,
        amp: float,
        phase: Optional[float] = None,
        width: Optional[float] = None,
        delay: float = 0.0,
        phase_shift: float = 0,
        dc_bias: float = 0,
        ini_top: float = 0.2,
        ini_bot: float = 0.3,
        final_top: float = 0.8,
        final_bot: float = -0.8,
        ini_width: float = 0.2,
        final_width: float = 0.2,
        ex_delay: float = 0,
) -> np.ndarray:
    """
    Generate a clear square wave, which is used to fast reset the resonator population.
    See more details at https://doi.org/10.1103/PhysRevApplied.5.011001.

    Parameters:
    - sampling_rate (int): Sampling rate in Megasamples per second.
    - amp (float): Amplitude of the signal.
    - phase (float, optional): Phase of the signal.
    - width (float, optional): Width of the pulse.
    - delay (float, optional): Delay of the signal.
    - phase_shift (float, optional): Phase shift of the signal.
    - dc_bias (float, optional): DC bias of the signal.
    - ini_top (float, optional): Initial top level.
    - ini_bot (float, optional): Initial bottom level.
    - final_top (float, optional): Final top level.
    - final_bot (float, optional): Final bottom level.
    - ini_width (float, optional): Initial width.
    - final_width (float, optional): Final width.
    - ex_delay (float, optional): Extra delay of the signal.

    Returns:
    - np.ndarray: Generated clear square wave.
    """
    assert 1 <= ini_top
    assert 0 <= ini_bot <= 1
    assert final_bot <= 0
    assert final_top >= 0
    assert 0 <= ini_width <= 0.5
    assert 0 <= final_width <= 0.5

    total_width = width + delay + ex_delay + 2 * final_width * width
    x = get_t_list(sampling_rate, total_width)
    x += total_width / 2

    y = np.empty(shape=(len(x),), dtype="cfloat")
    y.fill(amp * np.exp(1.0j * (phase + phase_shift)))

    x_initial_top = delay
    x_initial_bottom = delay + ini_width * width
    x_amp = delay + ini_width * width * 2
    x_final_bottom = delay + width

    x_final_top = delay + width + final_width * width
    x_end = delay + width + 2 * final_width * width

    y[x < x_initial_top] = 0
    y[(x_initial_top < x) & (x < x_initial_bottom)] = ini_top * amp
    y[(x_initial_bottom < x) & (x < x_amp)] = ini_bot * amp
    y[(x_amp < x) & (x < x_final_bottom)] = amp
    y[(x_final_bottom < x) & (x < x_final_top)] = final_bot * amp
    y[(x_final_top < x) & (x < x_end)] = final_top * amp

    y = y
    return y

def arbitrary_pulse(
        sampling_rate: int,
        amp: float,
        pulse_window_array: List,
        phase: float = 0,
        width: Optional[float] = None,
)-> np.ndarray:
    """
Generate an arbitrary pulse shape using a window array.

Parameters:
- sampling_rate (int): Sampling rate in Megasamples per second.
- amp (float): Amplitude of the signal.
- pulse_window_array (np.ndarray): Array representing the pulse shape.
- phase (float, optional): Phase of the signal.
- width (float, optional): Width of the pulse, not being used.
"""

    pulse_shape = np.asarray(pulse_window_array)

    pulse_shape = pulse_shape[0, :] + 1.j * pulse_shape[1, :]
    pulse = amp * np.exp(1.0j * phase) * pulse_shape
    return pulse

def clk(
        sampling_rate: int,
        amp: float,
        width: Optional[float] = None,
        phase: Optional[float] = 0,
        delay: float = 0.0,
        phase_shift: float = 0,
        dc_bias: float = 0,
        ex_delay: float = 0,
        kappa: float = 0,
) -> np.ndarray:
    """
    Generate a decaying square wave, suitable for the transient part of the cloaking pulse.

    Parameters:
    - sampling_rate (int): Sampling rate in Megasamples per second.
    - amp (float): Amplitude of the signal.
    - phase (float, optional): Phase of the signal.
    - width (float, optional): Width of the pulse.
    - delay (float, optional): Delay of the signal.
    - phase_shift (float, optional): Phase shift of the signal.
    - dc_bias (float, optional): DC bias of the signal.
    - ex_delay (float, optional): Extra delay of the signal.
    - kappa (float, optional): Resonator linewidth.

    Returns:
    - np.ndarray: Generated square wave.
    """
    delay += ex_delay
    x = get_t_list(sampling_rate, width + delay)
    y = np.empty(shape=(len(x),), dtype="cfloat")
    y.fill(amp * np.exp(1.0j * (phase + phase_shift)))
    y[x < (delay - width) / 2.0] = 0.0
    return y * np.exp(-kappa * x / 2) + dc_bias

