import functools

from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory
from leeq.compiler.utils.time_base import get_t_list


def get_qubic_envelope_name_from_leeq_name(leeq_pulse_shape_name: str):
    """
    Get the registered pulse shape name in Qubic from the pulse shape name in LeeQ.

    Parameters:
        leeq_pulse_shape_name (str): The pulse shape name in LeeQ.

    Returns:
        str: The pulse shape name in Qubic.
    """

    return 'leeq_' + leeq_pulse_shape_name


def register_leeq_pulse_shapes_to_qubic_pulse_shape_factory():
    """
    Register all the built-in pulse shapes in LeeQ to the Qubic pulse shape factory.
    """

    from qubic.pulse_factory import PulseShapeFactory as QubicPulseShapeFactory

    pulse_shape_factory_leeq = PulseShapeFactory()
    pulse_shape_factory_qubic = QubicPulseShapeFactory()

    for name in pulse_shape_factory_leeq.get_available_pulse_shapes():
        pulse_shape_factory_qubic.register_pulse_shape(
            get_qubic_envelope_name_from_leeq_name(name),
            wrap_envelope_leeq_function_to_qubic_format(name))


def wrap_envelope_leeq_function_to_qubic_format(pulse_shape_name: str):
    """
    Wrap the envelope function defined in leeq to the qubic format.

    Parameters:
        pulse_shape_name (str): The name of the pulse shape in LeeQ.

    Returns:
        callable: evaluated pulse shape function for QubiC.

    """

    env_func = PulseShapeFactory().get_pulse_shape_function(pulse_shape_name)

    @functools.wraps(env_func)
    def func(dt, **kwargs):
        """
        Evaluate the pulse shape function with the given parameters.
        Adpat the interface between qubic defined functions and leek defined functions.

        Parameters:
            dt (float): The sampling rate of the pulse shape. In Msps unit.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The time list and the pulse shape envelope.
        """

        # amp will be modified by the qubic system, so we always pass amp = 1
        kwargs = kwargs.copy()
        kwargs['amp'] = 1

        sampling_rate = 1 / dt / 1e6  # In Msps unit
        t = get_t_list(sampling_rate=sampling_rate, width=kwargs['width'])
        env = env_func(sampling_rate=sampling_rate, **kwargs)

        return t, env

    return func
