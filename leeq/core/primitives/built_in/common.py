import uuid

from leeq.core.primitives import LogicalPrimitive


class DelayPrimitive(LogicalPrimitive):
    """
    Logical primitive for delay.

    Example parameters:
    ```
    {
    'type': 'Delay',
    'delay_time': 0.1,
    'hardware_stall': True
    }
    """

    def __init__(self, parameters, name=None):
        """
        Initialize the logical primitive for indicating a delay.

        Parameters:
            name (str): The name of the logical primitive.
            parameters (dict): The parameters of the logical primitive.
        """

        self._validate_parameters(parameters)

        if name is None:
            name = f"Delay_{parameters['delay_time']}:{uuid.uuid4()}"

        super().__init__(name, parameters)

    @staticmethod
    def _validate_parameters(parameters: dict):
        if "delay_time" not in parameters:
            raise ValueError("The delay time is not specified.")
        if "hardware_stall" not in parameters:
            raise ValueError("Whether to use hardware stall is not specified.")

    def get_delay_time(self):
        """
        Get the delay time.

        Returns:
            float: The delay time.
        """
        return self._parameters["delay_time"]

    def set_delay(self, delay: float):
        """
        Set the delay time. For compatibility reasons.

        Parameters:
            delay (float): The delay time.
        """
        self.update_parameters(delay_time=delay)


class Delay(object):
    """
    For compatibility reasons, a factory class for DelayPrimitive is provided.
    """

    def __new__(cls, delay: float, hardware_stall=True):
        """
        Create a delay primitive.

        Parameters:
            delay (float): The delay time.
            hardware_stall (bool): Whether to use hardware stall.

        Returns:
            DelayPrimitive: The delay primitive.
        """
        return DelayPrimitive(
            parameters={"delay_time": delay, "hardware_stall": hardware_stall}
        )


class PhaseShift(LogicalPrimitive):
    """
    Logical primitive for phase shift on all subsequent signals. This is mainly for implementing virtual-z gate
    on superconducting circuits.
    """

    def __init__(self, name, parameters):
        """
        Initialize the logical primitive for indicating a phase shift.

        Parameters:
            name (str): The name of the logical primitive.
            parameters (dict): The parameters of the logical primitive.

        Example parameters:
        ```
        {
        'type': 'PhaseShift',
        'channel' : 1, # Which channel to apply the phase shift
        'phase_shift': np.pi/3, # The phase shift value
        'transition_multiplier': { # The transition multiplier for the phase shift
            'f01': 1,
            'f12': -1,
            'f23': 1
        }
        ```

        Note that the actual shift on each transition is phase_shift * transition_multiplier[transition_name]. This
        design is for fast implementation of qudit virtual z gates which you need to shift the phase of multiple
        transitions at the same time.

        When the transition name is something like 'f02', which indicates a multi-photon transition, the phase shift
        will be divided by the number of transition photons to make it logically corresponds to the shift the user
         would like to obtain. This feature needs to be implemented at the compiler level.
        """
        self._virtual_width = 0  # The virtual width of the phase shift, which is used to calculate the phase shift
        # given a Z hamiltonian term.
        self._omega = 0  # The strength of the Z hamiltonian term.

        super().__init__(name, parameters)

    @staticmethod
    def _validate_parameters(parameters: dict):
        if "channel" not in parameters:
            raise ValueError("The channel is not specified.")
        if "phase_shift" not in parameters:
            raise ValueError("The phase is not specified.")
        if "transition_multiplier" not in parameters:
            raise ValueError("The transition multiplier is not specified.")
        if not isinstance(parameters["transition_multiplier"], dict):
            raise TypeError("The transition multiplier is not a dict.")

    def set_virtual_width(self, width: float):
        """
        Set the virtual width of the phase shift.

        Parameters:
            width (int): The virtual width.
        """
        self._virtual_width = width
        self.update_parameters(
            phase_shift=-self._omega * self._virtual_width
        )

    def set_omega(self, omega: float):
        """
        Set the strength of the Z hamiltonian term.
        """
        self._omega = omega
        self.update_parameters(
            phase_shift=-self._omega * self._virtual_width
        )
