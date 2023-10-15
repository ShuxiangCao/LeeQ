import contextlib
from typing import Any, Optional, Dict

from labchronicle import log_event
from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import Sweeper
from leeq.utils import setup_logging, is_running_in_jupyter

logger = setup_logging(__name__)


class SetupStatusParameters(LeeQObject):
    """
    Stores the status and configuration of the setup, allows a shortcut to directly communicate with the
    instruments.

    This class is used to store the experiment parameters such as the short number, shot period, etc. It is also used
    to store the configuration of the instruments, such as the channel configuration, etc.
    """

    pass

    def __init__(self, name):
        """
        Initialize the SetupStatusParameters class.

        Parameters:
            name (str): The name of the setup status.
        """
        super().__init__(name)
        self._internal_dict = {}
        self._channel_dict = {}
        self._channel_callbacks = {}

    def set_param(self, key: str, value: Any):
        """
        Same as set_parameter, for compatibility reasons.

        Parameters:
             key (str): The name of the parameter.
             value (Any): The value of the parameter.

        """
        return self.set_parameter(key=key, value=value)

    @log_event
    def set_parameter(self, key: str, value: Any):
        """
         Set a parameter from the setup status.

         Parameters:
             key (str): The name of the parameter.
             value (Any): The value of the parameter.

        """

        key = key.lower()

        if key in self._internal_dict:
            self._internal_dict[key] = value
        else:
            msg = f"{self._name} may not accept parameter {key}."
            logger.warning(msg)

    def set_parameters(self, **kwargs: dict):
        """
        Set a set of parameters in the setup status.
        Parameters:
            kwargs (dict): The key arguments to set.
        """

        for key, val in kwargs.items():
            self.set_parameter(key, val)

    def add_parameter(self, key, default_value):
        """
        Add a parameter to the setup status.

        Parameters:
            key (str): The name of the parameter.
            default_value (Any): The value of the parameter.
        """
        key = key.lower()
        self._internal_dict[key] = default_value

    def add_param(self, key, default_value):
        """
        Same as add_parameter, for compatibility.

        Parameters:
            key (str): The name of the parameter.
            default_value (Any): The value of the parameter.
        """
        self.add_parameter(key, default_value)

    def get_parameters(self, key: str = None):
        """
        Get the parameter from the setup status.

        Parameters:
            key (str): The name of the parameter.

        Returns:
            Any: The value of the parameter.
        """
        if key is None:
            return self._internal_dict.copy()

        key = key.lower()
        return self._internal_dict[key]

    def get_params(self):
        """
        Same as get_parameters, for compatibility.

        Returns:
            Any: The value of the parameter.
        """
        return self.get_parameters()

    def get_param(self, key: str):
        """
        Same as get_parameters, for compatibility.

        Parameters:
            key (str): The name of the parameter.

        Returns:
            Any: The value of the parameter.
        """
        return self.get_parameters(key)

    def add_channel(self, channel, **kwargs):
        """
        Add a channel to the setup status.

        Parameters:
            channel (str): The name of the channel.
            kwargs (dict): The keyword arguments to be passed to the channel.
        """

        if channel in self._channel_dict:
            msg = f"{self._name} already configured channel {channel}."
            logger.error(msg)
            raise ValueError(msg)

        self._channel_dict[channel] = kwargs
        self._channel_callbacks[channel] = None

    def set_channel_parameters(self, channel, **kwargs):
        """
        Set the channel parameters.
        """

        # Make all the kwargs keys lowercase
        new_kwargs = dict([(k.lower(), v) for k, v in kwargs.items()])

        if channel not in self._channel_dict:
            msg = f"{self._name} does not have channel {channel}."
            logger.error(msg)
            raise ValueError(msg)

        self._channel_dict[channel].update(new_kwargs)

    def set_channel_params(self, channel, **kwargs):
        """
        Same as set_channel_parameters, for compatibility.
        """
        self.set_channel_parameters(channel, **kwargs)

    def get_channel_parameters(self, channel, key=None):
        """
        Get the channel parameters.
        """

        if channel not in self._channel_dict:
            msg = f"{self._name} does not have channel {channel}."
            logger.error(msg)
            raise ValueError(msg)

        if key is None:
            return self._channel_dict[channel].copy()

        key = key.lower()

        if key not in self._channel_dict[channel]:
            msg = f"{self._name} does not have channel {key}."
            logger.error(msg)
            raise ValueError(msg)

        return self._channel_dict[channel][key]

    def get_channel_params(self, channel: int):
        """
        Same as get_channel_parameters, for compatibility.

        Parameters:
            channel (int): The channel id.

        Return:
            dict: Channel parameters
        """
        return self.get_channel_parameters(channel=channel)

    def get_channel_param(self, key):
        """
        Same as get_channel_parameters, for compatibility.
        """
        return self.get_channel_parameters(key)

    @contextlib.contextmanager
    def with_parameters(self, **kwargs: Dict):
        """
        Set a set of parameters, and recover them after execution
        """
        original_parameters = self.get_parameters()
        self.set_parameters(**kwargs)
        yield
        self.set_parameters(**original_parameters)

    def register_compile_lpb_callback(self, channel: int, callback: Optional[callable] = None):
        """
        Register a callback function that will be called when the compiler
        compiles the logical primitive block. If the callback is None, the callback will be removed.

        This functionality allows the setup to make changes to the lpb parameters before them get compiled to
        the instructions for backends. For instance, we could be using a frequency mixing on a specific channel,
        therefore the driving frequency of the pulse needs to be subtracted by the mixing frequency (or added, based
        on the sideband choice).

        The callback function should accept the following parameters:
            parameters (dict): The parameter of the lpb.
        And returns the following parameters:
            parameters (dict): The updated parameter of the lpb.

        Parameters:
            channel (int): The channel id.
            callback (callable): The callback function.
        """

        if channel not in self._channel_dict:
            msg = f"{self._name} does not have channel {channel}."
            logger.error(msg)
            raise ValueError(msg)

        if not callable(callback):
            msg = f"Callback is not callable."
            logger.error(msg)
            raise ValueError(msg)

        if self._channel_callbacks[channel] is not None:
            msg = f"Callback for channel {channel} has now been overwritten."
            logger.warning(msg)

        self._channel_callbacks[channel] = callback

    def get_modified_lpb_parameters_from_channel_callback(self, channel: int, parameters: dict):
        """
            Get the modified lpb parameters from the channel callback. If the callback is not registered,
            return the original parameters.

            Parameters:
                channel (int): The channel id.
                parameters (dict): The original parameters.

            Returns:
                dict: The modified parameters.
        """
        if channel not in self._channel_dict:
            msg = f"{self._name} does not have channel {channel}."
            logger.error(msg)
            raise ValueError(msg)

        if self._channel_callbacks[channel] is None:
            return parameters

        return self._channel_callbacks[channel](parameters)


class ExperimentalSetup(LeeQObject):
    """
    The ExperimentalSetup class is used to template the experimental setup.
    """

    def __init__(self, name):
        """
        Initialize the ExperimentalSetup class. Use init to define all the equipments in the setup.
        """
        assert (
                "_compiler" in self.__dict__
        ), "The compiler is not defined in the setup, please define it in the __init__."
        assert (
                "_engine" in self.__dict__
        ), "The engine is not defined in the setup, please define it in the __init__."
        super().__init__(name)
        self._active = False
        self._status = SetupStatusParameters(name + ".status")

        # Add parameters that all setups should have
        # The number of shots to be taken for each point
        self._status.add_param("Shot_Number", 2000)
        # The period between each shot, usually should choose more than 3 T1
        self._status.add_param("Shot_Period", 500.0)
        # The type of data to be acquired, could be IQ, IQ_average, traces,
        # etc.
        self._status.add_param("Acquisition_Type", "IQ")
        # Plot the pulse sequence in the plotter
        self._status.add_param("Debug_Plotter", False)
        self._status.add_param(
            "Debug_Plotter_Ignore_Readout", False
        )  # Ignore the readout in the plotter
        # Whether the code is running in Jupyter
        self._status.add_param("In_Jupyter", is_running_in_jupyter())
        # Whether to plot the result in Jupyter
        self._status.add_param("Plot_Result_In_Jupyter", True)
        # Whether to show the AMP warning
        self._status.add_param("AMP_Warning", True)
        # Whether to ignore the plot error
        self._status.add_param("Ignore_Plot_Error", True)
        # The time to wait before auto releasing the resources (disconnect the
        # devices)
        self._status.add_param("ResourceAutoReleaseTime", 300)
        # Whether to auto release the resources
        self._status.add_param("ResourceAutoRelease", True)
        # Whether to disable the sweep progress bar
        self._status.add_param("DisableSweepProgressBar", False)
        # Whether to leave the sweep progress bar
        self._status.add_param("LeaveSweepProgressBar", True)
        # The description of the sweep progress bar
        self._status.add_param("SweepProgressBarDesc", "Sweep")
        # The global logical primitive block that runs before any expreiment, #
        # could be used for active reset for initial state preparation.
        self._status.add_param("GlobalPreLPB", None)
        # The global logical primitive block that attach to all runs, # could
        # be used for active reset for fast cooling.
        self._status.add_param("GlobalPostLPB", None)

    def get_live_status(self):
        """
        Get a gist of the current status of the setup for the live monitor.

        It returns the general configuration of the setup, such as the number of shots, the shot period, etc.
        It also contains the status of the sweep engine.
        """

        status = {
            "setup_name": self._name,
            "shot_number": self._status.get_parameters('Shot_Number'),
            "shot_period": self._status.get_parameters('Shot_Period'),
        }

        return {
            'setup_status': status,
            'engine_status': self._engine.get_live_status(),
        }

    @property
    def status(self):
        """
        Get the status of the setup.
        """
        return self._status

    @log_event
    def connect_devices(self):
        """
        Connect all the devices in the setup.
        """
        self._active = True

    @log_event
    def disconnect_devices(self):
        """
        Disconnect all the devices in the setup.
        """
        self._active = False

    def is_active(self):
        """
        Check if the setup is active.
        """
        return self._active

    def push_instrument_settings(self):
        """
        Push the instrument settings to the instruments.
        """
        raise NotImplementedError()

    def pull_instrument_settings(self):
        """
        Pull the instrument settings from the instruments.
        """
        raise NotImplementedError()

    def run(self, lpb: LogicalPrimitiveBlock, sweep: Sweeper):
        """
        Run the experiment.

        Note that the setup mainly maintains the instruments, so the compiling from lpb to instructions should be done
        by the compiler.
        """
        raise NotImplementedError()

    def update_setup_parameters(self, instructions):
        """
        Update the setup parameters of the compiler. It accepts the compiled instructions from the compiler, and update
        the local cache first. then use push_instrument_settings to push the settings to the instruments.

        Parameters:
            instructions (Any): The instructions to be executed.
        """
        raise NotImplementedError()

    def fire_experiment(self, instructions):
        """
        Fire the experiment and wait for it to finish.

        Parameters:
            instructions (Any): The instructions to be executed.
        """
        raise NotImplementedError()

    def collect_data(self, instructions):
        """
        Collect the data from the compiler and commit it to the measurement primitives.

        Parameters:
            instructions (Any): The instructions to be executed.
        """
        raise NotImplementedError()

    @property
    def name(self):
        """
        The name of the setup.
        """
        return self._name
