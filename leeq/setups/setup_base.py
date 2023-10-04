from labchronicle import log_event

from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import Sweeper


class ExperimentalSetup(LeeQObject):
    """
    The ExperimentalSetup class is used to template the experimental setup.
    """

    def __init__(self, name):
        """
        Initialize the ExperimentalSetup class. Use init to define all the equipments in the setup.
        """
        assert '_compiler' in self.__dict__, 'The compiler is not defined in the setup, please define it in the __init__.'
        assert '_engine' in self.__dict__, 'The engine is not defined in the setup, please define it in the __init__.'
        super().__init__(name)
        self._active = False

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
