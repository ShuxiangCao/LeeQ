import datetime
import inspect

from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveCombinable
from leeq.experiments.sweeper import Sweeper
from leeq.setups.setup_base import SetupStatusParameters
from leeq.utils import Singleton, setup_logging

logger = setup_logging(__name__)


class ExperimentManager(Singleton):
    """
    The ExperimentManager class is used to manage all the experiments in the program. It is used as a singleton.
    """

    def __init__(self):
        """
        Initialize the ExperimentManager class.
        """

        if not self._initialized:
            self._setups = {}
            self._default_setup = None

        super().__init__()

    def register_setup(self, setup, set_as_default=True):
        """
        Register a setup to the experiment manager.
        """
        if setup.name in self._setups:
            msg = f"Setup with name {setup.name} already exists."
            logger.error(msg)
            raise RuntimeError(msg)

        self._setups[setup.name] = setup

        if set_as_default:
            self.set_default_setup(setup.name)

    def set_default_setup(self, name):
        """
        Set the default setup.
        """
        if name not in self._setups:
            msg = f"Setup with name {name} does not exist."
            logger.error(msg)
            raise RuntimeError(msg)

        self._default_setup = name

    def get_default_setup(self):
        """
        Get the default setup.
        """

        if self._default_setup is None:
            return None
            #msg = f"Default setup is not set. Available setups are {self.get_available_setup_names()}"
            #logger.error(msg)
            #raise RuntimeError(msg)

        return self.get_setup(self._default_setup)

    def get_setup(self, name):
        """
        Get the setup by name.
        """
        return self._setups[name]

    def get_available_setup_names(self):
        """
        Get the names of all available setups.
        """
        return list(self._setups.keys())

    def run(self, *args, **kwargs):
        """
        Run an experiment.
        """
        return self.get_default_setup().run(*args, **kwargs)

    def status(self) -> SetupStatusParameters:
        """
        Get the status of the active setup.
        """
        return self.get_default_setup().status

    def clear_setups(self):
        """
        Clear all the setups.
        """
        self._setups = {}
        self._default_setup = None


def setup():
    """
    Get the setup manager. A shortcut for compatibility.
    """
    return ExperimentManager()


def basic_run(lpb: LogicalPrimitiveCombinable, swp: Sweeper, basis: str):
    """
    A shortcut for compatibility. In this version we ignore the basis, as it should be specified in the transformation
    function.
    """
    return setup().run(lpb, swp)


class Experiment(LeeQObject):
    """
    Base class for all experiments.

    An experiment contains the script to execute the experiment, analyze the data and visualize the result.

    1. Scripts execution
        To allow labchronicle to log the experiment, the main experiment script should be written in the `run` method.
        The `run` method should be decorated with the `labchronicle.log_and_record` method to log the events in the
         experiment. The decorator will save the arguments and return values of the `run` method and the entire object
          to the log.
        The `run` method will always be executed at the end of `__init__` to start the experiment.
        This mechanism is mostly for compatibility reasons.

    2. Data analysis
        The data analysis can be written in any arbitrary method, and suggested to run in a separate method than `run`,
         ideally the first few lines of the visualization code. This is because if the data analysis failed, it may
         crash the program before labchronicle can log the experiment data.

    3. Visualization
        The visualization code should live in a separate function and decorated by `labchronicle.browser_function`, so
         that the function will be executed when the experiment is finished execution in Jupyter notebook. It also
         allows the function to be executed later when data loaded from the log file.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the experiment.
        """
        super(
            Experiment, self).__init__(
            name=f"Experiment: {self.__class__.__name__}")

        # Run the experiment
        self.run(*args, **kwargs)

        # Check if we need to plot
        if setup().status().get_parameters("Plot_Result_In_Jupyter"):
            # Print the datetime of the experiment
            self.logger.info(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            for name, func in self.get_browser_functions():
                f_args, f_kwargs = (
                    func._browser_function_args,
                    func._browser_function_kwargs,
                )

                # For compatibility, select the argument that the function
                # accepts with inspect
                sig = inspect.signature(func)

                # Extract the parameter names that the function accepts
                valid_parameter_names = set(sig.parameters.keys())

                # Filter the kwargs
                filtered_kwargs = {
                    k: v for k, v in f_kwargs.items() if k in valid_parameter_names}

                try:
                    func(*f_args, **filtered_kwargs)
                except Exception as e:
                    self.logger.warning(
                        f"Error when executing {func.__qualname__} with parameters ({f_args},{f_kwargs}): {e}"
                    )
                    self.logger.warning(f"Ignore the error and continue.")
                    self.logger.warning(f"{e}")

    def run(self, *args, **kwargs):
        """
        The main experiment script. Should be decorated by `labchronicle.log_and_record` to log the experiment.
        """
        raise NotImplementedError()
