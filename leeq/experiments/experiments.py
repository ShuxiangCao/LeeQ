import datetime
import inspect

from labchronicle import Chronicle

from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveCombinable
from leeq.experiments.sweeper import Sweeper
from leeq.setups.setup_base import SetupStatusParameters
from leeq.utils import Singleton, setup_logging, display_json_dict
import leeq.experiments.plots.live_dash_app as live_monitor

logger = setup_logging(__name__)


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

        # Register the active experiment instance
        ExperimentManager().register_active_experiment_instance(self)

        try:
            # Run the experiment
            self.run(*args, **kwargs)
        finally:
            # Make sure we print the record details before throwing the exception
            if Chronicle().is_recording():
                # Print the record details
                record_details = self.retrieve_latest_record_entry_details(
                    self.run).copy()
                record_details.update({'print_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                display_json_dict(record_details, root=self.__class__.__qualname__, expanded=False)

        # Check if we need to plot
        if setup().status().get_parameters("Plot_Result_In_Jupyter"):
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

    def get_experiment_details(self):
        """
        Get the experiment details. It includes the labchronicle record details, experiment arguments, and the
        experiment specific details.

        Returns:
            dict: The experiment details.
        """

        return {
            "record_details": self.retrieve_latest_record_entry_details(self.run),
            "experiment_arguments": self.retrieve_args(self.run),
        }


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
            self._active_experiment_instance = None

        super().__init__()

    def start_live_monitor(self, **kwargs):
        """
        Start the live monitor.
        """
        live_monitor.start_app(experiment_manager=self, **kwargs)

    @staticmethod
    def stop_live_monitor():
        """
        Stop the live monitor.
        """
        live_monitor.stop_app()

    def get_live_status(self):
        """
        Get the live status of the running experiment.

        It collects the experiment details from the active experiment instance and the sweep status the setup.

        Returns:
            dict: The live status.
        """
        status = {'experiment_status': self._active_experiment_instance.get_experiment_details()}

        status.update(self.get_default_setup().get_live_status())

        return status

    def register_active_experiment_instance(self, instance: Experiment):
        """
        Register the active experiment instance.
        """

        assert isinstance(instance, Experiment), f"instance must be an instance of Experiment, got {type(instance)}"
        self._active_experiment_instance = instance

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
            # msg = f"Default setup is not set. Available setups are {self.get_available_setup_names()}"
            # logger.error(msg)
            # raise RuntimeError(msg)

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

        Here we assert the active instance has been registered.

        Parameters:
            *args: The arguments to pass to the experiment.
            **kwargs: The keyword arguments to pass to the experiment.

        Returns:
            Any: The return value of the experiment.
        """

        assert self._active_experiment_instance is not None, "No active experiment instance is registered."
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
