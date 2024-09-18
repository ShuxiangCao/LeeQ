import datetime
import inspect

import mllm
import numpy as np
import matplotlib
import plotly
from IPython.display import display
from labchronicle import Chronicle
from k_agents.experiment.experiment import Experiment as KExperiment
from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveCombinable
from leeq.experiments.sweeper import Sweeper
from leeq.setups.setup_base import SetupStatusParameters
from leeq.utils import Singleton, setup_logging, display_json_dict
import leeq.experiments.plots.live_dash_app as live_monitor

logger = setup_logging(__name__)

class LeeQAIExperiment(LeeQObject, KExperiment):
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

        LeeQObject.__init__(self, name=f"Experiment: {self.__class__.__name__}")
        KExperiment.__init__(self, *args, **kwargs)
        self._llm_logger = mllm.chat.ChatLogger(show_table=False)

        # Check the input arguments
        args, kwargs = self._check_arguments(self.bare_run, *args, **kwargs)
        self.run(*args, **kwargs)

    def _check_arguments(self, func, *args, **kwargs):
        """
        Check the arguments of the function.

        Parameters:
            func (callable): The function to check.
            args (list): The arguments of the function.
            kwargs (dict): The keyword arguments of the function.

        Returns:
            dict: The arguments of the function.
        """
        sig = inspect.signature(func)

        if 'ai_inspection' in kwargs and 'ai_inspection' not in sig.parameters:
            del kwargs['ai_inspection']

        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            msg = None
        except TypeError as e:
            msg = f"{e}\n\n"
            msg += f"Function signature: {sig}\n\n"
            msg += f"Documents:\n\n {self.bare_run.__doc__}\n\n"

        if msg is not None:
            raise TypeError(msg)

        return bound.args, bound.kwargs

    def _before_run(self, args, kwargs):
        KExperiment._before_run(self, args, kwargs)
        self._llm_logger.__enter__()
        # Register the active experiment instance
        setup().register_active_experiment_instance(self)


    def _post_run(self, args, kwargs):
        KExperiment._post_run(self, args, kwargs)
        if self.to_show_figure_in_notebook:
            self.show_plots()
        self.chronicle_log()
        self._llm_logger.__exit__(None, None, None)

    def chronicle_log(self):
        # Make sure we print the record details before throwing the
        # exception
        if Chronicle().is_recording():
            # Print the record details
            record_details = self.retrieve_latest_record_entry_details(
                self.bare_run)
            if record_details is not None:
                record_details = record_details.copy()
                record_details.update(
                    {'print_time': datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S")})
                display_json_dict(
                    record_details,
                    root=self.__class__.__qualname__,
                    expanded=False)
            else:
                msg = f"Failed to retrieve and save the record details for {self.bare_run.__qualname__}"
                logger.warning(msg)

    def get_experiment_details(self):
        """
        Get the experiment details. It includes the labchronicle record details, experiment arguments, and the
        experiment specific details.

        Returns:
            dict: The experiment details.
        """

        if self.bare_run.__qualname__ not in self._register_log_and_record_args_map:
            return {}

        kwargs = self.retrieve_args(self.bare_run)
        kwargs = {k: repr(v) for k, v in kwargs.items()}
        kwargs['name'] = self._name

        return {"record_details": self.retrieve_latest_record_entry_details(
            self.bare_run), "experiment_arguments": kwargs, }

    def show_plots(self):
        for name, func in self._get_plot_functions():
            if not func.__dict__.get('_browser_function', False):
                continue
            try:
                self._execute_single_plot_function(func)
                result = self._plot_function_result_objs[func.__qualname__]
            except Exception as e:
                self.log_warning(
                    f"Error when executing the browsable plot function {name}:{e}."
                )
                self.log_warning(f"Ignore the error and continue.")
                self.log_warning(f"{e}")
                continue

            try:
                if isinstance(result, plotly.graph_objs.Figure):
                    result.show()
                if isinstance(result, matplotlib.figure.Figure):
                    from matplotlib import pyplot as plt
                    display(result)
                    plt.close(result)
            except Exception as e:
                self.log_warning(
                    f"Error when displaying experiment result of {func.__qualname__}: {e}"
                )
                self.log_warning(f"Ignore the error and continue.")
                self.log_warning(f"{e}")

    @property
    def to_show_figure_in_notebook(self):
        return setup().status().get_parameters("Plot_Result_In_Jupyter")

    @property
    def to_run_ai_inspection(self):
        return setup().status().get_parameters("AIAutoInspectPlots")

    @property
    def is_simulation(self):
        return setup().status().get_parameters("High_Level_Simulation_Mode")

    def log_warning(self, message):
        self.logger.warning(message)


Experiment = LeeQAIExperiment


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

        if self._active_experiment_instance is None:
            experiment_status = {}
        else:
            experiment_status = self._active_experiment_instance.get_experiment_details()

        experiment_status.update(self.get_default_setup().get_live_status())

        return experiment_status

    def get_live_plots(self):
        """
         Get the live figure of the running experiment.

         It calls the live figure function defined in the experiment, and pass the figure to the live monitor.

         Returns:
             figure: The live figure.
         """

        fig = plotly.graph_objects.Figure()

        if self.get_default_setup() is None:
            return fig

        step_no = self.get_default_setup().get_live_status()[
            'engine_status']['step_no']

        if np.sum(step_no) == 0:
            # No data yet
            return fig

        # No active experiment instance, or the instance does not support live
        # plots
        if self._active_experiment_instance is None or not hasattr(
                self._active_experiment_instance, 'live_plots'):
            return fig

        try:
            args = self._active_experiment_instance.retrieve_args(
                self._active_experiment_instance.run)
        except ValueError as e:
            # The experiment has not been registered for plotting
            return fig

        # try:
        fig = self._active_experiment_instance.live_plots(step_no)
        # except Exception as e:
        #    logger.warning(e)

        return fig

    def register_active_experiment_instance(self, instance: Experiment):
        """
        Register the active experiment instance.
        """

        assert isinstance(
            instance, Experiment), f"instance must be an instance of Experiment, got {type(instance)}"
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

    def run(self, lpb, swp=None, basis=None):  # *args, **kwargs):
        """
        Run an experiment.

        Here we assert the active instance has been registered.

        Parameters:
            lpb (LogicalPrimitiveCombinable): The logical primitive block to run.
            swp (Sweeper): The sweeper to run.
            basis (str): The basis to run, depends on the transformation function.

        Returns:
            Any: The return value of the experiment.
        """

        assert self._active_experiment_instance is not None, "No active experiment instance is registered."
        with self.status().with_parameters(measurement_basis=basis):
            return self.get_default_setup().run(lpb, swp)

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
    return setup().run(lpb, swp, basis)
