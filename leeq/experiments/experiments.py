import datetime
from typing import Union
from functools import wraps

from IPython.display import display
import inspect

import matplotlib
import numpy as np
import plotly
from IPython.core.display import Markdown

from labchronicle import Chronicle

from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveCombinable
from leeq.experiments.sweeper import Sweeper
from leeq.setups.setup_base import SetupStatusParameters
from leeq.utils import Singleton, setup_logging, display_json_dict
from leeq.utils.notebook import show_spinner, hide_spinner
from leeq.utils.ai.display_chat.notebooks import display_chat, dict_to_html
import leeq.experiments.plots.live_dash_app as live_monitor
from leeq.utils.ai.vlms import (has_visual_analyze_prompt, visual_inspection, get_visual_analyze_prompt)
from leeq.utils.ai.experiment_summarize import get_experiment_summary

logger = setup_logging(__name__)


class LeeQExperiment(LeeQObject):
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

    def _execute_browsable_plot_function(self, build_static_image=False):
        """
        Execute the browsable plot function.

        Parameters:
            build_static_image (bool): Whether to build the static image.
        """
        for name, func in self.get_browser_functions():
            try:
                self._execute_single_browsable_plot_function(func, build_static_image=build_static_image)
            except Exception as e:
                msg = f"Error when executing the browsable plot function {name}:{e}."
                self.logger.warning(msg)

    def _execute_single_browsable_plot_function(self, func: callable, build_static_image=False):
        """
        Execute the browsable plot function. The result and image will be stored in the function object
        attributes '_result' and '_image'.

        Parameters:
            func (callable): The browsable plot function.
            build_static_image (bool): Whether to build the static image.

        """
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

        result = None

        try:
            result = func(*f_args, **filtered_kwargs)
            if build_static_image:
                from leeq.utils.ai.utils import matplotlib_plotly_to_pil
                image = matplotlib_plotly_to_pil(result)
                self._browser_function_images[func.__qualname__] = image

        except Exception as e:
            self.logger.warning(
                f"Error when executing {func.__qualname__} with parameters ({f_args},{f_kwargs}): {e}"
            )
            self.logger.warning(f"Ignore the error and continue.")
            self.logger.warning(f"{e}")
            raise e

        self._browser_function_results[func.__qualname__] = result

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
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            msg = None
        except TypeError as e:
            msg = f"{e}\n\n"
            msg += f"Function signature: {sig}\n\n"
            msg += f"Documents:\n\n {self.run.__doc__}\n\n"

        if msg is not None:
            raise TypeError(msg)

        return bound

    def __init__(self, *args, **kwargs):
        """
        Initialize the experiment.
        """

        # Check the input arguments
        bound = self._check_arguments(self.run, *args, **kwargs)

        super(
            LeeQExperiment, self).__init__(
            name=f"Experiment: {self.__class__.__name__}")

        # Register the active experiment instance
        ExperimentManager().register_active_experiment_instance(self)

        self._browser_function_results = {}
        self._browser_function_images = {}

        try:
            # Run the experiment
            if setup().status().get_parameters("High_Level_Simulation_Mode"):
                self.run_simulated(*bound.args, **bound.kwargs)
            else:
                self.run(*bound.args, **bound.kwargs)
        finally:
            # Make sure we print the record details before throwing the
            # exception
            if Chronicle().is_recording():
                # Print the record details
                record_details = self.retrieve_latest_record_entry_details(
                    self.run)
                if record_details is not None:
                    record_details = record_details.copy()
                    record_details.update(
                        {'print_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                    display_json_dict(
                        record_details,
                        root=self.__class__.__qualname__,
                        expanded=False)
                else:
                    msg = f"Failed to retrieve and save the record details for {self.run.__qualname__}"
                    logger.warning(msg)

        self._post_run()

    def _post_run(self):
        """
        The post run method to execute after the experiment is finished.
        """

        # Check if we need to plot
        show_plots = setup().status().get_parameters("Plot_Result_In_Jupyter")

        if show_plots:
            for name, func in self.get_browser_functions():
                try:
                    self._execute_single_browsable_plot_function(func)
                    result = self._browser_function_results[func.__qualname__]
                except Exception as e:
                    self.logger.warning(
                        f"Error when executing the browsable plot function {name}:{e}."
                    )
                    self.logger.warning(f"Ignore the error and continue.")
                    self.logger.warning(f"{e}")
                    continue

                try:
                    if isinstance(
                            result, plotly.graph_objs.Figure):
                        result.show()
                    if isinstance(result, matplotlib.figure.Figure):
                        from matplotlib import pyplot as plt
                        display(result)
                        plt.close(result)
                except Exception as e:
                    self.logger.warning(
                        f"Error when displaying experiment result of {func.__qualname__}: {e}"
                    )
                    self.logger.warning(f"Ignore the error and continue.")
                    self.logger.warning(f"{e}")

    def run_simulated(self, *args, **kwargs):
        """
        Run the experiment in simulation mode. This is useful for debugging.
        """
        raise NotImplementedError()

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

        if self.run.__qualname__ not in self._register_log_and_record_args_map:
            return {}

        kwargs = self.retrieve_args(self.run)
        kwargs = {k: repr(v) for k, v in kwargs.items()}
        kwargs['name'] = self._name

        return {"record_details": self.retrieve_latest_record_entry_details(
            self.run), "experiment_arguments": kwargs, }


class LeeQAIExperiment(LeeQExperiment):
    """
    A extension class for AI compatible experiments definitions.
    """

    _experiment_result_analysis_instructions = None

    def __init__(self, *args, **kwargs):
        """
        Initialize the experiment.
        """
        self._ai_inspection_results = {}
        self._ai_final_analysis = None
        super(LeeQAIExperiment, self).__init__(*args, **kwargs)

    @classmethod
    def is_ai_compatible(cls):
        """
        A method to indicate that the experiment is AI compatible.
        """
        return cls._experiment_result_analysis_instructions is not None

    def _run_ai_inspection_on_single_function(self, func):
        """
        Run the AI inspection on a single function.

        Parameters:
            func (callable): The function to analyze.
        Returns:
            dict: The result of the analysis.
        """

        if self._ai_inspection_results.get(func.__qualname__) is not None:
            return self._ai_inspection_results.get(func.__qualname__)

        try:
            if has_visual_analyze_prompt(func):
                if self._browser_function_images.get(func.__qualname__) is None:
                    self._execute_single_browsable_plot_function(func, build_static_image=True)

                image = self._browser_function_images.get(func.__qualname__)

                spinner_id = show_spinner(f"Vision AI is inspecting plots...")
                prompt = get_visual_analyze_prompt(func)
                inspect_answer = visual_inspection(image, prompt)
                self._ai_inspection_results[func.__qualname__] = inspect_answer
                hide_spinner(spinner_id)
                return inspect_answer

        except Exception as e:
            self.logger.warning(
                f"Error when running single AI inspection on {func.__qualname__}: {e}"
            )
            self.logger.warning(f"Ignore the error and continue.")
            self.logger.warning(f"{e}")

        return None

    def _get_all_ai_inspectable_functions(self) -> dict:
        """
        Get all the AI inspectable functions.

        Returns:
            dict: The AI inspectable functions.
        """
        return dict(
            [(name, func) for name, func in self.get_browser_functions() if has_visual_analyze_prompt(func)])

    def get_analyzed_result_prompt(self) -> Union[str, None]:
        """
        Get the natual language description of the analyzed result for AI.

        Returns
        -------
        str: The prompt to analyze the result.
        """
        return None

    def get_ai_inspection_results(self):
        """
        Get the AI inspection results.

        Returns:
            dict: The AI inspection results.
        """
        ai_inspection_results = {}
        for name, func in self._get_all_ai_inspectable_functions().items():

            if self._ai_inspection_results.get(func.__qualname__) is None:
                try:
                    self._run_ai_inspection_on_single_function(func)
                except Exception as e:
                    self.logger.warning(
                        f"Error when doing get AI inspection on {func.__qualname__}: {e}"
                    )
                    self.logger.warning(f"Ignore the error and continue.")
                    self.logger.warning(f"{e}")

        ai_inspection_results = {key.split('.')[-1]: val['analysis'] for key, val in
                                 self._ai_inspection_results.items()}
        fitting_results = self.get_analyzed_result_prompt()

        if fitting_results is not None:
            ai_inspection_results['fitting'] = fitting_results

        if self._experiment_result_analysis_instructions is not None:

            if self._ai_final_analysis is None:
                spinner_id = show_spinner(f"AI is analyzing the experiment results...")

                run_args_prompt = f"""
                Document of this experiment:
                {self.run.__doc__}

                Running arguments:
                {self.retrieve_args(self.run)}
                """

                summary = get_experiment_summary(self._experiment_result_analysis_instructions, run_args_prompt,
                                                 ai_inspection_results)
                self._ai_final_analysis = summary
            else:
                summary = self._ai_final_analysis

            ai_inspection_results['Final analysis'] = summary['analysis']
            ai_inspection_results['Suggested parameter updates'] = summary['parameter_updates']
            ai_inspection_results['Experiment success'] = summary['success']
            hide_spinner(spinner_id)

        return ai_inspection_results

    def _post_run(self):
        """
        The post run method to execute after the experiment is finished.
        """

        super(LeeQAIExperiment, self)._post_run()

        run_ai_inspection = setup().status().get_parameters("AIAutoInspectPlots")

        if run_ai_inspection:
            for name, func in self.get_browser_functions():
                inspect_answer = self._run_ai_inspection_on_single_function(func)
                if inspect_answer is not None:
                    html = dict_to_html(inspect_answer)
                    display_chat(agent_name=f"Inspection AI",
                                 content='<br>' + html,
                                 background_color='#f0f8ff')


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
