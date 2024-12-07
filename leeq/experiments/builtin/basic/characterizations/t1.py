from typing import Optional, Any, List, Union
import numpy as np
from plotly import graph_objects as go

from labchronicle import register_browser_function, log_and_record

from k_agents.inspection.decorator import text_inspection, visual_inspection
from leeq import Experiment, Sweeper
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.theory.fits import fit_exp_decay_with_cov
from leeq.theory.utils import to_dense_probabilities
from leeq.utils import setup_logging
from leeq.utils.compatibility import *
from leeq.utils.compatibility.prims import SweepLPB
from leeq.theory.fits.multilevel_decay import fit_decay as fit_multilevel_decay, plot

logger = setup_logging(__name__)

__all__ = ['SimpleT1', 'MultiQubitT1', 'MultiQuditT1Decay']


class SimpleT1(Experiment):
    """
    A class used to represent a Simple T1 Experiment.

    ...

    Attributes
    ----------
    trace : np.ndarray
        Stores the result of the measurement primitive.

    fit_params : dict
        Stores the parameters of the fitted exponential decay.

    Methods
    -------
    run(qubit, collection_name, initial_lpb, mprim_index, time_length, time_resolution, hardware_stall)
        Runs the T1 experiment.
    plot_t1()
        Plots the T1 decay.
    """

    _experiment_result_analysis_instructions = """The T1 experiment measures the relaxation time of a qubit. 
    Please analyze the fitted plots and the fitting model to verify the data's validity. Subsequently, determine
    if the experiment needs to be rerun and adjust the experimental parameters as necessary. The suggested time
    length should be approximately 5 times the T1 value. If there is a significant discrepancy, adjust the time
    length accordingly. Consider the experiment a failure if no decay is observed in the data or if adjustments to the
    parameters are necessary. Additionally, modify the time resolution to capture approximately 100 data points.
    """

    @log_and_record(overwrite_func_name='SimpleT1.run')
    def run_simulated(self,
            qubit: Any,  # Add the expected type for 'qubit' instead of Any
            collection_name: str = 'f01',
            # Add the expected type for 'initial_lpb' instead of Any
            initial_lpb: Optional[Any] = None,
            mprim_index: int = 0,
            time_length: float = 100.0,
            time_resolution: float = 1.0
            ) -> None:
        """Run the T1 experiment with the specified parameters.

        Parameters:
        qubit (Any): The qubit object to be used in the experiment.
        collection_name (str): The collection name for the qubit transition.
        initial_lpb (Optional[Any]): Initial list of pulse blocks (LPB).
        mprim_index (int): Index of the measurement primitive.
        time_length (float): Total time length of the experiment in microseconds.
        time_resolution (float): Time resolution for the experiment in microseconds.
        """

        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()
        virtual_transmon = simulator_setup.get_virtual_qubit(qubit)
        t1 = virtual_transmon.t1

        sweep_range = np.arange(0.0, time_length, time_resolution)

        data = np.exp(-sweep_range / t1)

        # If sampling noise is enabled, simulate the noise
        if setup().status().get_param('Sampling_Noise'):
            # Get the number of shot used in the simulation
            shot_number = setup().status().get_param('Shot_Number')

            # generate binomial distribution of the result to simulate the
            # sampling noise
            data = np.random.binomial(
                shot_number, data) / shot_number

        quiescent_state_distribution = virtual_transmon.quiescent_state_distribution
        standard_deviation = np.sum(quiescent_state_distribution[1:])

        random_noise_factor = 1 + np.random.normal(
            0, standard_deviation, data.shape)

        self.trace = np.clip(data * quiescent_state_distribution[0] * random_noise_factor, -1, 1)

    @log_and_record
    def run(self,
            qubit: Any,  # Add the expected type for 'qubit' instead of Any
            collection_name: str = 'f01',
            # Add the expected type for 'initial_lpb' instead of Any
            initial_lpb: Optional[Any] = None,
            mprim_index: int = 0,
            time_length: float = 100.0,
            time_resolution: float = 1.0
            ) -> None:
        """Run the T1 experiment with the specified parameters.

        Parameters:
        qubit (Any): The qubit object to be used in the experiment.
        collection_name (str): The collection name for the qubit transition.
        initial_lpb (Optional[Any]): Initial list of pulse blocks (LPB).
        mprim_index (int): Index of the measurement primitive.
        time_length (float): Total time length of the experiment in microseconds.
        time_resolution (float): Time resolution for the experiment in microseconds.
        """
        self.trace = None

        c1 = qubit.get_c1(collection_name)
        mp = qubit.get_measurement_prim_intlist(mprim_index)
        self.mp = mp
        delay = prims.Delay(0)

        lpb = c1['X'] + delay + mp

        if initial_lpb:
            lpb = initial_lpb + lpb

        sweep_range = np.arange(0.0, time_length, time_resolution)
        swp = Sweeper(sweep_range,
                      params=[sparam.func(delay.set_delay, {}, 'delay')])

        basic(lpb, swp, 'p(1)')
        self.trace = np.squeeze(mp.result())

    @text_inspection
    def fitting(self) -> Union[str, None]:
        """
        Get the prompt to analyze the data.

        Returns:
        str: The prompt to analyze the data.
        """
        args = self._get_run_args_dict()

        t = np.arange(0, args['time_length'], args['time_resolution'])
        trace = self.trace

        fit_params = fit_exp_decay_with_cov(trace, args['time_resolution'])

        self.fit_params = fit_params

        t1 = fit_params['Decay']

        return f"The sweep time length is {args['time_length']} us and " + "the fitted curve reports a T1 value of " + f"{t1} us."

    @register_browser_function(available_after=(run,))
    @visual_inspection(
        "Please analyze the experimental data in the plot to determine if there's a clear exponential"
        "decay pattern followed by stabilization. It is important that the decay is observable, as the "
        "absence of decay is considered a failure of the experiment. Check if the tail of the decay "
        "stabilizes within the observed time frame and inform me what portion of the time frame is "
        "occupied by this stable section. The total sweep time frame value should be approximately 5 times"
        "the estimated T1 time to ensure a accurate estimation. If the values are too far apart, adjust the "
        "time frame accordingly."
    )
    def plot_t1(self, fit=True, step_no=None) -> go.Figure:
        """
        Plot the T1 decay graph based on the trace and fit parameters using Plotly.

        Parameters:
        fit (bool): Whether to fit the trace. Defaults to True.
        step_no (Tuple[int]): Number of steps to plot.

        Returns:
        go.Figure: The Plotly figure object.
        """
        self.fit_params = {}  # Initialize as an empty dictionary or suitable default value

        args = self._get_run_args_dict()

        t = np.arange(0, args['time_length'], args['time_resolution'])

        if self.trace is None:
            trace = np.squeeze(self.mp.result())
        else:
            trace = self.trace

        if step_no is not None:
            t = t[:step_no[0]]
            trace = trace[:step_no[0]]

        # Create traces for scatter and line plot
        trace_scatter = go.Scatter(
            x=t, y=trace,
            mode='markers',
            marker=dict(
                # symbol='x',
                size=5,
                # color='blue'
            ),
            name='Experiment data'
        )

        title = f"T1 decay {args['qubit'].hrid} transition {args['collection_name']}"

        data = [trace_scatter]

        if fit:
            fit_params = fit_exp_decay_with_cov(trace, args['time_resolution'])

            self.fit_params = fit_params

            trace_line = go.Scatter(
                x=t,
                y=fit_params['Amplitude'].n * np.exp(-t / fit_params['Decay'].n) + fit_params['Offset'].n,
                mode='lines',
                line=dict(
                    color='blue'
                ),
                visible='legendonly',
                name='Decay fit'
            )
            title = (
                f"T1 decay {args['qubit'].hrid} transition {args['collection_name']}<br>"
                f"T1={fit_params['Decay']} us")

            data = [trace_scatter, trace_line]

        layout = go.Layout(
            title=title,
            xaxis=dict(title='Time (us)'),
            yaxis=dict(title='P(1)'),
            plot_bgcolor='white',
            showlegend=True
        )

        fig = go.Figure(data=data, layout=layout)

        return fig

    def live_plots(self, step_no):
        """
        Plot the T1 decay graph live using Plotly.
        We start to plot after 10 points

        Parameters:
        step_no (Tuple[int]): Number of steps to plot.

        Returns:
        go.Figure: The Plotly figure object.
        """
        return self.plot_t1(fit=step_no[0] > 10, step_no=step_no)


class MultiQubitT1(Experiment):
    """
    A class used to represent a multi qubit T1 Experiment.

    ...

    Attributes
    ----------
    trace : np.ndarray
        Stores the result of the measurement primitive.

    fit_params : dict
        Stores the parameters of the fitted exponential decay.

    Methods
    -------
    run(qubit, collection_name, initial_lpb, mprim_index, time_length, time_resolution, hardware_stall)
        Runs the T1 experiment.
    plot_t1()
        Plots the T1 decay.
    """

    @log_and_record
    def run(self,
            # Add the expected type for 'qubit' instead of Any
            duts: List[Any],
            collection_names: Union[str, List[str]] = 'f01',
            # Add the expected type for 'initial_lpb' instead of Any
            initial_lpb: Optional[Any] = None,
            mprim_indexes: int = 0,
            time_length: float = 100.0,
            time_resolution: float = 1.0
            ) -> None:
        """Run the multi qubit T1 experiment with the specified parameters.

        Parameters:
        duts (List[Any]): A list of qubit objects to be used in the experiment.
        collection_names (Union[str,List[str]]): The collection name for the qubit transition.
        initial_lpb (Optional[Any]): Initial list of pulse blocks (LPB).
        mprim_indexes (int): Index of the measurement primitive.
        time_length (float): Total time length of the experiment in microseconds.
        time_resolution (float): Time resolution for the experiment in microseconds.
        """
        if isinstance(collection_names, str):
            collection_names = [collection_names] * len(duts)

        self.collection_names = collection_names
        if isinstance(mprim_indexes, int):
            mprim_indexes = [mprim_indexes] * len(duts)

        c1s = [
            qubit.get_c1(collection_name) for qubit,
            collection_name in zip(
                duts,
                collection_names)]
        mps = [
            qubit.get_measurement_prim_intlist(mprim_index) for qubit,
            mprim_index in zip(
                duts,
                mprim_indexes)]
        self.mps = mps
        delay = prims.Delay(0)

        lpb = prims.ParallelLPB([c1['X'] for c1 in c1s]) + \
              delay + prims.ParallelLPB(mps)

        if initial_lpb:
            lpb = initial_lpb + lpb

        sweep_range = np.arange(0.0, time_length, time_resolution)
        swp = Sweeper(sweep_range,
                      params=[sparam.func(delay.set_delay, {}, 'delay')])

        basic(lpb, swp, 'p(1)')
        self.traces = [np.squeeze(mp.result()) for mp in mps]

    @register_browser_function(available_after=(run,))
    def plot_all(self):
        """
        Plot the T1 decay graph based on the trace and fit parameters using Plotly.
        """
        for i in range(len(self.traces)):
            fig = self.plot_t1(i=i)
            fig.show()

    def plot_t1(self, i, fit=True) -> go.Figure:
        """
        Plot the T1 decay graph based on the trace and fit parameters using Plotly.

        Parameters:
        fit (bool): Whether to fit the trace. Defaults to True.
        step_no (Tuple[int]): Number of steps to plot.

        Returns:
        go.Figure: The Plotly figure object.
        """

        args = self._get_run_args_dict()

        t = np.arange(0, args['time_length'], args['time_resolution'])
        trace = self.traces[i]

        # Create traces for scatter and line plot
        trace_scatter = go.Scatter(
            x=t, y=trace,
            mode='markers',
            marker=dict(
                symbol='x',
                size=10,
                color='blue'
            ),
            name='Experiment data'
        )

        data = [trace_scatter]
        title = f"T1 decay {args['duts'][i].hrid} transition {self.collection_names[i]}"

        if fit:
            fit_params = fit_exp_decay_with_cov(trace, args['time_resolution'])

            trace_line = go.Scatter(
                x=t,
                y=fit_params['Amplitude'].n * np.exp(-t / fit_params['Decay'].n) + fit_params['Offset'].n,
                mode='lines',
                line=dict(
                    color='blue'
                ),
                name='Decay fit'
            )
            title = (
                f"T1 decay {args['duts'][i].hrid} transition {self.collection_names[i]}<br>"
                f"T1={fit_params['Decay']} us")

            data = [trace_scatter, trace_line]

        layout = go.Layout(
            title=title,
            xaxis=dict(title='Time (us)'),
            yaxis=dict(title='P(1)'),
            plot_bgcolor='white',
            showlegend=True
        )

        fig = go.Figure(data=data, layout=layout)

        return fig


class MultiQuditT1Decay(Experiment):

    @log_and_record
    def run(self,
            duts: List[Any],
            time_length: float = 200,
            time_resolution: float = 5,
            mprim_indexes: Union[int, List[int]] = 2,
            max_level: int = 3,
            measurement_mitigation: bool = False
            ):
        """
        Run the T1 experiment with the specified parameters.

        Parameters:
        duts (List[Any]): A list of qubit objects to be used in the experiment.
        time_length (float): Total time length of the experiment in microseconds.
        time_resolution (float): Time resolution for the experiment in microseconds.
        mprim_indexes (Union[int, List[int]]): Index of the measurement primitive.
        max_level (int): The highest level we reach to here.
        measurement_mitigation (bool): Whether to apply measurement mitigation, evaluate the assignment
        matrix and apply inverse to the population distribution.
        """

        self.time_length = time_length
        self.time_resolution = time_resolution
        self.max_level = max_level

        self.assignment_calibration = None

        if measurement_mitigation:
            from leeq.experiments.builtin import CalibrateSingleDutAssignmentMatrices
            self.assignment_calibration = CalibrateSingleDutAssignmentMatrices(duts=duts, mprim_index=mprim_indexes)

        if self.max_level > 3:
            msg = f"Level {self.max_level} not supported yet."
            logger.error(msg)
            raise RuntimeError(msg)

        if isinstance(mprim_indexes, int):
            mprim_indexes = [mprim_indexes] * len(duts)

        c1_01s = [dut.get_c1('f01') for dut in duts]
        c1_12s = [dut.get_c1('f12') for dut in duts]
        c1_23s = [dut.get_c1('f23') for dut in duts]

        c1_01_pulses = prims.ParallelLPB([c1['X'] for c1 in c1_01s])
        c1_12_pulses = prims.ParallelLPB([c1['X'] for c1 in c1_12s])
        c1_23_pulses = prims.ParallelLPB([c1['X'] for c1 in c1_23s])

        delay = prims.Delay(0)

        lpb_list = [
            c1_01_pulses,
            c1_01_pulses + c1_12_pulses,
            c1_01_pulses + c1_12_pulses + c1_23_pulses]

        lpb = SweepLPB(
            lpb_list[:self.max_level],
        )

        swp_lpb = sweeper.from_sweep_lpb(lpb)

        delay = prims.Delay(0)

        lpb = lpb + delay
        swp_time = sweeper(
            np.arange,
            n_kwargs={
                'start': 0.0,
                'stop': time_length,
                'step': time_resolution},
            params=[
                sparam.func(
                    delay.set_delay,
                    {},
                    'delay')])

        mprims = [dut.get_measurement_prim_intlist(mprim_index) for mprim_index, dut in zip(mprim_indexes, duts)]

        lpb = lpb + prims.ParallelLPB(mprims)

        basic(lpb, swp_time + swp_lpb, '<zs>')

        self.results = [
            np.squeeze(mprim.result()) for mprim in mprims
        ]

    def analyze_data(self):
        """
        Analyze the data and fit the decay.
        """
        probs = []
        for r in self.results:
            r_reindexed = r[np.newaxis, :, :, :, ].transpose([0, 3, 1, 2])
            p = to_dense_probabilities(r_reindexed, base=self.max_level + 1)
            probs.append(p)

        if self.assignment_calibration is not None:
            self.probs = self.assignment_calibration.apply_inverse(probs)
        else:
            self.probs = probs
        self.fit_params = []
        self.t1_list = []

        for i, prob in enumerate(self.probs):
            initial_state, gamma = self.analyze_single_dut(i)
            self.fit_params.append((initial_state, gamma))

            t1s = []
            for j in range(1, self.max_level + 1):
                t1 = -1 / np.sum(gamma[j, :j])
                t1s.append(t1)

            self.t1_list.append(t1s)

    def analyze_single_dut(self, dut_index):
        """
        Analyze the data for a single DUT and fit the decay.

        Parameters:
            dut_index (int): The index of the DUT to analyze.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The initial state and gamma values.
        """
        probs = self.probs[dut_index].transpose([1, 2, 0])
        initial_state, gamma = fit_multilevel_decay(probs, time_length=self.time_length,
                                                    time_resolution=self.time_resolution)
        return initial_state, gamma

    @register_browser_function(available_after=(run,))
    def plot_all(self):
        """
        Plot the T1 decay graph based on the trace and fit parameters using Plotly.
        """
        self.analyze_data()
        for i in range(len(self.probs)):
            probs = self.probs[i].transpose([1, 2, 0])
            fit_param = self.fit_params[i]
            t1s = self.t1_list[i]
            fig = plot(probs=probs, time_length=self.time_length, time_resolution=self.time_resolution,
                       initial_distribution=fit_param[0], gamma=fit_param[1])
            for i in range(self.max_level):
                print(f"T1_{i + 1}{i} = {t1s[i]}")
            fig.show()
