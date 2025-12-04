from typing import Any, Optional, Union

import numpy as np
from k_agents.inspection.decorator import text_inspection, visual_inspection
from plotly import graph_objects as go

from leeq import Experiment
from leeq.chronicle import log_and_record, register_browser_function
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.theory.fits import fit_exp_decay_with_cov
from leeq.theory.utils import to_dense_probabilities
from leeq.utils.compatibility import *

__all__ = [
    'SpinEchoMultiLevel',
    'MultiQubitSpinEchoMultiLevel'
]


class SpinEchoMultiLevel(
        Experiment):  # Class names should follow the CapWords convention
    """
    A class used to represent the SpinEchoMultiLevel experiment.

    Methods
    -------
    run(...)
        Runs the experiment.

    plot_echo()
        Plots the results of the echo experiment.
    """

    EPII_INFO = {
        "name": "SpinEchoMultiLevel",
        "description": "Spin echo T2 coherence time measurement experiment",
        "purpose": "Measures the T2 echo coherence time using a spin echo pulse sequence (π/2 - τ/2 - π - τ/2 - π/2 - measure). This refocuses low-frequency noise and provides the intrinsic T2 coherence time, which is typically longer than T2* from Ramsey experiments. Essential for characterizing qubit coherence properties.",
        "attributes": {
            "trace": {
                "type": "np.ndarray[float]",
                "description": "Population measurements as function of free evolution time",
                "shape": "(n_time_points,)"
            },
            "mp": {
                "type": "MeasurementPrimitive",
                "description": "Measurement primitive used for qubit readout"
            },
            "fit_params": {
                "type": "dict",
                "description": "Fitted exponential decay parameters",
                "keys": {
                    "Decay": "ufloat - T2 echo time constant in microseconds",
                    "Amplitude": "ufloat - Initial amplitude of decay",
                    "Offset": "ufloat - Steady-state offset value"
                }
            }
        },
        "notes": [
            "Free evolution time should be ~5x expected T2 echo value",
            "The π pulse refocuses static field inhomogeneities",
            "T2 echo is typically longer than T2* (Ramsey)",
            "Fits to: A * exp(-t/T2_echo) + Offset",
            "Time resolution should capture ~50 data points"
        ]
    }

    _experiment_result_analysis_instructions = """The Spin echo experiment measures the T2 echo relaxation time of a qubit.
    Please analyze the fitted plots and the fitting model to verify the data's validity. Subsequently, determine
    if the experiment needs to be rerun and adjust the experimental parameters as necessary. The suggested time
    length should be approximately 5 times the T2 value. If there is a significant discrepancy, adjust the time
    length accordingly and report experiment failure. Additionally, modify the time resolution to capture approximately 50 data points.
    """

    @log_and_record(overwrite_func_name='SpinEchoMultiLevel.run')
    def run_simulated(
            self,
            dut: Any,  # Replace 'Any' with the actual type of qubit
            collection_name: str = 'f01',
            mprim_index: int = 0,
            free_evolution_time: float = 100.0,
            time_resolution: float = 2.0,
            start: float = 0.0,
            initial_lpb: Optional[Any] = None,
    ) -> None:
        """Execute the SpinEchoMultiLevel experiment in simulation mode.

        This experiment implements a spin echo sequence: π/2 - τ/2 - π - τ/2 - π/2 - measure.
        The π pulse refocuses dephasing, measuring intrinsic T2 coherence time.

        Parameters
        ----------
        dut : Any
            The qubit to be used in the experiment.
        collection_name : str, optional
            The name of the pulse collection. Default: 'f01'
        mprim_index : int, optional
            Index of the measurement primitive. Default: 0
        free_evolution_time : float, optional
            The maximum free evolution time in microseconds. Default: 100.0
        time_resolution : float, optional
            The time resolution for the experiment in microseconds. Default: 2.0
        start : float, optional
            Start time of the sweep in microseconds. Default: 0.0
        initial_lpb : Optional[Any], optional
            The initial local pulse builder. Default: None

        Returns
        -------
        None
            Updates the instance's trace attribute with simulated data.
        """

        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()
        virtual_transmon = simulator_setup.get_virtual_qubit(dut)
        t2 = virtual_transmon.t2

        sweep_range = np.arange(0.0, free_evolution_time, time_resolution)

        data = 0.5 + np.exp(-sweep_range / t2) / 2

        # If sampling noise is enabled, simulate the noise
        if setup().status().get_param('Sampling_Noise'):
            # Get the number of shot used in the simulation
            shot_number = setup().status().get_param('Shot_Number')

            # generate binomial distribution of the result to simulate the
            # sampling noise
            data = np.random.binomial(
                shot_number, data) / shot_number

        quiescent_state_distribution = virtual_transmon.quiescent_state_distribution
        standard_deviation = np.sum(quiescent_state_distribution[1:]) / 5

        random_noise_factor = 1 + np.random.normal(
            0, standard_deviation, data.shape)

        self.trace = np.clip(data * quiescent_state_distribution[0] * random_noise_factor, -1, 1)

    @log_and_record
    def run(
            self,
            dut: Any,  # Replace 'Any' with the actual type of qubit
            collection_name: str = 'f01',
            mprim_index: int = 0,
            free_evolution_time: float = 100.0,
            time_resolution: float = 2.0,
            start: float = 0.0,
            initial_lpb: Optional[Any] = None,
    ) -> None:
        """Execute the SpinEchoMultiLevel experiment on hardware.

        This experiment implements a spin echo sequence: π/2 - τ/2 - π - τ/2 - π/2 - measure.
        The π pulse refocuses dephasing, measuring intrinsic T2 coherence time.

        Parameters
        ----------
        dut : Any
            The qubit to be used in the experiment.
        collection_name : str, optional
            The name of the pulse collection. Default: 'f01'
        mprim_index : int, optional
            Index of the measurement primitive. Default: 0
        free_evolution_time : float, optional
            The maximum free evolution time in microseconds. Default: 100.0
        time_resolution : float, optional
            The time resolution for the experiment in microseconds. Default: 2.0
        start : float, optional
            Start time of the sweep in microseconds. Default: 0.0
        initial_lpb : Optional[Any], optional
            The initial local pulse builder. Default: None

        Returns
        -------
        None
            Updates the instance's trace attribute with measured data.

        Example
        -------
        >>> # Assume 'dut' is the qubit object
        >>> experiment = SpinEchoMultiLevel(
        >>>     dut=dut, collection_name='f01', mprim_index=0,
        >>>     free_evolution_time=100.0, time_resolution=2.0, start=0.0
        >>> )
        """

        qubit = dut
        c1 = qubit.get_c1(collection_name)
        mp = qubit.get_measurement_prim_intlist(mprim_index)
        delay = prims.Delay(0)
        self.trace = None

        self.mp = mp

        lpb = c1['Xp'] + delay + c1['Y'] + delay + c1['Xp'] + mp

        swp_args = {
            'start': start / 2.0,
            'stop': free_evolution_time / 2.0,
            'step': time_resolution / 2.0
        }
        swp = sweeper(
            np.arange,
            n_kwargs=swp_args,
            params=[sparam.func(delay.set_delay, {}, 'delay')]
        )

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        basic(lpb, swp, 'p(1)')

        self.trace = np.squeeze(mp.result())

    @register_browser_function(available_after=(run,))
    @visual_inspection(
        "Please analyze the experimental data in the plot to determine if there's a clear exponential"
        "decay pattern followed by stabilization. It is important that the decay is observable, as the "
        "absence of decay is considered a failure of the experiment. Check if the tail of the decay "
        "stabilizes within the observed time frame and inform me what portion of the time frame is "
        "occupied by this stable section. The total sweep time frame value should be approximately 5 times"
        "the estimated T2 time to ensure a accurate estimation. If the values are too far apart, adjust the "
        "time frame accordingly."
    )
    def plot_echo(self, fit=True, step_no=None) -> go.Figure:
        """
        Plot the results of the echo experiment using Plotly.
        """

        if self.trace is None:
            trace = np.squeeze(self.mp.result())
        else:
            trace = self.trace

        args = self._get_run_args_dict()

        t = np.arange(0, args['free_evolution_time'], args['time_resolution'])

        if step_no is not None:
            t = t[:step_no[0]]
            trace = trace[:step_no[0]]

        # Create traces for scatter and line plot
        trace_scatter = go.Scatter(
            x=t, y=trace,
            mode='markers',
            marker={
                'size': 5,
                'color': 'blue'
            },
            name='Experiment data'
        )

        title = f"T2 decay {args['dut'].hrid} transition {args['collection_name']}"

        data = [trace_scatter]

        if fit:
            fit_params = fit_exp_decay_with_cov(trace, args['time_resolution'])
            self.fit_params = fit_params
            trace_line = go.Scatter(
                x=t,
                y=fit_params['Amplitude'].n * np.exp(-t / fit_params['Decay'].n) + fit_params['Offset'].n,
                mode='lines',
                line={
                    'color': 'blue'
                },
                name='Decay fit',
                visible='legendonly'
            )
            title = (
                f"T2 echo {args['dut'].hrid} transition {args['collection_name']}<br>"
                f"T2={fit_params['Decay']} us")

            data = [trace_scatter, trace_line]

        layout = go.Layout(
            title=title,
            xaxis={'title': 'Time (us)'},
            yaxis={'title': 'P(0)'},
            plot_bgcolor='white',
            showlegend=True
        )

        # Combine the traces and layout into a figure
        fig = go.Figure(data=data, layout=layout)

        return fig

    def live_plots(self, step_no):
        """
        Plot the results of the echo experiment using Matplotlib.
        """

        return self.plot_echo(fit=step_no[0] > 10, step_no=step_no)

    @text_inspection
    def fitting(self) -> Union[str, None]:

        trace = self.trace
        args = self._get_run_args_dict()

        fit_params = fit_exp_decay_with_cov(trace, args['time_resolution'])
        self.fit_params = fit_params

        t2 = fit_params['Decay']

        return f"The sweep time length is {args['free_evolution_time']} us and " + "the fitted curve reports a T2 echo value of " + f"{t2} us."


class MultiQubitSpinEchoMultiLevel(
        Experiment):  # Class names should follow the CapWords convention
    """
    A class used to represent the SpinEchoMultiLevel experiment.

    Methods
    -------
    run(...)
        Runs the experiment.

    plot_echo()
        Plots the results of the echo experiment.
    """

    EPII_INFO = {
        "name": "MultiQubitSpinEchoMultiLevel",
        "description": "Parallel spin echo T2 measurement for multiple qubits",
        "purpose": "Simultaneously measures T2 echo coherence times for multiple qubits using spin echo sequences. Enables efficient characterization of multi-qubit systems by running parallel experiments. Supports multi-level systems and provides normalized population analysis.",
        "attributes": {
            "result": {
                "type": "List[np.ndarray]",
                "description": "Raw measurement results for each qubit",
                "shape": "List of arrays with measurement data"
            },
            "mp": {
                "type": "List[MeasurementPrimitive]",
                "description": "Measurement primitives for each qubit"
            },
            "collection_names": {
                "type": "List[str]",
                "description": "Collection names for each qubit transition"
            },
            "probs": {
                "type": "np.ndarray",
                "description": "Probability distributions for all qubits",
                "shape": "(n_qubits, n_levels, n_time_points)"
            },
            "normalized_population": {
                "type": "np.ndarray",
                "description": "Normalized population between interested levels",
                "shape": "(n_qubits, n_time_points)"
            }
        },
        "notes": [
            "All qubits measured in parallel for efficiency",
            "Supports different collection names per qubit",
            "Can handle multi-level systems beyond two-level qubits",
            "Normalized population computed between specified transition levels"
        ]
    }

    @log_and_record(overwrite_func_name='MultiQubitSpinEchoMultiLevel.run')
    def run_simulated(
            self,
            duts: Any,
            collection_names: Union[str, list[str]] = 'f01',
            mprim_indexes: Union[int, list[str]] = 0,
            free_evolution_time: float = 100.0,
            time_resolution: float = 4.0,
            start: float = 0.0,
            initial_lpb: Optional['LogicalPrimitiveBlock'] = None,
    ) -> None:
        """Execute the multi-qubit spin echo experiment in simulation mode.

        Parameters
        ----------
        duts : List[Any]
            List of qubit objects to be used in the experiment.
        collection_names : Union[str, List[str]], optional
            The pulse collection name(s). If a list, length must match number of qubits. Default: 'f01'
        mprim_indexes : Union[int, List[int]], optional
            Measurement primitive index(es). If a list, length must match number of qubits. Default: 0
        free_evolution_time : float, optional
            The maximum free evolution time in microseconds. Default: 100.0
        time_resolution : float, optional
            The time resolution for the experiment in microseconds. Default: 4.0
        start : float, optional
            Start time of the sweep in microseconds. Default: 0.0
        initial_lpb : Optional[LogicalPrimitiveBlock], optional
            The initial local pulse builder. Default: None

        Returns
        -------
        None
            Updates the instance's result attribute with simulated data.
        """
        # Simulation implementation placeholder
        raise NotImplementedError("Simulation for MultiQubitSpinEchoMultiLevel not yet implemented")

    @log_and_record
    def run(
            self,
            duts: Any,
            collection_names: Union[str, list[str]] = 'f01',
            mprim_indexes: Union[int, list[str]] = 0,
            free_evolution_time: float = 100.0,
            time_resolution: float = 4.0,
            start: float = 0.0,
            initial_lpb: Optional['LogicalPrimitiveBlock'] = None,
    ) -> None:
        """Execute the multi-qubit spin echo experiment on hardware.

        Parameters
        ----------
        duts : List[Any]
            List of qubit objects to be used in the experiment.
        collection_names : Union[str, List[str]], optional
            The pulse collection name(s). If a list, length must match number of qubits. Default: 'f01'
        mprim_indexes : Union[int, List[int]], optional
            Measurement primitive index(es). If a list, length must match number of qubits. Default: 0
        free_evolution_time : float, optional
            The maximum free evolution time in microseconds. Default: 100.0
        time_resolution : float, optional
            The time resolution for the experiment in microseconds. Default: 4.0
        start : float, optional
            Start time of the sweep in microseconds. Default: 0.0
        initial_lpb : Optional[LogicalPrimitiveBlock], optional
            The initial local pulse builder. Default: None

        Returns
        -------
        None
            Updates the instance's result attribute with measured data.
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
                collection_names, strict=False)]
        mprims = [
            qubit.get_measurement_prim_intlist(mprim_index) for qubit,
            mprim_index in zip(
                duts,
                mprim_indexes, strict=False)]
        delay = prims.Delay(0)

        self.mp = mprims

        lpb = prims.ParallelLPB([c1['Xp'] for c1 in c1s]) + delay + prims.ParallelLPB(
            [c1['Y'] for c1 in c1s]) + delay + prims.ParallelLPB([c1['Xp'] for c1 in c1s]) + prims.ParallelLPB(mprims)

        swp_args = {
            'start': start / 2.0,
            'stop': free_evolution_time / 2.0,
            'step': time_resolution / 2.0
        }
        swp = sweeper(
            np.arange,
            n_kwargs=swp_args,
            params=[sparam.func(delay.set_delay, {}, 'delay')]
        )

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        basic(lpb, swp, '<zs>')

        self.result = [np.squeeze(mp.result()) for mp in mprims]

    def analyze_data(self):

        probs = []
        normalized_population = []
        interested_level_low = int(self.collection_names[-1][-2])
        interested_level_high = int(self.collection_names[-1][-1])
        for r in self.result:
            prob = to_dense_probabilities(r.T[np.newaxis, :, :], base=interested_level_high + 1)
            probs.append(prob)
            normalized_population.append(
                prob[interested_level_high, :] / (prob[interested_level_high, :] + prob[interested_level_low, :]))

        self.probs = np.asarray(probs)
        self.normalized_population = np.asarray(normalized_population)

    @register_browser_function(available_after=(run,))
    def plot_all(self):
        """
        Plot the results of the echo experiment using Plotly.
        """
        self.analyze_data()
        for i in range(len(self.result)):
            fig = self.plot_echo(index=i)
            fig.show()

    def plot_echo(self, index, fit=True, step_no=None) -> go.Figure:
        """
        Plot the results of the echo experiment using Plotly.

        Parameters
        ----------
        index : int
            Index of the qubit to plot.
        fit : bool, optional
            Whether to fit the data, by default True.
        step_no : Optional[int], optional
            Number of steps to plot, by default None.

        Returns
        -------
        go.Figure
        """

        trace = self.probs[index, :, :]
        args = self._get_run_args_dict()

        colors = ['red', 'blue', 'green', 'orange']

        t = np.arange(0, args['free_evolution_time'], args['time_resolution'])
        normalized_population = self.normalized_population[index, :]

        if step_no is not None:
            t = t[:step_no[0]]
            trace = trace[:, :step_no[0]]
            normalized_population = normalized_population[:step_no[0]]

        data = []

        for i in range(trace.shape[0]):
            # Create traces for scatter and line plot
            trace_scatter = go.Scatter(
                x=t, y=trace[i, :],
                mode='markers',
                marker={
                    'symbol': "circle-open",
                    'size': 5,
                    'color': colors[i]
                },
                name=f'State {i}'
            )
            data.append(trace_scatter)

        if trace.shape[0] > 2:
            trace_scatter = go.Scatter(
                x=t, y=normalized_population,
                mode='markers',
                marker={
                    'symbol': "diamond-open",
                    'size': 5,
                    'color': 'black'
                },
                name=f'Normalized State {trace.shape[0] - 1}'
            )
            data.append(trace_scatter)

        title = f"T2 decay {args['duts'][index].hrid} transition {self.collection_names[index]}"

        if fit:
            fit_params = fit_exp_decay_with_cov(normalized_population, args['time_resolution'])
            trace_line = go.Scatter(
                x=t,
                y=fit_params['Amplitude'].n * np.exp(-t / fit_params['Decay'].n) + fit_params['Offset'].n,
                mode='lines',
                line={
                    'color': 'black'
                },
                name='Decay fit'
            )
            title = (
                f"T2 echo {args['duts'][index].hrid} transition {self.collection_names[index]}<br>"
                f"T2={fit_params['Decay']} us")

            data.append(trace_line)

        layout = go.Layout(
            title=title,
            xaxis={'title': 'Time (us)'},
            yaxis={'title': 'Population'},
            plot_bgcolor='white',
            showlegend=True
        )

        # Combine the traces and layout into a figure
        fig = go.Figure(data=data, layout=layout)

        return fig
