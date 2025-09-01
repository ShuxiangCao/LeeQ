from typing import Any, List, Optional, Union

import numpy as np
from k_agents.inspection.decorator import text_inspection, visual_inspection
from plotly import graph_objects as go

from leeq import Experiment, Sweeper
from leeq.chronicle import log_and_record, register_browser_function
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.theory.fits import fit_exp_decay_with_cov
from leeq.theory.fits.multilevel_decay import fit_decay as fit_multilevel_decay
from leeq.theory.fits.multilevel_decay import plot
from leeq.theory.utils import to_dense_probabilities
from leeq.utils import setup_logging
from leeq.utils.compatibility import *
from leeq.utils.compatibility.prims import SweepLPB

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

    EPII_INFO = {
        "name": "SimpleT1",
        "description": "Single qubit T1 relaxation time measurement experiment",
        "purpose": "Measures the characteristic time T1 for a qubit to relax from the excited state |1> to the ground state |0>. This is a fundamental qubit characterization that determines the energy relaxation time and helps assess qubit quality. The experiment prepares the qubit in |1> state and measures population decay over time.",
        "attributes": {
            "trace": {
                "type": "np.ndarray[float]",
                "description": "Population measurements as function of delay time",
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
                    "Decay": "ufloat - T1 time constant in microseconds",
                    "Amplitude": "ufloat - Initial amplitude of decay",
                    "Offset": "ufloat - Steady-state offset value"
                }
            }
        },
        "notes": [
            "Time length should be approximately 5 times the expected T1 value",
            "Typical T1 values range from 10-200 microseconds for superconducting qubits",
            "The experiment fits data to: A * exp(-t/T1) + Offset",
            "In simulation mode, virtual qubit T1 value is used with optional sampling noise"
        ]
    }

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
        """Execute the T1 experiment in simulation mode.

        Parameters
        ----------
        qubit : Any
            The qubit object to be used in the experiment.
        collection_name : str, optional
            The collection name for the qubit transition. Default: 'f01'
        initial_lpb : Optional[Any], optional
            Initial list of pulse blocks (LPB). Default: None
        mprim_index : int, optional
            Index of the measurement primitive. Default: 0
        time_length : float, optional
            Total time length of the experiment in microseconds. Default: 100.0
        time_resolution : float, optional
            Time resolution for the experiment in microseconds. Default: 1.0

        Returns
        -------
        None
            Updates the instance's trace attribute with simulated data.
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
        """Execute the T1 experiment on hardware.

        Parameters
        ----------
        qubit : Any
            The qubit object to be used in the experiment.
        collection_name : str, optional
            The collection name for the qubit transition. Default: 'f01'
        initial_lpb : Optional[Any], optional
            Initial list of pulse blocks (LPB). Default: None
        mprim_index : int, optional
            Index of the measurement primitive. Default: 0
        time_length : float, optional
            Total time length of the experiment in microseconds. Default: 100.0
        time_resolution : float, optional
            Time resolution for the experiment in microseconds. Default: 1.0

        Returns
        -------
        None
            Updates the instance's trace attribute with measured data.
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

        np.arange(0, args['time_length'], args['time_resolution'])
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
            marker={
                # symbol='x',
                'size': 5,
                # color='blue'
            },
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
                line={
                    'color': 'blue'
                },
                visible='legendonly',
                name='Decay fit'
            )
            title = (
                f"T1 decay {args['qubit'].hrid} transition {args['collection_name']}<br>"
                f"T1={fit_params['Decay']} us")

            data = [trace_scatter, trace_line]

        layout = go.Layout(
            title=title,
            xaxis={'title': 'Time (us)'},
            yaxis={'title': 'P(1)'},
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

    EPII_INFO = {
        "name": "MultiQubitT1",
        "description": "Parallel T1 relaxation time measurement for multiple qubits",
        "purpose": "Simultaneously measures T1 relaxation times for multiple qubits to improve experimental efficiency. Each qubit is prepared in |1> state and population decay is measured independently but in parallel. This allows rapid characterization of multi-qubit systems.",
        "attributes": {
            "traces": {
                "type": "List[np.ndarray[float]]",
                "description": "Population measurements for each qubit",
                "shape": "List of (n_time_points,) arrays"
            },
            "mps": {
                "type": "List[MeasurementPrimitive]",
                "description": "Measurement primitives for each qubit"
            },
            "collection_names": {
                "type": "List[str]",
                "description": "Collection names for each qubit transition"
            }
        },
        "notes": [
            "All qubits are measured in parallel for efficiency",
            "Each qubit can have different collection names and measurement primitives",
            "Fitting is performed independently for each qubit",
            "Time parameters apply to all qubits equally"
        ]
    }

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
        """Execute the multi qubit T1 experiment on hardware.

        Parameters
        ----------
        duts : List[Any]
            A list of qubit objects to be used in the experiment.
        collection_names : Union[str, List[str]], optional
            The collection name(s) for the qubit transitions. Default: 'f01'
        initial_lpb : Optional[Any], optional
            Initial list of pulse blocks (LPB). Default: None
        mprim_indexes : Union[int, List[int]], optional
            Index(es) of the measurement primitive(s). Default: 0
        time_length : float, optional
            Total time length of the experiment in microseconds. Default: 100.0
        time_resolution : float, optional
            Time resolution for the experiment in microseconds. Default: 1.0

        Returns
        -------
        None
            Updates the instance's traces attribute with measured data.
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
        mps = [
            qubit.get_measurement_prim_intlist(mprim_index) for qubit,
            mprim_index in zip(
                duts,
                mprim_indexes, strict=False)]
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

    @log_and_record(overwrite_func_name='MultiQubitT1.run')
    def run_simulated(self,
                      duts: List[Any],
                      collection_names: Union[str, List[str]] = 'f01',
                      initial_lpb: Optional[Any] = None,
                      mprim_indexes: int = 0,
                      time_length: float = 100.0,
                      time_resolution: float = 1.0
                      ) -> None:
        """Execute the multi qubit T1 experiment in simulation mode.

        Parameters
        ----------
        duts : List[Any]
            A list of qubit objects to be used in the experiment.
        collection_names : Union[str, List[str]], optional
            The collection name(s) for the qubit transitions. Default: 'f01'
        initial_lpb : Optional[Any], optional
            Initial list of pulse blocks (LPB). Default: None
        mprim_indexes : Union[int, List[int]], optional
            Index(es) of the measurement primitive(s). Default: 0
        time_length : float, optional
            Total time length of the experiment in microseconds. Default: 100.0
        time_resolution : float, optional
            Time resolution for the experiment in microseconds. Default: 1.0

        Returns
        -------
        None
            Updates the instance's traces attribute with simulated data.
        """
        if isinstance(collection_names, str):
            collection_names = [collection_names] * len(duts)

        self.collection_names = collection_names

        # Get setup and virtual qubits
        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()

        # Create time array
        sweep_range = np.arange(0.0, time_length, time_resolution)

        # Simulate T1 decay for each qubit independently
        self.traces = []
        for dut in duts:
            virtual_qubit = simulator_setup.get_virtual_qubit(dut)
            if virtual_qubit is None:
                raise ValueError(f"No virtual qubit found for {dut}")

            # Get T1 value for this qubit
            t1 = virtual_qubit.t1  # in microseconds

            # Calculate T1 decay
            data = np.exp(-sweep_range / t1)

            # If sampling noise is enabled, simulate the noise
            if setup().status().get_param('Sampling_Noise'):
                # Get the number of shots used in the simulation
                shot_number = setup().status().get_param('Shot_Number')
                # Generate binomial distribution to simulate sampling noise
                data = np.random.binomial(shot_number, data) / shot_number

            self.traces.append(data)

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
            marker={
                'symbol': 'x',
                'size': 10,
                'color': 'blue'
            },
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
                line={
                    'color': 'blue'
                },
                name='Decay fit'
            )
            title = (
                f"T1 decay {args['duts'][i].hrid} transition {self.collection_names[i]}<br>"
                f"T1={fit_params['Decay']} us")

            data = [trace_scatter, trace_line]

        layout = go.Layout(
            title=title,
            xaxis={'title': 'Time (us)'},
            yaxis={'title': 'P(1)'},
            plot_bgcolor='white',
            showlegend=True
        )

        fig = go.Figure(data=data, layout=layout)

        return fig


class MultiQuditT1Decay(Experiment):
    EPII_INFO = {
        "name": "MultiQuditT1Decay",
        "description": "Multi-level T1 decay measurement for qudit systems",
        "purpose": "Measures T1 relaxation times between multiple energy levels in qudit systems (up to 4 levels). This experiment characterizes the decay rates between all adjacent energy levels, providing comprehensive information about multi-level relaxation dynamics. Supports measurement error mitigation through assignment matrix calibration.",
        "attributes": {
            "results": {
                "type": "List[np.ndarray]",
                "description": "Raw measurement results for each qudit",
                "shape": "List of arrays with shape depending on measurement"
            },
            "probs": {
                "type": "List[np.ndarray]",
                "description": "Probability distributions for each qudit over time",
                "shape": "List of (n_levels, n_time_points, n_initial_states) arrays"
            },
            "fit_params": {
                "type": "List[Tuple[np.ndarray, np.ndarray]]",
                "description": "Fitted initial state and gamma matrix for each qudit",
                "keys": {
                    "initial_state": "np.ndarray - Initial population distribution",
                    "gamma": "np.ndarray - Transition rate matrix"
                }
            },
            "t1_list": {
                "type": "List[List[float]]",
                "description": "Extracted T1 times for each transition of each qudit",
                "shape": "List of lists with T1 values in microseconds"
            },
            "time_length": {
                "type": "float",
                "description": "Total time length of the experiment in microseconds"
            },
            "time_resolution": {
                "type": "float",
                "description": "Time step between measurements in microseconds"
            },
            "max_level": {
                "type": "int",
                "description": "Maximum energy level measured (0-indexed)"
            },
            "assignment_calibration": {
                "type": "Optional[CalibrateSingleDutAssignmentMatrices]",
                "description": "Assignment matrix calibration for measurement mitigation"
            }
        },
        "notes": [
            "Supports up to 4 energy levels (0, 1, 2, 3)",
            "Measurement mitigation can be enabled to correct for readout errors",
            "Fits multi-level decay using master equation approach",
            "Each level is prepared sequentially to measure all decay pathways"
        ]
    }

    @log_and_record(overwrite_func_name='MultiQuditT1Decay.run')
    def run_simulated(self,
                      duts: List[Any],
                      time_length: float = 200,
                      time_resolution: float = 5,
                      mprim_indexes: Union[int, List[int]] = 2,
                      max_level: int = 3,
                      measurement_mitigation: bool = False
                      ):
        """Execute the multi-level T1 experiment in simulation mode.

        Parameters
        ----------
        duts : List[Any]
            A list of qubit objects to be used in the experiment.
        time_length : float, optional
            Total time length of the experiment in microseconds. Default: 200
        time_resolution : float, optional
            Time resolution for the experiment in microseconds. Default: 5
        mprim_indexes : Union[int, List[int]], optional
            Index(es) of the measurement primitive(s). Default: 2
        max_level : int, optional
            The highest energy level to measure (0-indexed). Default: 3
        measurement_mitigation : bool, optional
            Whether to apply measurement mitigation using assignment matrix. Default: False

        Returns
        -------
        None
            Updates the instance's results attribute with simulated data.
        """
        # Simulation implementation placeholder
        # In a real implementation, this would simulate multi-level decay
        raise NotImplementedError("Simulation for MultiQuditT1Decay not yet implemented")

    @log_and_record
    def run(self,
            duts: List[Any],
            time_length: float = 200,
            time_resolution: float = 5,
            mprim_indexes: Union[int, List[int]] = 2,
            max_level: int = 3,
            measurement_mitigation: bool = False
            ):
        """Execute the multi-level T1 experiment on hardware.

        Parameters
        ----------
        duts : List[Any]
            A list of qubit objects to be used in the experiment.
        time_length : float, optional
            Total time length of the experiment in microseconds. Default: 200
        time_resolution : float, optional
            Time resolution for the experiment in microseconds. Default: 5
        mprim_indexes : Union[int, List[int]], optional
            Index(es) of the measurement primitive(s). Default: 2
        max_level : int, optional
            The highest energy level to measure (0-indexed). Default: 3
        measurement_mitigation : bool, optional
            Whether to apply measurement mitigation using assignment matrix. Default: False

        Returns
        -------
        None
            Updates the instance's results attribute with measured data.
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

        mprims = [dut.get_measurement_prim_intlist(mprim_index) for mprim_index, dut in zip(mprim_indexes, duts, strict=False)]

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

        for i, _prob in enumerate(self.probs):
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
            self.t1_list[i]
            fig = plot(probs=probs, time_length=self.time_length, time_resolution=self.time_resolution,
                       initial_distribution=fit_param[0], gamma=fit_param[1])
            for i in range(self.max_level):
                pass
            fig.show()
