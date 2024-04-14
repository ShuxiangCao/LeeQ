from typing import Optional, Any, Dict, List, Union
import numpy as np
from plotly import graph_objects as go

from labchronicle import register_browser_function, log_and_record
from leeq import Experiment, Sweeper
from leeq.theory.fits import fit_1d_freq_exp_with_cov, fit_exp_decay_with_cov
from leeq.utils.compatibility import *
from leeq.utils.compatibility.prims import SweepLPB


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

    @register_browser_function(available_after=(run,))
    def plot_t1(self, fit=True, step_no=None) -> go.Figure:
        """
        Plot the T1 decay graph based on the trace and fit parameters using Plotly.

        Parameters:
        fit (bool): Whether to fit the trace. Defaults to True.
        step_no (Tuple[int]): Number of steps to plot.

        Returns:
        go.Figure: The Plotly figure object.
        """
        self.trace = None
        self.fit_params = {}  # Initialize as an empty dictionary or suitable default value

        args = self.retrieve_args(self.run)

        t = np.arange(0, args['time_length'], args['time_resolution'])
        trace = np.squeeze(self.mp.result())

        if step_no is not None:
            t = t[:step_no[0]]
            trace = trace[:step_no[0]]

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

        title = f"T1 decay {args['qubit'].hrid} transition {args['collection_name']}"

        data = [trace_scatter]

        if fit:
            fit_params = fit_exp_decay_with_cov(trace, args['time_resolution'])

            self.fit_params = fit_params

            trace_line = go.Scatter(
                x=t,
                y=fit_params['Amplitude'][0] * np.exp(-t / fit_params['Decay'][0]) + fit_params['Offset'][0],
                mode='lines',
                line=dict(
                    color='blue'
                ),
                name='Decay fit'
            )
            title = (
                f"T1 decay {args['qubit'].hrid} transition {args['collection_name']}<br>"
                f"T1={fit_params['Decay'][0]:.2f} ± {fit_params['Decay'][1]:.2f} us")

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

        args = self.retrieve_args(self.run)

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
                y=fit_params['Amplitude'][0] * np.exp(-t / fit_params['Decay'][0]) + fit_params['Offset'][0],
                mode='lines',
                line=dict(
                    color='blue'
                ),
                name='Decay fit'
            )
            title = (
                f"T1 decay {args['duts'][i].hrid} transition {self.collection_names[i]}<br>"
                f"T1={fit_params['Decay'][0]:.2f} ± {fit_params['Decay'][1]:.2f} us")

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
    color_set = ['orange', 'b', 'g', 'r']

    title_dict = {
        '00': rf"Qubit prepared to |0$\rangle$",
        '01': rf"Qubit prepared to |1$\rangle$",
        '12': rf"Qubit prepared to |2$\rangle$",
        '23': rf"Qubit prepared to |3$\rangle$",
        0: rf"Qubit prepared to |0$\rangle$",
        1: rf"Qubit prepared to |1$\rangle$",
        2: rf"Qubit prepared to |2$\rangle$",
        3: rf"Qubit prepared to |3$\rangle$",
    }

    @log_and_record
    def run(self,
            duts: List[Any],
            time_length: float = 200,
            time_resolution: float = 5,
            mprim_indexes: Union[int, List[int]] = 2
            ):
        """
        Run the T1 experiment with the specified parameters.

        Parameters:
        duts (List[Any]): A list of qubit objects to be used in the experiment.
        time_length (float): Total time length of the experiment in microseconds.
        time_resolution (float): Time resolution for the experiment in microseconds.
        mprim_indexes (Union[int, List[int]]): Index of the measurement primitive.
        """

        self.time_length = time_length
        self.time_resolution = time_resolution

        if isinstance(mprim_indexes, int):
            mprim_indexes = [mprim_indexes] * len(duts)

        c1_01s = [dut.get_c1('f01') for dut in duts]
        c1_12s = [dut.get_c1('f12') for dut in duts]
        c1_23s = [dut.get_c1('f23') for dut in duts]

        c1_01_pulses = prims.ParallelLPB([c1['X'] for c1 in c1_01s])
        c1_12_pulses = prims.ParallelLPB([c1['X'] for c1 in c1_12s])
        c1_23_pulses = prims.ParallelLPB([c1['X'] for c1 in c1_23s])

        delay = prims.Delay(0)

        lpb = SweepLPB([
            c1_01_pulses,
            c1_01_pulses + c1_12_pulses,
            c1_01_pulses + c1_12_pulses + c1_23_pulses]
        )

        swp_lpb = sweeper.from_sweep_lpb(lpb)

        delay = prims.Delay(0)

        lpb = lpb + delay
        swp = sweeper(
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

        mprims = [dut.get_measurement_prim_intlist(mprim_index) for mprim_index,dut in zip(mprim_indexes,duts)]

        lpb = lpb + prims.ParallelLPB(mprims)

        basic(lpb, swp_time + swp_lpb, '<zs>')

        self.results = [
            np.squeeze(mprim.result()) for mprim in mprims
        ]

    def analyze_data(self):
        pass
