from typing import Optional, Any, Dict, List
import numpy as np
from plotly import graph_objects as go

from labchronicle import register_browser_function, log_and_record
from leeq import Experiment, Sweeper
from leeq.theory.fits import fit_1d_freq_exp_with_cov, fit_exp_decay_with_cov
from leeq.utils.compatibility import *


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
            initial_lpb: Optional[Any] = None,  # Add the expected type for 'initial_lpb' instead of Any
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

            trace_line = go.Scatter(
                x=t,
                y=fit_params['Amplitude'][0] * np.exp(-t / fit_params['Decay'][0]) + fit_params['Offset'][0],
                mode='lines',
                line=dict(
                    color='blue'
                ),
                name='Decay fit'
            )
            title = (f"T1 decay {args['qubit'].hrid} transition {args['collection_name']}<br>"
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
