import numpy as np
from matplotlib import pyplot as plt
from typing import Optional, Any, Dict, List
from typing import Optional, Any, Dict, List
import numpy as np
from plotly import graph_objects as go

from labchronicle import register_browser_function, log_and_record
from leeq import Experiment, Sweeper
from leeq.theory.fits import fit_1d_freq_exp_with_cov, fit_exp_decay_with_cov
from leeq.utils.compatibility import *


class SpinEchoMultiLevel(Experiment):  # Class names should follow the CapWords convention
    """
    A class used to represent the SimpleSpinEchoMultiLevel experiment.

    Methods
    -------
    run(...)
        Runs the experiment.

    plot_echo()
        Plots the results of the echo experiment.
    """

    @log_and_record
    def run(
            self,
            qubit: Any,  # Replace 'Any' with the actual type of qubit
            collection_name: str = 'f01',
            mprim_index: int = 0,
            free_evolution_time: float = 100.0,
            time_resolution: float = 4.0,
            start: float = 0.0,
            initial_lpb: Optional[Any] = None,  # Replace 'Any' with the actual type
    ) -> None:
        """
        Run the SimpleSpinEchoMultiLevel experiment.

        Parameters
        ----------
        qubit : Any
            The qubit to be used in the experiment.
        collection_name : str
            The name of the pulse collection.
        mprim_index : int
            Index of the measurement primitive.
        free_evolution_time : float, optional
            The free evolution time, by default 0.0.
        time_resolution : float, optional
            The time resolution for the experiment, by default 1.0.
        start : float, optional
            Start time of the sweep, by default 0.0.
        initial_lpb : Any, optional
            The initial local pulse builder, by default None.
        """

        c1 = qubit.get_c1(collection_name)
        mp = qubit.get_measurement_prim_intlist(mprim_index)
        delay = prims.Delay(0)

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

        basic(lpb, swp, 'p(0)')

        self.trace = np.squeeze(mp.result())

    @register_browser_function(available_after=(run,))
    def plot_echo(self, fit=True, step_no=None) -> go.Figure:
        """
        Plot the results of the echo experiment using Plotly.
        """

        trace = np.squeeze(self.mp.result())
        args = self.retrieve_args(self.run)

        t = np.arange(0, args['free_evolution_time'], args['time_resolution'])

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

        title = f"T2 decay {args['qubit'].hrid} transition {args['collection_name']}"

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
            title = (f"T2 echo {args['qubit'].hrid} transition {args['collection_name']}<br>"
                     f"T2={fit_params['Decay'][0]:.2f} ± {fit_params['Decay'][1]:.2f} us")

            data = [trace_scatter, trace_line]

        layout = go.Layout(
            title=title,
            xaxis=dict(title='Time (us)'),
            yaxis=dict(title='P(0)'),
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