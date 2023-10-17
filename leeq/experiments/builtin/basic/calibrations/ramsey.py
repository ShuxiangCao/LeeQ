from datetime import datetime

from plotly.subplots import make_subplots

from labchronicle import register_browser_function, log_and_record
from leeq.utils import setup_logging
from leeq import Experiment, SweepParametersSideEffectFactory, Sweeper
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
from leeq.utils.compatibility import *
from leeq.theory import fits
from plotly import graph_objects as go

logger = setup_logging(__name__)

from typing import Any, Optional, Tuple, Dict, Union
import numpy as np
import datetime
import copy


class SimpleRamseyMultilevel(Experiment):
    """
    Represents a simple Ramsey experiment with multilevel frequency sweeps.
    This version has changed the step size from 0.001 to 0.005.
    """

    @log_and_record
    def run(self,
            qubit: Any,  # Replace 'Any' with the actual type of qubit
            collection_name: str = 'f01',
            mprim_index: int = 0,
            initial_lpb: Optional[Any] = None,  # Replace 'Any' with the actual type
            start: float = 0.0,
            stop: float = 1.0,
            step: float = 0.005,
            set_offset: float = 10.0,
            update: bool = True) -> None:
        """
        Run the Ramsey experiment.

        Parameters:
            qubit: The qubit on which the experiment is performed.
            collection_name: The name of the frequency collection (e.g., 'f01').
            mprim_index: The index of the measurement primitive.
            initial_lpb: Initial set of commands, if any.
            start: The start frequency for the sweep.
            stop: The stop frequency for the sweep.
            step: The step size for the frequency sweep.
            set_offset: The frequency offset.
            update: Whether to update parameters after the experiment.

        Returns:
            None
        """
        c1q = qubit.get_c1(collection_name)  # Retrieve the control object
        self.set_offset = set_offset
        self.step = step

        # Define the levels for the sweep based on the collection name
        start_level = int(collection_name[1])
        end_level = int(collection_name[2])
        self.level_diff = end_level - start_level

        # Save original frequency
        original_freq = c1q['Xp'].freq
        self.original_freq = original_freq

        # Create a delay primitive
        delay = prims.Delay(0)

        # Update the frequency with the calculated offset
        c1q.update_parameters(freq=original_freq + set_offset / self.level_diff)

        # Setup the sweeper for the Ramsey experiment
        swp = sweeper(np.arange, n_kwargs={'start': start, 'stop': stop, 'step': step},
                      params=[sparam.func(delay.update_parameters, {}, 'delay_time')])

        # Get the measurement primitive
        mprim = qubit.get_measurement_prim_intlist(mprim_index)
        self.mp = mprim

        # Construct the logic primitive block
        lpb = c1q['Xp'] + delay + c1q['Xm'] + mprim
        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        # Execute the basic experiment routine
        basic(lpb, swp, '<z>')
        self.data = np.squeeze(mprim.result())
        self.update = update

        # Analyze data if update is true
        if update:
            self.analyze_data()
            c1q.update_parameters(freq=self.frequency_guess)
        else:
            c1q.update_parameters(freq=original_freq)

    def live_plots(self, step_no: Optional[Tuple[int]] = None) -> go.Figure:
        """
        Generate live plots for the experiment.

        Parameters:
            step_no: The current step number, if applicable.

        Returns:
            A plotly graph object containing the live data.
        """
        args = self.retrieve_args(self.run)
        data = np.squeeze(self.mp.result())
        t = np.arange(args['start'], args['stop'], args['step'])

        # If a specific step number is provided, slice the data
        if step_no is not None:
            t = t[:step_no[0]]
            data = data[:step_no[0]]

        # Create and return the figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=data, mode='lines+markers', name='data'))
        fig.update_layout(title=f"Ramsey {args['qubit'].hrid} transition {args['collection_name']}",
                          xaxis_title="Time (us)",
                          yaxis_title="<z>",
                          legend_title="Legend",
                          font=dict(
                              family="Courier New, monospace",
                              size=12,
                              color="Black"
                          ),
                          plot_bgcolor="white")
        return fig

    def analyze_data(self) -> None:
        """
        Analyze the experiment data to extract frequency and error information.

        Returns:
            None
        """
        args = self.retrieve_args(self.run)
        try:
            # Fit the data to an exponential decay model to extract frequency
            # Fit the data using a predefined fitting function
            from leeq.theory.fits import fit_1d_freq_exp_with_cov
            self.fit_params = fit_1d_freq_exp_with_cov(self.data, dt=args['step'])
            fitted_freq_offset = (self.fit_params['Frequency'][0] - self.set_offset) / self.level_diff
            self.frequency_guess = self.original_freq - fitted_freq_offset
            self.error_bar = self.fit_params['Frequency'][1]

        except Exception as e:
            # In case of fit failure, default the frequency guess and error
            self.frequency_guess = 0
            self.error_bar = np.inf

    def dump_results_and_configuration(self) -> Tuple[
        float, float, Any, Dict[str, Union[float, str]], datetime.datetime]:
        """
        Dump the results and configuration of the experiment.

        Returns:
            A tuple containing the guessed frequency, error bar, trace, arguments, and current timestamp.
        """
        args = copy.copy(self.retrieve_args(self.run))
        del args['initial_lpb']
        args['drive_freq'] = args['qubit'].get_c1(args['collection_name'])['X'].freq
        args['qubit'] = args['qubit'].hrid
        return self.frequency_guess, self.error_bar, self.trace, args, datetime.datetime.now()

    @register_browser_function(available_after=('run',))
    def plot(self) -> go.Figure:
        """
        Plots the Ramsey decay with fitted curve using data from the experiment.

        This method uses Plotly for generating the plot. It analyzes the data, performs
        curve fitting, and then plots the actual data along with the fitted curve.
        """
        self.analyze_data()
        print(self.frequency_guess)

        args = self.retrieve_args(self.run)

        # Generate time points based on the experiment arguments
        time_points = np.arange(args['start'], args['stop'], args['step'])
        time_points_interpolate = np.arange(args['start'], args['stop'], args['step'] / 10)

        # Extract fitting parameters
        frequency = self.fit_params['Frequency'][0]
        amplitude = self.fit_params['Amplitude'][0]
        phase = self.fit_params['Phase'][0] - 2.0 * np.pi * frequency * args['start']
        offset = self.fit_params['Offset'][0]
        decay = self.fit_params['Decay'][0]

        # Generate the fitted curve
        fitted_curve = amplitude * np.exp(-time_points_interpolate / decay) * \
                       np.sin(2.0 * np.pi * frequency * time_points_interpolate + phase) + offset

        # Create a plot using Plotly
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=time_points, y=self.data, mode='markers', name='Data'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=time_points_interpolate, y=fitted_curve, mode='lines', name='Fit'),
                      row=1, col=1)

        # Set plot layout details
        title_text = f"Ramsey decay {args['qubit'].hrid} transition {args['collection_name']}: \n" \
                     f"{decay} ± {self.fit_params['Decay'][1]} us"
        fig.update_layout(title_text=title_text,
                          xaxis_title=f"Time (us) \n Frequency: {frequency} ± {self.fit_params['Frequency'][1]}",
                          yaxis_title="<z>",
                          plot_bgcolor="white")
        return fig

    def plot_fft(self, plot_range: Tuple[float, float] = (0.05, 1)) -> go.Figure:
        """
        Plots the Fast Fourier Transform (FFT) of the data from the Ramsey experiment.

        Parameters:
        plot_range: Tuple[float, float], optional
            The frequency range for the plot. Defaults to (0.05, 1).

        This method uses Plotly for plotting. It computes the FFT of the data and plots the
        spectrum within the specified range.
        """
        self.analyze_data()
        data = self.data
        args = self.retrieve_args(self.run)
        time_step = args['step']

        # Compute the (real) FFT of the data
        fft_magnitudes = np.abs(np.fft.rfft(data))
        frequencies = np.fft.rfftfreq(len(data), time_step)

        # Apply frequency range mask
        mask = (frequencies > plot_range[0]) & (frequencies < plot_range[1])

        # Create a plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frequencies[mask], y=fft_magnitudes[mask], mode='lines'))

        # Set plot layout details
        fig.update_layout(title='Ramsey Spectrum',
                          xaxis_title='Frequency [MHz]',
                          yaxis_title='Strength',
                          plot_bgcolor="white")
        return fig
