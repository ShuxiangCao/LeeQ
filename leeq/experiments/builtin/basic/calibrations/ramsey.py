from datetime import datetime

from plotly.subplots import make_subplots

from labchronicle import register_browser_function, log_and_record

from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
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
            print(f"Frequency updated: {self.frequency_guess} MHz")
        else:
            c1q.update_parameters(freq=original_freq)

    @log_and_record(overwrite_func_name='SimpleRamseyMultilevel.run')
    def run_simulated(self,
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

        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()
        virtual_transmon = simulator_setup.get_virtual_qubit(qubit)

        c1 = qubit.get_c1(collection_name)

        f_q = virtual_transmon.qubit_frequency
        f_d = c1['X'].freq
        f_o = set_offset
        self.set_offset = set_offset

        # Save original frequency
        original_freq = c1['Xp'].freq
        self.original_freq = original_freq

        # Define the levels for the sweep based on the collection name
        start_level = int(collection_name[1])
        end_level = int(collection_name[2])
        self.level_diff = end_level - start_level

        t = np.arange(start, stop, step)

        decay_rate = 0.1  # TODO: Implement decay rate

        # Ramsey fringes formula

        f_o_actual = f_q - (f_d + f_o)

        ramsey_fringes = (1 + np.cos(2 * np.pi * f_o_actual * t) * np.exp(-decay_rate * t)) / 2

        self.data = ramsey_fringes

        # If sampling noise is enabled, simulate the noise
        if setup().status().get_param('Sampling_Noise'):
            # Get the number of shot used in the simulation
            shot_number = setup().status().get_param('Shot_Number')

            # generate binomial distribution of the result to simulate the sampling noise
            self.data = np.random.binomial(shot_number, self.data) / shot_number

        self.data = self.data * 2 - 1

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
        args = self.retrieve_args(self.run)

        # Generate time points based on the experiment arguments
        time_points = np.arange(args['start'], args['stop'], args['step'])
        time_points_interpolate = np.arange(args['start'], args['stop'], args['step'] / 10)

        # Create a plot using Plotly
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=time_points, y=self.data, mode='markers', name='Data'),
                      row=1, col=1)

        if hasattr(self,'fit_params'):

            # Extract fitting parameters
            frequency = self.fit_params['Frequency'][0]
            amplitude = self.fit_params['Amplitude'][0]
            phase = self.fit_params['Phase'][0] - 2.0 * np.pi * frequency * args['start']
            offset = self.fit_params['Offset'][0]
            decay = self.fit_params['Decay'][0]

            # Generate the fitted curve
            fitted_curve = amplitude * np.exp(-time_points_interpolate / decay) * \
                           np.sin(2.0 * np.pi * frequency * time_points_interpolate + phase) + offset

            fig.add_trace(go.Scatter(x=time_points_interpolate, y=fitted_curve, mode='lines', name='Fit'),
                          row=1, col=1)

            # Set plot layout details
            title_text = f"Ramsey decay {args['qubit'].hrid} transition {args['collection_name']}: <br>" \
                         f"{decay} ± {self.fit_params['Decay'][1]} us"
            fig.update_layout(title_text=title_text,
                              xaxis_title=f"Time (us) <br> Frequency: {frequency} ± {self.fit_params['Frequency'][1]}",
                              yaxis_title="<z>",
                              plot_bgcolor="white")

        else:
            # Set plot layout details
            title_text = f"Ramsey decay {args['qubit'].hrid} transition {args['collection_name']}: <br>" \
                         f"Fit failed"
            fig.update_layout(title_text=title_text,
                              xaxis_title=f"Time (us)",
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


class MultiQubitRamseyMultilevel(Experiment):
    """
    Implement a multi-qubit Ramsey experiment with multilevel frequency sweeps.
    This version has changed the step size from 0.001 to 0.005.
    """

    @log_and_record
    def run(self,
            duts: list[Any],
            collection_names: Union[list[str], str] = 'f01',
            mprim_indexes: Union[int, list[int]] = 0,
            initial_lpb: Optional[Any] = None,
            start: float = 0.0,
            stop: float = 1.0,
            step: float = 0.005,
            set_offset: float = 10.0,
            update: bool = True) -> None:
        """
        Run the Ramsey experiment.

        Parameters:
            duts: The DUTs on which the experiment is performed.
            collection_names: The name of the frequency collection (e.g., 'f01'). If a single string is provided,
                it is used for all DUTs. If a list of strings is provided, it should have the same length as the DUTs.
            mprim_indexes: The index of the measurement primitive.
                If a single integer is provided, it is used for all DUTs. If a list of integers is provided,
                it should have the same length as the DUTs.
            initial_lpb: Initial set of commands, if any.
            start: The start frequency for the sweep.
            stop: The stop frequency for the sweep.
            step: The step size for the frequency sweep.
            set_offset: The frequency offset.
            update: Whether to update parameters after the experiment.

        Returns:
            None
        """

        if isinstance(collection_names, str):
            collection_names = [collection_names] * len(duts)
        if isinstance(mprim_indexes, int):
            mprim_indexes = [mprim_indexes] * len(duts)

        self.collection_names = collection_names
        # Make sure the collection names and mprim indexes have the same length as the DUTs
        assert len(duts) == len(collection_names) == len(mprim_indexes), \
            "The number of DUTs, collection names, and mprim indexes must be the same."

        c1s = [qubit.get_c1(collection_name) for qubit, collection_name in
               zip(duts, collection_names)]  # Retrieve the control object
        self.set_offset = set_offset
        self.step = step

        # Define the levels for the sweep based on the collection name

        self.level_diffs = []
        for collection_name in collection_names:
            start_level = int(collection_name[1])
            end_level = int(collection_name[2])
            self.level_diffs.append(end_level - start_level)

        # Save original frequency
        original_freqs = [c1['Xp'].freq for c1 in c1s]
        self.original_freqs = original_freqs

        # Create a delay primitive
        delay = prims.Delay(0)

        # Update the frequency with the calculated offset
        for i, c1 in enumerate(c1s):
            c1.update_parameters(freq=original_freqs[i] + set_offset / self.level_diffs[i])

        # Setup the sweeper for the Ramsey experiment
        swp = sweeper(np.arange, n_kwargs={'start': start, 'stop': stop, 'step': step},
                      params=[sparam.func(delay.update_parameters, {}, 'delay_time')])

        # Get the measurement primitive
        mprims = [qubit.get_measurement_prim_intlist(mprim_index) for qubit, mprim_index in zip(duts, mprim_indexes)]
        self.mp = mprims

        # Construct the logic primitive block
        lpb = prims.ParallelLPB([c1['Xp'] for c1 in c1s]) + delay + \
              prims.ParallelLPB([c1['Xm'] for c1 in c1s]) + prims.ParallelLPB(mprims)

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        # Execute the basic experiment routine
        basic(lpb, swp, '<z>')
        self.data = [np.squeeze(mprim.result()) for mprim in mprims]
        self.update = update

        # Analyze data if update is true
        if update:
            self.analyze_data()
            for i, c1 in enumerate(c1s):
                c1.update_parameters(freq=self.frequency_guess[i])
                print(f"Frequency updated: {duts[i].hrid} {self.frequency_guess[i]} MHz")
        else:
            for i, c1 in enumerate(c1s):
                c1.update_parameters(freq=original_freqs[i])

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
            self.fit_params = [fit_1d_freq_exp_with_cov(data, dt=args['step']) for data in self.data]
            fitted_freq_offsets = [(self.fit_params[i]['Frequency'][0] - self.set_offset) / self.level_diffs[i] for i in
                                   range(len(self.data))]
            self.frequency_guess = [self.original_freqs[i] - fitted_freq_offset for i, fitted_freq_offset in
                                    enumerate(fitted_freq_offsets)]
            self.error_bar = [x['Frequency'][1] for x in self.fit_params]

        except Exception as e:
            # In case of fit failure, default the frequency guess and error
            logger.warning(f"Fit failed: {e}")
            self.frequency_guess = []
            self.error_bar = [np.inf]

    @register_browser_function(available_after=('run',))
    def plot_all(self):
        """
        Plots the Ramsey decay with fitted curve using data from the experiment.
        """
        self.analyze_data()
        for i in range(len(self.fit_params)):
            fig = self.plot(i)
            fig.show()

    def plot(self, i) -> go.Figure:
        """
        Plots the Ramsey decay with fitted curve using data from the experiment.

        This method uses Plotly for generating the plot. It analyzes the data, performs
        curve fitting, and then plots the actual data along with the fitted curve.

        Parameters:
            i: The index of the qubit for which to plot the data.
        """
        args = self.retrieve_args(self.run)
        fit_params = self.fit_params[i]

        # Generate time points based on the experiment arguments
        time_points = np.arange(args['start'], args['stop'], args['step'])
        time_points_interpolate = np.arange(args['start'], args['stop'], args['step'] / 10)

        # Extract fitting parameters
        frequency = fit_params['Frequency'][0]
        amplitude = fit_params['Amplitude'][0]
        phase = fit_params['Phase'][0] - 2.0 * np.pi * frequency * args['start']
        offset = fit_params['Offset'][0]
        decay = fit_params['Decay'][0]

        # Generate the fitted curve
        fitted_curve = amplitude * np.exp(-time_points_interpolate / decay) * \
                       np.sin(2.0 * np.pi * frequency * time_points_interpolate + phase) + offset

        # Create a plot using Plotly
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=time_points, y=self.data[i], mode='markers', name='Data'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=time_points_interpolate, y=fitted_curve, mode='lines', name='Fit'),
                      row=1, col=1)

        # Set plot layout details
        title_text = f"Ramsey decay {args['duts'][i].hrid} transition {self.collection_names[i]}: <br>" \
                     f"{decay} ± {fit_params['Decay'][1]} us"
        fig.update_layout(title_text=title_text,
                          xaxis_title=f"Time (us) <br> Frequency: {frequency} ± {fit_params['Frequency'][1]}",
                          yaxis_title="<z>",
                          plot_bgcolor="white")
        return fig
