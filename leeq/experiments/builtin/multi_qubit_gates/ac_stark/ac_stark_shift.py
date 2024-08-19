# Conditional AC stark shift induced CZ gate

from datetime import datetime

import numpy as np

from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial, LogicalPrimitiveBlockSweep
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.utils import setup_logging

logger = setup_logging(__name__)

import datetime
import copy

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from labchronicle import log_and_record, register_browser_function
from leeq import Experiment, Sweeper, basic_run, SweepParametersSideEffectFactory
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.utils.compatibility import *

import matplotlib.pyplot as plt

from leeq.utils.compatibility import prims

from leeq.theory.fits import *

from scipy.optimize import curve_fit, OptimizeWarning

from qutip import Bloch

from typing import List, Any

import pandas as pd
from IPython.display import display


# from ..characterization import *
# from ..tomography import *

# Conditional Stark Spectroscopy
class StarkSingleQubitT1(experiment):
    # Perform a T1 experiment applying a Stark shifting drive instead of the delay time. Based on SimpleT1()
    @log_and_record
    def run(self,
            qubit: Any,  # Add the expected type for 'qubit' instead of Any
            collection_name: str = 'f01',
            initial_lpb: Optional[Any] = None,  # Add the expected type for 'initial_lpb' instead of Any
            mprim_index: int = 0,
            start=0, stop=3, step=0.03,
            stark_offset=50,
            amp=0.1,
            width=400,
            rise=0.01,
            trunc=1.2):

        self.width = 0
        self.start = start
        self.stop = stop
        self.step = step
        self.stark_offset = stark_offset

        c1 = qubit.get_c1(collection_name)
        mp = qubit.get_measurement_prim_intlist(mprim_index)

        self.mp = mp

        self.original_freq = c1['Xp'].freq
        self.frequency = self.original_freq + self.stark_offset

        cs_pulse = c1['X'].clone()
        cs_pulse.update_pulse_args(amp=amp, freq=self.frequency, phase=0., shape='soft_square', width=self.stop,
                                   rise=rise, trunc=trunc)

        lpb = c1['X'] + cs_pulse + mp

        if initial_lpb:
            lpb = initial_lpb + lpb

        swpparams = [
            sparam.func(cs_pulse.update_pulse_args, {}, 'width'),
        ]

        swp = sweeper(np.arange, n_kwargs={'start': 0.0, 'stop': self.stop, 'step': self.step},
                      params=swpparams)

        basic(lpb, swp=swp, basis="<z>")
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

        t = np.arange(0, args['stop'], args['step'])
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
            fit_params = self.fit_exp_decay_with_cov(trace, args[
                'step'])  # Assuming fit_exp_decay_with_cov is a method of the class

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

    def fit_exp_decay_with_cov(self, trace, time_resolution):
        # Example implementation of the fit_exp_decay_with_cov function
        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        t = np.arange(0, len(trace) * time_resolution, time_resolution)
        try:
            popt, pcov = curve_fit(exp_decay, t, trace, maxfev=2400)
            A, tau, C = popt
            perr = np.sqrt(np.diag(pcov))
            return {'Amplitude': (A, perr[0]), 'Decay': (tau, perr[1]), 'Offset': (C, perr[2])}
        except (OptimizeWarning, RuntimeError) as e:
            print(f"Optimization warning or error: {e}")
            return {'Amplitude': (np.nan, np.nan), 'Decay': (np.nan, np.nan), 'Offset': (np.nan, np.nan)}


class StarkTwoQubitsSWAP(experiment):
    # Perform a Stark Shifted T1 experiment on one qubit while also measuring another
    @log_and_record
    def run(self, qubits, amp, rise=0.01, start=0, stop=3, step=0.03,
            stark_offset=50, initial_lpb=None,
            trunc=1.2):

        self.duts = qubits
        self.stark_offset = stark_offset
        self.amp_control = amp
        self.phase = 0
        self.width = 0
        self.start = start
        self.stop = stop
        self.step = step

        c1_control = self.duts[0].get_default_c1()  # the qubit to be stark shifted and T1 performed on
        c1_target = self.duts[1].get_default_c1()  # the qubit which will just be measured at the same time

        self.original_freq = c1_control['Xp'].freq
        self.frequency = self.original_freq + self.stark_offset

        mprim_control = self.duts[0].get_measurement_prim_intlist(0)
        mprim_target = self.duts[1].get_measurement_prim_intlist(0)

        cs_pulse = c1_control['X'].clone()
        cs_pulse.update_pulse_args(amp=self.amp_control, freq=self.frequency, phase=0., shape='soft_square',
                                   width=self.stop, rise=rise, trunc=trunc)

        swpparams = [
            sparam.func(cs_pulse.update_pulse_args, {}, 'width'),
        ]

        swp = sweeper(np.arange, n_kwargs={'start': 0.0, 'stop': self.stop, 'step': self.step},
                      params=swpparams)

        lpb = c1_control['X'] + cs_pulse + mprim_control * mprim_target

        if initial_lpb:
            lpb = initial_lpb + lpb

        basic(lpb, swp, 'p(1)')

        self.result_control = np.asarray(mprim_control.result(), dtype=float).flatten()
        self.result_target = np.asarray(mprim_target.result(), dtype=float).flatten()

    @register_browser_function(available_after=(run,))
    def plot_t1(self):
        args = self.retrieve_args(self.run)

        dark_navy = '#000080'
        dark_purple = '#800080'

        fit_control = self.fit_exp_decay_with_cov(self.result_control, args['step'])
        fit_target = self.fit_exp_decay_with_cov(self.result_target, args['step'])

        t = np.arange(0, args['stop'], args['step'])

        fig, axs = plt.subplots(1, 3, figsize=[25, 5])

        # Plot for result_control
        axs[0].set_title(f"Control T1 decay\nT1={fit_control['Decay'][0]:.2f} ± {fit_control['Decay'][1]:.2f} us")
        axs[0].set_xlabel("Time (us)")
        axs[0].set_ylabel("P(1)")
        axs[0].scatter(t, self.result_control, marker='o', color=dark_navy)
        axs[0].plot(t, fit_control['Amplitude'][0] * np.exp(-t / fit_control['Decay'][0]) + fit_control['Offset'][0],
                    color=dark_navy)

        # Plot for result_target
        axs[1].set_title(f"Target T1 decay\nT1={fit_target['Decay'][0]:.2f} ± {fit_target['Decay'][1]:.2f} us")
        axs[1].set_xlabel("Time (us)")
        axs[1].set_ylabel("P(1)")
        axs[1].scatter(t, self.result_target, marker='o', color=dark_purple)
        axs[1].plot(t, fit_target['Amplitude'][0] * np.exp(-t / fit_target['Decay'][0]) + fit_target['Offset'][0],
                    color=dark_purple)

        # Combined plot
        axs[2].set_title("Control and Target T1 decay")
        axs[2].set_xlabel("Time (us)")
        axs[2].set_ylabel("P(1)")
        # Plot Control data with both line and scatter markers
        axs[2].scatter(t, self.result_control, marker='o', color=dark_navy, label='Control', zorder=3)
        axs[2].plot(t, self.result_control, linestyle='-', marker='_', color=dark_navy, zorder=2)

        # Plot Target data with both line and scatter markers
        axs[2].scatter(t, self.result_target, marker='o', color=dark_purple, label='Target', zorder=3)
        axs[2].plot(t, self.result_target, linestyle='-', marker='_', color=dark_purple, zorder=2)

        # axs[2].scatter(t, self.result_control, marker='o', color=dark_navy, label='Control')
        # axs[2].scatter(t, self.result_target, marker='o', color=dark_purple, label='Target')
        # axs[2].plot(t, fit_control['Amplitude'][0] * np.exp(-t / fit_control['Decay'][0]) + fit_control['Offset'][0],
        #             color=dark_navy, label=f'Control Fit T1={fit_control["Decay"][0]:.2f} us')
        # axs[2].plot(t, fit_target['Amplitude'][0] * np.exp(-t / fit_target['Decay'][0]) + fit_target['Offset'][0],
        #             color=dark_purple, label=f'Target Fit T1={fit_target["Decay"][0]:.2f} us')
        axs[2].legend()

        fig.tight_layout()
        plt.show()

    def fit_exp_decay_with_cov(self, trace, time_resolution):
        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        t = np.arange(0, len(trace) * time_resolution, time_resolution)
        try:
            popt, pcov = curve_fit(exp_decay, t, trace, maxfev=2400)
            A, tau, C = popt
            perr = np.sqrt(np.diag(pcov))
            return {'Amplitude': (A, perr[0]), 'Decay': (tau, perr[1]), 'Offset': (C, perr[2])}
        except (OptimizeWarning, RuntimeError) as e:
            print(f"Optimization warning or error: {e}")
            return {'Amplitude': (np.nan, np.nan), 'Decay': (np.nan, np.nan), 'Offset': (np.nan, np.nan)}


class StarkTwoQubitsSWAPTwoDrives(experiment):
    # Perform a Stark Shifted T1 experiment on one qubit while also measuring another
    @log_and_record
    def run(self, qubits, amp_control, amp_target, rise=0.01, start=0, stop=3, step=0.03,
            phase_diff=0, stark_offset=50, initial_lpb=None):

        self.duts = qubits
        self.stark_offset = stark_offset
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase = 0
        self.width = 0
        self.start = start
        self.stop = stop
        self.step = step
        self.fitting_2D = None
        self.phase_diff = phase_diff

        c1_control = self.duts[0].get_default_c1()  # the qubit to be stark shifted and T1 performed on
        c1_target = self.duts[1].get_default_c1()  # the qubit which will just be measured at the same time

        self.original_freq = c1_control['Xp'].freq
        self.frequency = self.original_freq + self.stark_offset

        c2 = prims.build_CZ_stark_from_parameters(control_q=self.duts[0], target_q=self.duts[1],
                                                  amp_target=self.amp_target, amp_control=self.amp_control,
                                                  frequency=self.frequency, rise=rise, width=self.width,
                                                  phase_diff=self.phase_diff,
                                                  iz_control=0,
                                                  iz_target=0,
                                                  echo=False,
                                                  trunc=1.0, zz_interaction_positive=True)

        mprim_control = self.duts[0].get_measurement_prim_intlist(0)
        mprim_target = self.duts[1].get_measurement_prim_intlist(0)

        cs_pulse = c2.get_stark_drive_pulses()
        stark_drive_target_pulse = c2['stark_drive_target']
        stark_drive_control_pulse = c2['stark_drive_control']

        lpb = c1_control['X'] + cs_pulse + mprim_control * mprim_target

        if initial_lpb:
            lpb = initial_lpb + lpb

        swpparams = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'width'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'width'),
        ]

        swp = sweeper(np.arange, n_kwargs={'start': start, 'stop': stop, 'step': step},
                      params=swpparams)

        basic(lpb, swp, 'p(1)')

        self.result_control = np.asarray(mprim_control.result(), dtype=float).flatten()
        self.result_target = np.asarray(mprim_target.result(), dtype=float).flatten()

    @register_browser_function(available_after=(run,))
    def plot_t1(self):
        args = self.retrieve_args(self.run)

        dark_navy = '#000080'
        dark_purple = '#800080'

        fit_control = self.fit_exp_decay_with_cov(self.result_control, args['step'])
        fit_target = self.fit_exp_decay_with_cov(self.result_target, args['step'])

        t = np.arange(0, args['stop'], args['step'])

        fig, axs = plt.subplots(1, 3, figsize=[25, 5])

        # Plot for result_control
        axs[0].set_title(f"Control T1 decay\nT1={fit_control['Decay'][0]:.2f} ± {fit_control['Decay'][1]:.2f} us")
        axs[0].set_xlabel("Time (us)")
        axs[0].set_ylabel("P(1)")
        axs[0].scatter(t, self.result_control, marker='o', color=dark_navy)
        axs[0].plot(t, fit_control['Amplitude'][0] * np.exp(-t / fit_control['Decay'][0]) + fit_control['Offset'][0],
                    color=dark_navy)

        # Plot for result_target
        axs[1].set_title(f"Target T1 decay\nT1={fit_target['Decay'][0]:.2f} ± {fit_target['Decay'][1]:.2f} us")
        axs[1].set_xlabel("Time (us)")
        axs[1].set_ylabel("P(1)")
        axs[1].scatter(t, self.result_target, marker='o', color=dark_purple)
        axs[1].plot(t, fit_target['Amplitude'][0] * np.exp(-t / fit_target['Decay'][0]) + fit_target['Offset'][0],
                    color=dark_purple)

        # Combined plot
        axs[2].set_title("Control and Target T1 decay")
        axs[2].set_xlabel("Time (us)")
        axs[2].set_ylabel("P(1)")
        axs[2].scatter(t, self.result_control, marker='o', color=dark_navy, label='Control')
        axs[2].scatter(t, self.result_target, marker='o', color=dark_purple, label='Target')
        axs[2].plot(t, fit_control['Amplitude'][0] * np.exp(-t / fit_control['Decay'][0]) + fit_control['Offset'][0],
                    color=dark_navy, label=f'Control Fit T1={fit_control["Decay"][0]:.2f} us')
        axs[2].plot(t, fit_target['Amplitude'][0] * np.exp(-t / fit_target['Decay'][0]) + fit_target['Offset'][0],
                    color=dark_purple, label=f'Target Fit T1={fit_target["Decay"][0]:.2f} us')
        axs[2].legend()

        fig.tight_layout()
        plt.show()

    def fit_exp_decay_with_cov(self, trace, time_resolution):
        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        t = np.arange(0, len(trace) * time_resolution, time_resolution)
        try:
            popt, pcov = curve_fit(exp_decay, t, trace, maxfev=2400)
            A, tau, C = popt
            perr = np.sqrt(np.diag(pcov))
            return {'Amplitude': (A, perr[0]), 'Decay': (tau, perr[1]), 'Offset': (C, perr[2])}
        except (OptimizeWarning, RuntimeError) as e:
            print(f"Optimization warning or error: {e}")
            return {'Amplitude': (np.nan, np.nan), 'Decay': (np.nan, np.nan), 'Offset': (np.nan, np.nan)}


class StarkRamseyMultilevel(Experiment):
    """
    Represents a simple Ramsey experiment with multilevel frequency sweeps.
    This version has changed the step size from 0.001 to 0.005.
    """

    @log_and_record
    def run(self,
            qubit: Any,  # Replace 'Any' with the actual type of qubit
            collection_name: str = 'f01',
            mprim_index: int = 0,
            # Replace 'Any' with the actual type
            initial_lpb: Optional[Any] = None,
            start: float = 0.0,
            stop: float = 1.0,
            step: float = 0.005,
            set_offset: float = 10.0,
            stark_offset=50,
            # frequency=4835,
            amp=0.3, width=0.1, rise=0.01, trunc=1.2,
            update: bool = False) -> None:
        """
        Run Stark Ramsey experiment.

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
        self.set_offset = set_offset
        self.step = step
        self.stop = stop
        self.stark_offset = stark_offset

        # Define the levels for the sweep based on the collection name
        start_level = int(collection_name[1])
        end_level = int(collection_name[2])
        self.level_diff = end_level - start_level

        c1q = qubit.get_c1(collection_name)  # Retrieve the gate collection object
        # Save original frequency
        original_freq = c1q['Xp'].freq
        self.original_freq = original_freq

        self.frequency = self.original_freq + self.stark_offset

        # self.frequency = frequency

        cs_pulse = c1q['X'].clone()
        cs_pulse.update_pulse_args(amp=amp, freq=self.frequency, phase=0., shape='soft_square', width=self.stop,
                                   rise=rise, trunc=trunc)

        # Update the frequency with the calculated offset
        c1q.update_parameters(
            freq=original_freq +
                 set_offset /
                 self.level_diff)

        # Get the measurement primitive
        mprim = qubit.get_measurement_prim_intlist(mprim_index)
        self.mp = mprim

        # Construct the logic primitive block
        lpb = c1q['Xp'] + cs_pulse + c1q['Xm'] + mprim

        if initial_lpb:
            lpb = initial_lpb + lpb

        swpparams = [
            sparam.func(cs_pulse.update_pulse_args, {}, 'width'),
        ]

        swp = sweeper(np.arange, n_kwargs={'start': 0.0, 'stop': self.stop, 'step': self.step},
                      params=swpparams)

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
                      # Replace 'Any' with the actual type
                      initial_lpb: Optional[Any] = None,
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

        if isinstance(virtual_transmon.t1, list):
            t1 = virtual_transmon.t1[0]
        else:
            t1 = virtual_transmon.t1

        decay_rate = 1 / t1

        # Ramsey fringes formula

        f_o_actual = f_q - (f_d + f_o)

        ramsey_fringes = (1 + np.cos(2 * np.pi * f_o_actual * t)
                          * np.exp(-decay_rate * t)) / 2

        self.data = ramsey_fringes

        quiescent_state_distribution = virtual_transmon.quiescent_state_distribution
        standard_deviation = np.sum(quiescent_state_distribution[1:])

        random_noise_factor = 1 + np.random.normal(
            0, standard_deviation, self.data.shape)

        self.data = np.clip(self.data * quiescent_state_distribution[0] * random_noise_factor, 0, 1)

        # If sampling noise is enabled, simulate the noise
        if setup().status().get_param('Sampling_Noise'):
            # Get the number of shot used in the simulation
            shot_number = setup().status().get_param('Shot_Number')

            # generate binomial distribution of the result to simulate the
            # sampling noise
            self.data = np.random.binomial(
                shot_number, self.data) / shot_number

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
        fig.add_trace(
            go.Scatter(
                x=t,
                y=data,
                mode='lines+markers',
                name='data'))
        fig.update_layout(
            title=f"Ramsey {args['qubit'].hrid} transition {args['collection_name']}",
            xaxis_title="Time (us)",
            yaxis_title="<z>",
            legend_title="Legend",
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="Black"),
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
            self.fit_params = fit_1d_freq_exp_with_cov(
                self.data, dt=args['step'])
            fitted_freq_offset = (self.fit_params['Frequency'].n - self.set_offset) / self.level_diff
            self.fitted_freq_offset = fitted_freq_offset
            self.frequency_guess = self.original_freq - fitted_freq_offset
            self.error_bar = self.fit_params['Frequency'].s

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
        args['drive_freq'] = args['qubit'].get_c1(
            args['collection_name'])['X'].freq
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
        time_points_interpolate = np.arange(
            args['start'], args['stop'], args['step'] / 10)

        # Create a plot using Plotly
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=self.data,
                mode='markers',
                name='Data'),
            row=1,
            col=1)

        if hasattr(self, 'fit_params'):

            # Extract fitting parameters
            frequency = self.fit_params['Frequency'].n
            amplitude = self.fit_params['Amplitude'].n
            phase = self.fit_params['Phase'].n - \
                    2.0 * np.pi * frequency * args['start']
            offset = self.fit_params['Offset'].n
            decay = self.fit_params['Decay'].n

            # Generate the fitted curve
            fitted_curve = amplitude * np.exp(-time_points_interpolate / decay) * \
                           np.sin(2.0 * np.pi * frequency * time_points_interpolate + phase) + offset

            fig.add_trace(
                go.Scatter(
                    x=time_points_interpolate,
                    y=fitted_curve,
                    mode='lines',
                    name='Fit'),
                row=1,
                col=1)

            # Set plot layout details
            title_text = f"Ramsey decay {args['qubit'].hrid} transition {args['collection_name']}: <br>" \
                         f"{decay} ± {self.fit_params['Decay'].n} us"
            fig.update_layout(
                title_text=title_text,
                xaxis_title=f"Time (us) <br> Frequency: {frequency} ± {self.fit_params['Frequency'].n}",
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

    def plot_fft(self, plot_range: Tuple[float, float] = (
            0.05, 1)) -> go.Figure:
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
        fig.add_trace(
            go.Scatter(
                x=frequencies[mask],
                y=fft_magnitudes[mask],
                mode='lines'))

        # Set plot layout details
        fig.update_layout(title='Ramsey Spectrum',
                          xaxis_title='Frequency [MHz]',
                          yaxis_title='Strength',
                          plot_bgcolor="white")
        return fig

    def get_analyzed_result_prompt(self) -> str:
        """
        Get a prompt for the analyzed result of the Ramsey experiment.

        Returns:
            A string containing the prompt for the analyzed result.
        """

        self.analyze_data()

        if self.error_bar == np.inf:
            return "The Ramsey experiment failed to fit the data."

        return (f"The Ramsey experiment for qubit {self.retrieve_args(self.run)['qubit'].hrid} has been analyzed. " \
                f"The expected offset was set to {self.set_offset:.3f} MHz, and the measured offset is "
                f"{self.fitted_freq_offset:.3f}+- {self.error_bar:.3f} MHz.")


class StarkDriveRamseyTwoQubits(experiment):
    # performs a ramsey experiment on two qubits while applying a stark shift drive to one of them. Extension of StarkDriveRamsey()
    @log_and_record
    def run(self, qubits,
            collection_name: str = 'f01',
            mprim_index: int = 0,
            # Replace 'Any' with the actual type
            initial_lpb: Optional[Any] = None,
            start: float = 0.0,
            stop: float = 1.0,
            step: float = 0.005,
            set_offset: float = 10.0,
            stark_offset=50, amp=0.1, width=0.1, rise=0.01, trunc=1.2,
            update: bool = False) -> None:

        assert len(qubits) == 2

        self.set_offset = set_offset
        self.step = step
        self.stop = stop
        self.stark_offset = stark_offset

        c1s = [qubit.get_c1(collection_name) for qubit in qubits]  # get qubits. Ramsey on both, stark shift first qubit
        mps = [qubit.get_measurement_prim_intlist(mprim_index) for qubit in qubits]

        start_level = int(collection_name[1])
        end_level = int(collection_name[2])
        self.level_diff = end_level - start_level

        self.original_freqs = [c1['Xp'].freq for c1 in c1s]
        for c1 in c1s: c1.update_parameters(freq=c1['Xp'].freq + self.set_offset / self.level_diff)

        self.original_freq = c1s[0]['Xp'].freq
        self.frequency = self.original_freq + self.stark_offset

        cs_pulse = c1s[0]['Xp'].clone()
        cs_pulse.update_pulse_args(amp=amp, freq=self.frequency, phase=0., shape='blackman_square', width=self.stop,
                                   rise=rise, trunc=trunc)

        lpb = prims.ParallelLPB([c1['Xp'] for c1 in c1s]) + cs_pulse + prims.ParallelLPB(
            [c1['Xm'] for c1 in c1s]) + prims.ParallelLPB(
            [mp for mp in mps])  # stark shift ramsey on first qubit, normal ramsey on second

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        swpparams = [
            sparam.func(cs_pulse.update_pulse_args, {}, 'width'),
        ]

        swp = sweeper(np.arange, n_kwargs={'start': 0.0, 'stop': self.stop, 'step': self.step},
                      params=swpparams)

        # Execute the basic experiment routine
        basic(lpb, swp, '<z>')

        self.result = [np.squeeze(mp.result()) for mp in mps]
        self.traces = self.result

        for i, c1 in enumerate(c1s): c1.update_parameters(freq=self.original_freqs[i])

    def live_plots(self, step_no: Optional[Tuple[int]] = None) -> go.Figure:
        args = self.retrieve_args(self.run)
        data0 = np.squeeze(self.traces[0])
        data1 = np.squeeze(self.traces[1])
        t = np.arange(args['start'], args['stop'], args['step'])

        if step_no is not None:
            t = t[:step_no[0]]
            data0 = data0[:step_no[0]]
            data1 = data1[:step_no[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=data0, mode='lines+markers', name='trace0'))
        fig.add_trace(go.Scatter(x=t, y=data1, mode='lines+markers', name='trace1'))
        fig.update_layout(
            title=f"Ramsey {args['qubits'][0].hrid} and {args['qubits'][1].hrid} transition {args['collection_name']}",
            xaxis_title="Time (us)",
            yaxis_title="<z>",
            legend_title="Legend",
            font=dict(family="Courier New, monospace", size=12, color="Black"),
            plot_bgcolor="white"
        )
        return fig

    def analyze_data(self) -> None:
        args = self.retrieve_args(self.run)
        from leeq.theory.fits import fit_1d_freq_exp_with_cov

        self.frequency_shift = [[], []]
        self.frequency_guess = [None, None]
        self.error_bar = [None, None]
        self.fitted_freq_offset = [None, None]
        self.fit_params = [None, None]  # Update to store fit_params for each trace

        for i in range(2):
            try:
                self.fit_params[i] = fit_1d_freq_exp_with_cov(self.traces[i], dt=args['step'])
                fitted_freq_offset = (self.fit_params[i]['Frequency'].n - self.set_offset) / self.level_diff
                self.fitted_freq_offset[i] = fitted_freq_offset
                self.frequency_guess[i] = self.original_freqs[i] - fitted_freq_offset
                self.error_bar[i] = self.fit_params[i]['Frequency'].s
                self.frequency_shift[i].append(self.frequency_guess[i] - self.original_freqs[i])
            except Exception as e:
                self.frequency_guess[i] = 0
                self.error_bar[i] = np.inf

    def dump_results_and_configuration(self) -> Tuple[
        float, float, Any, Dict[str, Union[float, str]], datetime.datetime]:
        args = copy.copy(self.retrieve_args(self.run))
        del args['initial_lpb']
        args['drive_freq'] = args['qubits'][0].get_c1(args['collection_name'])['X'].freq
        args['qubits'] = [qubit.hrid for qubit in args['qubits']]
        return self.frequency_guess, self.error_bar, self.traces, args, datetime.datetime.now()

    @register_browser_function(available_after=('run',))
    def plot(self) -> go.Figure:
        self.analyze_data()
        args = self.retrieve_args(self.run)

        time_points = np.arange(args['start'], args['stop'], args['step'])
        time_points_interpolate = np.arange(args['start'], args['stop'], args['step'] / 10)

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=[f"{args['qubits'][i].hrid} f = {self.frequency_guess[i]} MHz" for i in
                                            range(2)])

        for i in range(2):
            fig.add_trace(go.Scatter(x=time_points, y=self.traces[i], mode='markers', name=f'Trace {i}'), row=i + 1,
                          col=1)
            if self.fit_params[i]:
                frequency = self.fit_params[i]['Frequency'].n
                amplitude = self.fit_params[i]['Amplitude'].n
                phase = self.fit_params[i]['Phase'].n - 2.0 * np.pi * frequency * args['start']
                offset = self.fit_params[i]['Offset'].n
                decay = self.fit_params[i]['Decay'].n

                fitted_curve = amplitude * np.exp(-time_points_interpolate / decay) * \
                               np.sin(2.0 * np.pi * frequency * time_points_interpolate + phase) + offset

                fig.add_trace(go.Scatter(x=time_points_interpolate, y=fitted_curve, mode='lines', name=f'Fit {i}'),
                              row=i + 1, col=1)
                if i == 0:
                    fig.update_yaxes(title_text="<z>", row=i + 1, col=1)
                if i == 1:
                    fig.update_xaxes(title_text="Time (us)", row=i + 1, col=1)

        fig.update_layout(
            title_text=f"Stark drive Ramsey decay {args['qubits'][0].hrid} and {args['qubits'][1].hrid} transition {args['collection_name']}",
            plot_bgcolor="white",
            width=2000,  # Set your desired width here
            height=800  # Set your desired height here
        )
        return fig

    def plot_fft(self, plot_range: Tuple[float, float] = (0.05, 1)) -> go.Figure:
        self.analyze_data()
        args = self.retrieve_args(self.run)
        time_step = args['step']

        fig = go.Figure()
        for i in range(2):
            fft_magnitudes = np.abs(np.fft.rfft(self.traces[i]))
            frequencies = np.fft.rfftfreq(len(self.traces[i]), time_step)
            mask = (frequencies > plot_range[0]) & (frequencies < plot_range[1])
            fig.add_trace(go.Scatter(x=frequencies[mask], y=fft_magnitudes[mask], mode='lines', name=f'Trace {i}'))

        fig.update_layout(
            title='Ramsey Spectrum',
            xaxis_title='Frequency [MHz]',
            yaxis_title='Strength',
            plot_bgcolor="white"
        )
        return fig

    def get_analyzed_result_prompt(self) -> str:
        self.analyze_data()

        if all(err == np.inf for err in self.error_bar):
            return "The Ramsey experiment failed to fit the data for both traces."

        result_str = ""
        for i in range(2):
            if self.error_bar[i] != np.inf:
                result_str += (
                    f"The Ramsey experiment for trace {i} of qubit {self.retrieve_args(self.run)['qubits'].hrid} "
                    f"has been analyzed. The expected offset was set to {self.set_offset:.3f} MHz, "
                    f"and the measured offset is {self.fitted_freq_offset[i]:.3f}±{self.error_bar[i]:.3f} MHz.\n")
            else:
                result_str += f"The Ramsey experiment failed to fit the data for trace {i}.\n"

        return result_str


class StarkDriveRamseyMultiQubits(experiment):
    # performs a ramsey experiment on two qubits while applying a stark shift drive to one of them. Extension of StarkDriveRamsey()
    @log_and_record
    def run(self, qubits,
            collection_name: str = 'f01',
            mprim_index: int = 0,
            # Replace 'Any' with the actual type
            initial_lpb: Optional[Any] = None,
            start: float = 0.0,
            stop: float = 1.0,
            step: float = 0.005,
            set_offset: float = 10.0,
            stark_offset=50, amp=0.1, width=0.1, rise=0.01, trunc=1.2,
            update: bool = False) -> None:

        assert len(qubits) == 2

        self.set_offset = set_offset
        self.step = step
        self.stop = stop
        self.stark_offset = stark_offset

        c1s = [qubit.get_c1(collection_name) for qubit in qubits]  # get qubits. Ramsey on both, stark shift first qubit
        mps = [qubit.get_measurement_prim_intlist(mprim_index) for qubit in qubits]

        start_level = int(collection_name[1])
        end_level = int(collection_name[2])
        self.level_diff = end_level - start_level

        self.original_freqs = [c1['Xp'].freq for c1 in c1s]
        for c1 in c1s: c1.update_parameters(freq=c1['Xp'].freq + self.set_offset / self.level_diff)

        self.original_freq = c1s[0]['Xp'].freq
        self.frequency = self.original_freq + self.stark_offset

        cs_pulse = c1s[0]['Xp'].clone()
        cs_pulse.update_pulse_args(amp=amp, freq=self.frequency, phase=0., shape='soft_square', width=self.stop,
                                   rise=rise, trunc=trunc)

        lpb = prims.ParallelLPB([c1['Xp'] for c1 in c1s]) + cs_pulse + prims.ParallelLPB(
            [c1['Xm'] for c1 in c1s]) + prims.ParallelLPB(
            [mp for mp in mps])  # stark shift ramsey on first qubit, normal ramsey on second

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        swpparams = [
            sparam.func(cs_pulse.update_pulse_args, {}, 'width'),
        ]

        swp = sweeper(np.arange, n_kwargs={'start': 0.0, 'stop': self.stop, 'step': self.step},
                      params=swpparams)

        # Execute the basic experiment routine
        basic(lpb, swp, '<z>')

        self.result = [np.squeeze(mp.result()) for mp in mps]
        self.traces = self.result

        for i, c1 in enumerate(c1s): c1.update_parameters(freq=self.original_freqs[i])

    def live_plots(self, step_no: Optional[Tuple[int]] = None) -> go.Figure:
        args = self.retrieve_args(self.run)
        data0 = np.squeeze(self.traces[0])
        data1 = np.squeeze(self.traces[1])
        t = np.arange(args['start'], args['stop'], args['step'])

        if step_no is not None:
            t = t[:step_no[0]]
            data0 = data0[:step_no[0]]
            data1 = data1[:step_no[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=data0, mode='lines+markers', name='trace0'))
        fig.add_trace(go.Scatter(x=t, y=data1, mode='lines+markers', name='trace1'))
        fig.update_layout(
            title=f"Ramsey {args['qubits'][0].hrid} and {args['qubits'][1].hrid} transition {args['collection_name']}",
            xaxis_title="Time (us)",
            yaxis_title="<z>",
            legend_title="Legend",
            font=dict(family="Courier New, monospace", size=12, color="Black"),
            plot_bgcolor="white"
        )
        return fig

    def analyze_data(self) -> None:
        args = self.retrieve_args(self.run)
        from leeq.theory.fits import fit_1d_freq_exp_with_cov

        self.frequency_shift = [[], []]
        self.frequency_guess = [None, None]
        self.error_bar = [None, None]
        self.fitted_freq_offset = [None, None]
        self.fit_params = [None, None]  # Update to store fit_params for each trace

        for i in range(2):
            try:
                self.fit_params[i] = fit_1d_freq_exp_with_cov(self.traces[i], dt=args['step'])
                fitted_freq_offset = (self.fit_params[i]['Frequency'].n - self.set_offset) / self.level_diff
                self.fitted_freq_offset[i] = fitted_freq_offset
                self.frequency_guess[i] = self.original_freqs[i] - fitted_freq_offset
                self.error_bar[i] = self.fit_params[i]['Frequency'].s
                self.frequency_shift[i].append(self.frequency_guess[i] - self.original_freqs[i])
            except Exception as e:
                self.frequency_guess[i] = 0
                self.error_bar[i] = np.inf

    def dump_results_and_configuration(self) -> Tuple[
        float, float, Any, Dict[str, Union[float, str]], datetime.datetime]:
        args = copy.copy(self.retrieve_args(self.run))
        del args['initial_lpb']
        args['drive_freq'] = args['qubits'][0].get_c1(args['collection_name'])['X'].freq
        args['qubits'] = [qubit.hrid for qubit in args['qubits']]
        return self.frequency_guess, self.error_bar, self.traces, args, datetime.datetime.now()

    @register_browser_function(available_after=('run',))
    def plot(self) -> go.Figure:
        self.analyze_data()
        args = self.retrieve_args(self.run)

        time_points = np.arange(args['start'], args['stop'], args['step'])
        time_points_interpolate = np.arange(args['start'], args['stop'], args['step'] / 10)

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=[f"{args['qubits'][i].hrid} f = {self.frequency_guess[i]} MHz" for i in
                                            range(2)])

        for i in range(2):
            fig.add_trace(go.Scatter(x=time_points, y=self.traces[i], mode='markers', name=f'Trace {i}'), row=i + 1,
                          col=1)
            if self.fit_params[i]:
                frequency = self.fit_params[i]['Frequency'].n
                amplitude = self.fit_params[i]['Amplitude'].n
                phase = self.fit_params[i]['Phase'].n - 2.0 * np.pi * frequency * args['start']
                offset = self.fit_params[i]['Offset'].n
                decay = self.fit_params[i]['Decay'].n

                fitted_curve = amplitude * np.exp(-time_points_interpolate / decay) * \
                               np.sin(2.0 * np.pi * frequency * time_points_interpolate + phase) + offset

                fig.add_trace(go.Scatter(x=time_points_interpolate, y=fitted_curve, mode='lines', name=f'Fit {i}'),
                              row=i + 1, col=1)
                if i == 0:
                    fig.update_yaxes(title_text="<z>", row=i + 1, col=1)
                if i == 1:
                    fig.update_xaxes(title_text="Time (us)", row=i + 1, col=1)

        fig.update_layout(
            title_text=f"Stark drive Ramsey decay {args['qubits'][0].hrid} and {args['qubits'][1].hrid} transition {args['collection_name']}",
            plot_bgcolor="white",
            width=2000,  # Set your desired width here
            height=800  # Set your desired height here
        )
        return fig

    def plot_fft(self, plot_range: Tuple[float, float] = (0.05, 1)) -> go.Figure:
        self.analyze_data()
        args = self.retrieve_args(self.run)
        time_step = args['step']

        fig = go.Figure()
        for i in range(2):
            fft_magnitudes = np.abs(np.fft.rfft(self.traces[i]))
            frequencies = np.fft.rfftfreq(len(self.traces[i]), time_step)
            mask = (frequencies > plot_range[0]) & (frequencies < plot_range[1])
            fig.add_trace(go.Scatter(x=frequencies[mask], y=fft_magnitudes[mask], mode='lines', name=f'Trace {i}'))

        fig.update_layout(
            title='Ramsey Spectrum',
            xaxis_title='Frequency [MHz]',
            yaxis_title='Strength',
            plot_bgcolor="white"
        )
        return fig

    def get_analyzed_result_prompt(self) -> str:
        self.analyze_data()

        if all(err == np.inf for err in self.error_bar):
            return "The Ramsey experiment failed to fit the data for both traces."

        result_str = ""
        for i in range(2):
            if self.error_bar[i] != np.inf:
                result_str += (
                    f"The Ramsey experiment for trace {i} of qubit {self.retrieve_args(self.run)['qubits'].hrid} "
                    f"has been analyzed. The expected offset was set to {self.set_offset:.3f} MHz, "
                    f"and the measured offset is {self.fitted_freq_offset[i]:.3f}±{self.error_bar[i]:.3f} MHz.\n")
            else:
                result_str += f"The Ramsey experiment failed to fit the data for trace {i}.\n"

        return result_str


class StarkZZShiftTwoQubitMultilevel(Experiment):
    """Class to compute ZZ Shift for Two Qubit Multilevel system."""

    @log_and_record
    def run(self,
            duts: List[TransmonElement],
            collection_name: str = 'f01',
            mprim_index: int = 0,
            start: float = 0.0,
            stop: float = 1,
            step: float = 0.005,
            set_offset: int = 10,
            stark_offset=50,
            disable_sub_plot: bool = False) -> None:
        """Run the ZZ Shift experiment.

        Parameters:
            duts: The DUTs (Device Under Test).
            collection_name: The name of the frequency collection (e.g., 'f01').
            mprim_index: The index of the measurement primitive.
            start: The start frequency for the sweep.
            stop: The stop frequency for the sweep.
            step: The step size for the frequency sweep.
            set_offset: The frequency offset.
            disable_sub_plot: Whether to disable subplots.
            hardware_stall: Whether to use hardware stall.

        Returns:
            None
        """
        # Ensure there are exactly 2 DUTs (Device Under Test)
        assert len(duts) == 2

        plot_result_in_jupyter = setup().status().get_param("Plot_Result_In_Jupyter")

        if disable_sub_plot:
            setup().status().set_param("Plot_Result_In_Jupyter", False)

        c1q1 = duts[0].get_c1(collection_name)
        c1q2 = duts[1].get_c1(collection_name)

        # Q1 ramsey Q2 steady
        self.q1_ramsey_q2_ground = StarkRamseyMultilevel(
            duts[0],
            collection_name=collection_name,
            mprim_index=mprim_index,
            initial_lpb=None,
            start=start,
            stop=stop,
            step=step,
            set_offset=set_offset,
            stark_offset=stark_offset,
            update=False)

        self.q1_ramsey_q2_excited = StarkRamseyMultilevel(
            duts[0],
            collection_name=collection_name,
            mprim_index=mprim_index,
            initial_lpb=c1q2['X'],
            start=start,
            stop=stop,
            step=step,
            set_offset=set_offset,
            stark_offset=stark_offset,
            update=False)

        # Q2 ramsey Q1 steady
        self.q2_ramsey_q1_ground = StarkRamseyMultilevel(
            duts[1],
            collection_name=collection_name,
            mprim_index=mprim_index,
            initial_lpb=None,
            start=start,
            stop=stop,
            step=step,
            set_offset=set_offset,
            stark_offset=stark_offset,
            update=False)

        self.q2_ramsey_q1_excited = StarkRamseyMultilevel(
            duts[1],
            collection_name=collection_name,
            mprim_index=mprim_index,
            initial_lpb=c1q1['X'],
            start=start,
            stop=stop,
            step=step,
            set_offset=set_offset,
            stark_offset=stark_offset,
            update=False)

        self.zz = [
            self.q1_ramsey_q2_excited.frequency_guess -
            self.q1_ramsey_q2_ground.frequency_guess,
            self.q2_ramsey_q1_excited.frequency_guess -
            self.q2_ramsey_q1_ground.frequency_guess,
        ]

        self.zz_error = [
            self.q1_ramsey_q2_excited.error_bar -
            self.q1_ramsey_q2_ground.error_bar,
            self.q2_ramsey_q1_excited.error_bar -
            self.q2_ramsey_q1_ground.error_bar,
        ]

        setup().status().set_param("Plot_Result_In_Jupyter", plot_result_in_jupyter)


class StarkRepeatedGateRabi(Experiment):

    @log_and_record
    def run(self, dut, amp, frequency, phase=0, rise=0.01, trunc=1.0, width=0, start_gate_number=0, gate_count=40,
            initial_lpb=None):
        """
        Sweep time and find the initial guess of amplitude

        :return:
        """
        self.dut = dut
        self.frequency = frequency
        self.amp = amp
        self.phase = phase
        self.width = width
        self.rise = rise
        self.trunc = trunc
        self.start_gate_number = start_gate_number
        self.gate_count = gate_count

        pulse_count = np.arange(start_gate_number, start_gate_number + gate_count, 1)

        c1 = self.dut.get_default_c1()

        pulse = c1['X'].clone()
        pulse.update_pulse_args(
            amp=self.amp, freq=self.frequency, phase=self.phase, shape='blackman_square', width=self.width,
            rise=self.rise, trunc=self.trunc)

        lpb = pulse

        sequence_lpb = []
        mprim = self.dut.get_measurement_prim_intlist(0)

        for n in pulse_count:
            sequence = LogicalPrimitiveBlockSerial([pulse] * (n) + [mprim])
            sequence_lpb.append(sequence)

        lpb = LogicalPrimitiveBlockSweep(sequence_lpb)
        swp = sweeper.from_sweep_lpb(lpb)

        if initial_lpb is not None:
            lpb = initial_lpb + pulse

        basic(lpb, swp=swp, basis="<z>")

        self.result = np.squeeze(mprim.result())

    @register_browser_function(available_after=(run,))
    def plot(self):
        """
        Plot the results.
        """
        args = self.retrieve_args(self.run)

        t = np.arange(args['start_gate_number'], args['start_gate_number'] + args['gate_count'], 1)

        data = self.result.squeeze()

        plt.scatter(t, data)
        plt.xlabel('Number of pulses')
        plt.ylabel('<z>')

        return plt.gcf()


class StarkContinuesRabi(Experiment):

    @log_and_record
    def run(self, dut, amp, frequency, phase=0, rise=0.01, trunc=1.0, width_start=0, width_stop=4, width_step=0.01,
            initial_lpb=None):
        """
        Sweep time and find the initial guess of amplitude

        :return:
        """
        self.dut = dut
        self.frequency = frequency
        self.amp = amp
        self.phase = phase
        self.rise = rise
        self.trunc = trunc
        self.width_start = width_start
        self.width_stop = width_stop
        self.width_step = width_step

        c1 = self.dut.get_default_c1()

        pulse = c1['X'].clone()
        pulse.update_pulse_args(
            amp=self.amp, freq=self.frequency, phase=self.phase, shape='blackman_square', width=0,
            rise=self.rise, trunc=self.trunc)

        mprim = self.dut.get_measurement_prim_intlist(0)

        # Set up sweep parameters
        swpparams = [SweepParametersSideEffectFactory.func(
            pulse.update_pulse_args, {}, 'width'
        )]
        swp = Sweeper(
            np.arange,
            n_kwargs={'start': width_start, 'stop': width_stop, 'step': width_step},
            params=swpparams
        )

        if initial_lpb is not None:
            lpb = initial_lpb + pulse

        basic(pulse + mprim, swp=swp, basis="<z>")

        self.result = np.squeeze(mprim.result())

    @register_browser_function(available_after=(run,))
    def plot(self):
        """
        Plot the results.
        """
        args = self.retrieve_args(self.run)

        t = np.arange(args['width_start'], args['width_stop'], args['width_step'])

        data = self.result.squeeze()

        plt.scatter(t, data)
        plt.xlabel('Width [us]')
        plt.ylabel('<z>')

        return plt.gcf()


class StarkRepeatedGateDRAGLeakageCalibration(Experiment):

    @log_and_record
    def run(self, dut, amp, frequency, phase=0, rise=0.01, trunc=1.0, width=0.1, gate_count=40, initial_lpb=None,
            inv_alpha_start=None, inv_alpha_stop=None, sweep_count=20):
        """
        Sweep time and find the initial guess of amplitude

        :return:
        """
        self.dut = dut
        self.frequency = frequency
        self.amp = amp
        self.phase = phase
        self.width = width
        self.rise = rise
        self.trunc = trunc
        self.gate_count = gate_count

        c1 = self.dut.get_default_c1()

        pulse = c1['X'].clone()

        alpha = pulse.alpha
        print('alpha',alpha)

        if inv_alpha_start is None:
            inv_alpha_start = 1 / alpha - 0.006
        if inv_alpha_stop is None:
            inv_alpha_stop = 1 / alpha + 0.006

        def update_alpha(n):
            return pulse.update_parameters(alpha=1 / n)

        # Create a sweeper for the alpha parameter.
        self.sweep_values = np.linspace(inv_alpha_start, inv_alpha_stop, num=sweep_count)
        swp = Sweeper(
            self.sweep_values,
            params=[
                SweepParametersSideEffectFactory.func(
                    update_alpha,
                    argument_name='n',
                    kwargs={})])

        pulse.update_pulse_args(
            amp=self.amp, freq=self.frequency, phase=self.phase, shape='blackman_square', width=self.width,
            rise=self.rise, trunc=self.trunc)

        lpb = pulse

        mprim = self.dut.get_measurement_prim_intlist(0)

        sequence = LogicalPrimitiveBlockSerial([pulse] * (gate_count) + [mprim])

        if initial_lpb is not None:
            lpb = initial_lpb + sequence

        basic(lpb+mprim, swp=swp, basis="<z>")

        self.result = np.squeeze(mprim.result())

    @register_browser_function(available_after=(run,))
    def plot(self):
        """
        Plot the results.
        """
        args = self.retrieve_args(self.run)

        inv_alpha = self.sweep_values

        data = self.result.squeeze()

        plt.scatter(inv_alpha, data)
        plt.xlabel('Number of pulses')
        plt.ylabel('<z>')

        return plt.gcf()
