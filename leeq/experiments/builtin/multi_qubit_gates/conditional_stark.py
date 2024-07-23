# Conditional AC stark shift induced CZ gate

from datetime import datetime
from plotly.subplots import make_subplots
from labchronicle import register_browser_function, log_and_record

import leeq.theory.fits
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.utils import setup_logging
from leeq import Experiment, SweepParametersSideEffectFactory, Sweeper
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
from leeq.utils.compatibility import *
from leeq.theory import fits
from plotly import graph_objects as go
from leeq.core.primitives.built_in.sizzel_gate import *
from leeq.theory.fits.fit_exp import *
from matplotlib import pyplot as plt

from leeq.theory.fits.fit_exp import fit_2d_freq, fit_2d_freq_with_cov
import uncertainties as unc

from leeq.theory.estimator.kalman import KalmanFilter1D

logger = setup_logging(__name__)

from typing import Any, Optional, Tuple, Dict, Union
import numpy as np
import datetime
import copy

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import math
from labchronicle import log_and_record, register_browser_function
from leeq import Experiment, Sweeper, basic_run
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.utils.compatibility import *

import matplotlib.pyplot as plt
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial, LogicalPrimitiveBlockParallel, \
    LogicalPrimitiveBlock

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Optional, Any, Tuple, Union
from leeq.utils.compatibility import prims

from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.mplot3d.art3d as art3d

from leeq.theory.fits import *

from scipy.optimize import curve_fit, OptimizeWarning

from qutip import Bloch


from typing import List, Any


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
        cs_pulse.update_pulse_args(amp=amp, freq= self.frequency, phase=0., shape='soft_square', width=self.stop, rise=rise, trunc=trunc)

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
            fit_params = self.fit_exp_decay_with_cov(trace, args['step'])  # Assuming fit_exp_decay_with_cov is a method of the class

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
            stark_offset=50, initial_lpb = None,
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
        c1_target = self.duts[1].get_default_c1()   # the qubit which will just be measured at the same time

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
            phase_diff = 0, stark_offset=50, initial_lpb = None):

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
        c1_target = self.duts[1].get_default_c1()   # the qubit which will just be measured at the same time

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
            stark_offset=50, amp=0.1, width=0.1, rise=0.01, trunc=1.2,
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
            fitted_freq_offset = (
                                         self.fit_params['Frequency'][0] - self.set_offset) / self.level_diff
            self.fitted_freq_offset = fitted_freq_offset
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
            frequency = self.fit_params['Frequency'][0]
            amplitude = self.fit_params['Amplitude'][0]
            phase = self.fit_params['Phase'][0] - \
                    2.0 * np.pi * frequency * args['start']
            offset = self.fit_params['Offset'][0]
            decay = self.fit_params['Decay'][0]

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
                         f"{decay} ± {self.fit_params['Decay'][1]} us"
            fig.update_layout(
                title_text=title_text,
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
            stark_offset = 50, amp = 0.1, width = 0.1, rise = 0.01, trunc = 1.2,
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


        lpb = prims.ParallelLPB([c1['Xp'] for c1 in c1s]) + cs_pulse + prims.ParallelLPB([c1['Xm'] for c1 in c1s]) + prims.ParallelLPB([mp for mp in mps]) # stark shift ramsey on first qubit, normal ramsey on second

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
                fitted_freq_offset = (self.fit_params[i]['Frequency'][0] - self.set_offset) / self.level_diff
                self.fitted_freq_offset[i] = fitted_freq_offset
                self.frequency_guess[i] = self.original_freqs[i] - fitted_freq_offset
                self.error_bar[i] = self.fit_params[i]['Frequency'][1]
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

        fig = make_subplots(rows=2, cols=1, subplot_titles=[f"{args['qubits'][i].hrid} f = {self.frequency_guess[i]} MHz" for i in range(2)])

        for i in range(2):
            fig.add_trace(go.Scatter(x=time_points, y=self.traces[i], mode='markers', name=f'Trace {i}'), row=i + 1,
                          col=1)
            if self.fit_params[i]:
                frequency = self.fit_params[i]['Frequency'][0]
                amplitude = self.fit_params[i]['Amplitude'][0]
                phase = self.fit_params[i]['Phase'][0] - 2.0 * np.pi * frequency * args['start']
                offset = self.fit_params[i]['Offset'][0]
                decay = self.fit_params[i]['Decay'][0]

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




    # def analyze_data(self):
    #     self.fit_params = []
    #     self.frequency_guess = []
    #     self.error_bar = []
    #     for i,trace in enumerate(self.traces):
    #         try:
    #             fit_params = fits.Fit_1D_Freq_exp_with_cov(trace, dt=self.step)
    #
    #             fitted_freq_offset = (fit_params['Frequency'][0] - self.set_offset) / self.level_diff
    #
    #             frequency_guess = self.original_freqs[i] - fitted_freq_offset
    #             error_bar = fit_params['Frequency'][1]
    #
    #         except Exception as e:
    #             fit_params = 0
    #             frequency_guess = 0
    #             error_bar = np.inf
    #             # warnings.warn(repr(e))
    #
    #         self.fit_params.append(fit_params)
    #         self.frequency_guess.append(frequency_guess)
    #         self.error_bar.append(error_bar)
    #     # self.fit_params = fits.Fit_1D_Freq_Gaussian_Estimation(self.data, dt=step)[0]
    #
    #
    # def dump_results_and_configuration(self):
    #     args = copy.copy(self.retrieve_args(self.run))
    #     del args['initial_lpb']
    #     args['drive_freq'] = args['qubit'].get_c1(args['collection_name'])['X'].freq
    #     args['qubit'] = args['qubit'].hrid
    #     return self.frequency_guess, self.error_bar, self.trace, args, datetime.datetime.now()
    #
    # def plot(self):
    #     self.analyze_data()
    #     print(self.frequency_guess)
    #     args = self.retrieve_args(self.run)
    #
    #     # fit_params = fits.Fit_1D_Freq_exp_with_cov(self.data, dt=args['step'])
    #
    #     t = np.arange(args['start'], args['stop'], args['step'])
    #     t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 10)
    #
    #     for i,trace in enumerate(self.traces):
    #         fig, ax = plt.subplots(figsize=[6,3])
    #
    #         fit_params = self.fit_params[i]
    #
    #         # print(t.shape, t)
    #         # print(trace.shape, trace)
    #
    #         ax.set_title(f"Ramsey decay {args['qubits'][i].hrid} transition {args['collection_name']}: \n"
    #                      f"{fit_params['Decay'][0]} $\pm$ {fit_params['Decay'][1]} us")
    #         ax.set_xlabel(f"Time (us) Frequency:{fit_params['Frequency'][0]}$\pm${fit_params['Frequency'][1]}")
    #         ax.set_ylabel("<z>")
    #
    #         g = ax.scatter(t, trace, marker='o', alpha=0.5)
    #         f = fit_params['Frequency'][0]
    #         a = fit_params['Amplitude'][0]
    #         p = fit_params['Phase'][0] - 2.0 * np.pi * f * args['start']
    #         o = fit_params['Offset'][0]
    #         T = fit_params['Decay'][0]
    #         fit = a * np.exp(-t_interpolate / T) * np.sin(2.0 * np.pi * f * t_interpolate + p) + o
    #         ax.plot(t_interpolate, fit)
    #         fig.tight_layout()
    #         plt.show()
    #
    # ## Not updated this for multi qubit yet:
    # def plot_fft(self, plot_range=(0.05, 1)):
    #     self.analyze_data()
    #     args = self.retrieve_args(self.run)
    #
    #     for i,trace in enumerate(self.traces):
    #         fig, ax = plt.subplots(figsize=[6, 3])
    #
    #         dt = args['step']
    #         rfft = np.abs(np.fft.rfft(trace))
    #         f = np.fft.rfftfreq(len(trace), dt)
    #
    #         mask = (f > plot_range[0]) & (f < plot_range[1])
    #
    #         ax.plot(f[mask], rfft[mask])
    #         ax.set_xlabel('Frequency [MHz]')
    #         ax.set_ylabel('Strength')
    #         ax.set_title(f"{args['qubits'][i].hrid} Ramsey spectrum")
    #         fig.tight_layout()
    #         plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class ConditionalStarkTuneUpRabiXY(experiment):
    @log_and_record
    def run(self, qubits, amp_control, amp_target, frequency=None, rise=0.01, trunc=1.0, start=0, stop=3, step=0.03, axis='Y',
            echo=False, iz_rate_cancel=0, phase_diff=0, iz_rise_drop=0):
        self.duts = qubits
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase = 0
        self.width = 0
        self.start = start
        self.stop = stop
        self.step = step
        self.rise = rise
        self.trunc= trunc
        self.fitting_2D = None
        self.phase_diff = phase_diff

        if frequency is None:
            freq_01 = qubits[1].get_c1('f01')['X'].freq
            freq_12 = qubits[1].get_c1('f12')['X'].freq

            anharmonicity = freq_01 - freq_12
            self.frequency = freq_01 - 0.3 * anharmonicity
            print(f"Choosing frequency {self.frequency}")
        else:
            self.frequency = frequency

        c1_control = self.duts[0].get_default_c1()
        c1_target = self.duts[1].get_default_c1()

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

        # flip_both = c1_control['Y'] * c1_target['Y']
        flip_both = c1_control['X'] * c1_target['X']
        if echo:
            lpb = cs_pulse + flip_both + cs_pulse + flip_both
        else:
            lpb = cs_pulse

        lpb_flip_control = prims.SweepLPB([c1_control['I'], c1_control['X']])
        swp_flip = sweeper.from_sweep_lpb(lpb_flip_control)

        lpb_readout = prims.SweepLPB([c1_target['Yp'], c1_target['Xm']])
        swp_readout = sweeper.from_sweep_lpb(lpb_readout)

        iz_gate = c1_target.z_omega(iz_rate_cancel * 2 * np.pi)
        iz_gate_fix = c1_target.z(-iz_rise_drop)

        lpb = c1_target['Ym'] * lpb_flip_control + lpb + iz_gate + iz_gate_fix + lpb_readout + mprim_target * mprim_control

        swpparams = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'width'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'width'),
            sparam.func(iz_gate.set_virtual_width, {}, 'width'),
        ]

        if echo:
            swp = sweeper(np.arange, n_kwargs={'start': start / 2, 'stop': stop / 2, 'step': step / 2},
                          params=swpparams)
        else:
            swp = sweeper(np.arange, n_kwargs={'start': start, 'stop': stop, 'step': step},
                          params=swpparams)

        basic(lpb, swp=swp + swp_flip + swp_readout, basis="<z>")

        self.result = np.squeeze(mprim_target.result())
        self.result_control = np.squeeze(mprim_control.result())

    def analyze_results(self):
        # print("Shape of result:", self.result.shape)

        if self.fitting_2D is None:

            self.fitting_2D = []
            for i in range(2):
                self.real_part = self.result[:, i, 0]
                self.imag_part = self.result[:, i, 1]
                self.complex_data = self.real_part + 1j * self.imag_part

                self.fit_result = fits.fit_2d_freq(self.complex_data, dt=self.step, use_freq_bound=False)
                self.fitting_2D.append(self.fit_result)
                # print(f"Fit Results for {i}: {self.fit_result}")  # Debugging output

            self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1]['Frequency']) / 2
            self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1]['Frequency']) / 2

            self.iz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] + (self.fitting_2D[1]['Phase'])) / 2
            self.zz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] - (self.fitting_2D[1]['Phase'])) / 2

            print(f"IZ: {self.iz_rate: 0.5f} MHz, ZZ: {self.zz_rate: 0.5f} MHz")
            print(f"Phase IZ Contributions from Pulse Rise Drop: {self.iz_from_pulse_rise_drop: 0.5f} rad")
            print(f"Phase ZZ Contributions from Pulse Rise Drop: {self.zz_from_pulse_rise_drop: 0.5f} rad")

        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
            'iz_from_pulse_rise_drop': self.iz_from_pulse_rise_drop,
            'zz_from_pulse_rise_drop': self.zz_from_pulse_rise_drop
        }

    def analyze_results_with_errs(self):
        # print("Shape of result:", self.result.shape)

        if self.fitting_2D is None:
            self.fitting_2D = []
            for i in range(2):
                self.real_part = self.result[:, i, 0]
                self.imag_part = self.result[:, i, 1]
                self.complex_data = self.real_part + 1j * self.imag_part

                fit_results = fits.fit_2d_freq_with_cov(self.complex_data, dt=self.step, use_freq_bound=False)
                self.fitting_2D.append(fit_results)

            # Calculate iz_rate and zz_rate for Curve Fit
            self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1]['Frequency']) / 2
            self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1]['Frequency']) / 2
            self.iz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] + (self.fitting_2D[1]['Phase'])) / 2
            self.zz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] - (self.fitting_2D[1]['Phase'])) / 2

            print(f"IZ: {self.iz_rate: 0.5f} MHz: {self.zz_rate: 0.5f} MHz")
            print(f"Phase IZ Contributions from Pulse Rise Drop: {self.iz_from_pulse_rise_drop: 0.5f} rad")

        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
            'iz_from_pulse_rise_drop': self.iz_from_pulse_rise_drop,
            'zz_from_pulse_rise_drop': self.zz_from_pulse_rise_drop
        }

    @register_browser_function(available_after=(run,))
    def plot(self):

        self.plot_blochsphere()

        self.analyze_results_with_errs()

        args = {'start': self.start, 'stop': self.stop, 'step': self.step}

        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        def plot_specific_axis(data, label, fit_params, use_imaginary_part=False):
            plt.scatter(t, data, label=label, alpha=0.5)
            # plt.plot(t, data)

            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
            o = fit_params['Offset_real'].nominal_value + 1j * fit_params['Offset_imag'].nominal_value

            # fit = a * np.exp(-decay * t_interpolate) * np.exp(1j * (2.0 * np.pi * f * t_interpolate + p)) + o
            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o

            plt.plot(t_interpolate, np.real(fit) if not use_imaginary_part else np.imag(fit))

        plt.figure(figsize=(20, 5))
        plt.title(f"ZZ interaction Hamiltonian tomography - X axis")
        plot_specific_axis(data=self.result[:, 0, 0], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=False)

        plot_specific_axis(data=self.result[:, 1, 0], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=False)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<X>")
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.title(f"ZZ interaction Hamiltonian tomography - Y axis")
        plot_specific_axis(data=self.result[:, 0, 1], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=True)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=True)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<Y>")
        plt.legend()
        plt.show()

        # self.plot_rescaled_after_fit()

        self.plot_blochsphere_leakage_to_control()

        self.plot_leakage_to_control()

    # @register_browser_function(available_after=(run,))
    def plot_rescaled_after_fit(self):
        args = {'start': self.start, 'stop': self.stop, 'step': self.step}

        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        def rescale_and_center(data):
            data_centered = data - np.mean(data)
            data_min = np.min(data_centered)
            data_max = np.max(data_centered)
            data_rescaled = 2 * (data_centered - data_min) / (data_max - data_min) - 1
            return data_rescaled

        def plot_specific_axis(data, label, fit_params, use_imaginary_part=False, color='blue'):
            data_rescaled = rescale_and_center(data)
            plt.scatter(t, data_rescaled, label=label, alpha=0.5, color=color)

            f = fit_params['Frequency'].nominal_value
            a = 1  # Fixed amplitude
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
            o = 0  # Offset is set to 0 for centering

            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o
            fit_rescaled = rescale_and_center(np.real(fit) if not use_imaginary_part else np.imag(fit))

            plt.plot(t_interpolate, fit_rescaled, color=color)

        dark_navy = '#000080'
        dark_purple = '#800080'

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction rescaled Hamiltonian tomography - X axis")
        plot_specific_axis(data=self.result[:, 0, 0], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=False, color=dark_navy)
        plot_specific_axis(data=self.result[:, 1, 0], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=False, color=dark_purple)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<X>")
        plt.legend()

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction rescaled Hamiltonian tomography - Y axis")
        plot_specific_axis(data=self.result[:, 0, 1], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=True, color=dark_navy)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=True, color=dark_purple)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<Y>")
        plt.legend()
        plt.show()

    # @register_browser_function(available_after=(analyze_results_rescaled,))
    def plot_rescaled_before_fit(self):
        args = {'start': self.start, 'stop': self.stop, 'step': self.step}

        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        def rescale_and_center(data):
            data_centered = data - np.mean(data)
            data_min = np.min(data_centered)
            data_max = np.max(data_centered)
            data_rescaled = 2 * (data_centered - data_min) / (data_max - data_min) - 1
            return data_rescaled

        def plot_specific_axis(data, label, fit_params, use_imaginary_part=False, color='blue'):
            data_rescaled = rescale_and_center(data)
            plt.scatter(t, data_rescaled, label=label, alpha=0.5, color=color)

            f = fit_params['Frequency'].nominal_value
            a = 1  # Fixed amplitude
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
            o = 0  # Offset is set to 0 for centering

            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o
            fit_rescaled = rescale_and_center(np.real(fit) if not use_imaginary_part else np.imag(fit))

            plt.plot(t_interpolate, fit_rescaled, color=color)

        results = self.analyze_results_rescaled()
        rescaled_fitting_2D = results['rescaled_fitting_2D']

        dark_navy = '#000080'
        dark_purple = '#800080'

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Hamiltonian tomography (rescaled before fit)- X axis")
        plot_specific_axis(data=self.result[:, 0, 0], label="Ground", fit_params=rescaled_fitting_2D[0],
                           use_imaginary_part=False, color=dark_navy)
        plot_specific_axis(data=self.result[:, 1, 0], label="Excited", fit_params=rescaled_fitting_2D[1],
                           use_imaginary_part=False, color=dark_purple)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<X>")
        plt.legend()

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Hamiltonian tomography (rescaled before fit) - Y axis")
        plot_specific_axis(data=self.result[:, 0, 1], label="Ground", fit_params=rescaled_fitting_2D[0],
                           use_imaginary_part=True, color=dark_navy)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited", fit_params=rescaled_fitting_2D[1],
                           use_imaginary_part=True, color=dark_purple)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<Y>")
        plt.legend()
        plt.show()

    def plot_fft(self):
        args = {'start': self.start, 'stop': self.stop, 'step': self.step}

        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        def plot_axis(data1, data2, label1, label2, fit_params1, fit_params2, t, use_imaginary_part=False):
            plt.figure(figsize=(18, 9))

            # Plot the original data
            plt.subplot(2, 1, 1)
            plt.scatter(t, data1, label=label1, alpha=0.5)
            plt.scatter(t, data2, label=label2, alpha=0.5)

            plt.xlabel('Time (us)')
            plt.ylabel('Amplitude')
            plt.title(f'{label1} and {label2} Data')
            plt.legend()

            # Compute the FFT of the data
            N = len(data1)
            T = t[1] - t[0]  # Sampling interval
            yf1 = fft(data1)
            yf2 = fft(data2)
            xf = fftfreq(N, T)[:N // 2]
            amplitudes1 = 2.0 / N * np.abs(yf1[0:N // 2])
            amplitudes2 = 2.0 / N * np.abs(yf2[0:N // 2])

            # Plot the FFT
            plt.subplot(2, 1, 2)
            plt.plot(xf, amplitudes1, label='FFT of ' + label1)
            plt.plot(xf, amplitudes2, label='FFT of ' + label2)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude')
            plt.title('FFT of the Data')
            plt.legend()

            # Find the frequency with the highest amplitude for both datasets
            max_amplitude_index1 = np.argmax(amplitudes1)
            max_frequency1 = xf[max_amplitude_index1]
            max_amplitude1 = amplitudes1[max_amplitude_index1]

            max_amplitude_index2 = np.argmax(amplitudes2)
            max_frequency2 = xf[max_amplitude_index2]
            max_amplitude2 = amplitudes2[max_amplitude_index2]

            # Print the highest frequency component for both datasets
            print(f"Highest frequency component for {label1}: {max_frequency1} MHz with amplitude {max_amplitude1}")
            print(f"Highest frequency component for {label2}: {max_frequency2} MHz with amplitude {max_amplitude2}")

            plt.tight_layout()
            plt.show()

        result = self.result.squeeze()

        plot_axis(data1=result[:, 0, 0], data2=self.result[:, 1, 0], label1="Ground - ZZ interaction rabi drive X axis",
                  label2="Excited - ZZ interaction rabi drive X axis", fit_params1=self.fitting_2D[0],
                  fit_params2=self.fitting_2D[1], t=t, use_imaginary_part=False)

        plot_axis(data1=result[:, 0, 1], data2=self.result[:, 1, 1], label1="Ground - ZZ interaction rabi drive Y axis",
                  label2="Excited - ZZ interaction rabi drive Y axis", fit_params1=self.fitting_2D[0],
                  fit_params2=self.fitting_2D[1], t=t, use_imaginary_part=True)

    # @register_browser_function(available_after=(run,))
    def plot_blochsphere(self):
        # Define colors
        dark_navy = '#000080'
        dark_purple = '#800080'

        # Generate data for the first subplot  ----- C-Ground
        X1 = self.result[:, 0, 0]
        Y1 = self.result[:, 0, 1]

        # Generate data for the second subplot ----- C-Excited
        X2 = self.result[:, 1, 0]
        Y2 = self.result[:, 1, 1]

        # Combine the two Bloch spheres into one figure
        fig = plt.figure(figsize=(14, 7))

        # Add title to the entire figure
        fig.suptitle('Bloch Sphere of Target Qubit')

        # Bloch sphere for the first subplot (C-Ground)
        ax1 = fig.add_subplot(121, projection='3d')
        b1 = Bloch(fig=fig, axes=ax1)
        b1.add_vectors([1, 0, 0])  # X-axis
        b1.add_vectors([0, 1, 0])  # Y-axis
        z1 = np.zeros_like(X1)
        points1 = [X1, Y1, z1]
        b1.add_points(points1, dark_navy)
        b1.render()
        ax1.set_title('Control in ground state')

        # Bloch sphere for the second subplot (C-Excited)
        ax2 = fig.add_subplot(122, projection='3d')
        b2 = Bloch(fig=fig, axes=ax2)
        b2.add_vectors([1, 0, 0])  # X-axis
        b2.add_vectors([0, 1, 0])  # Y-axis
        z2 = np.zeros_like(X2)
        points2 = [X2, Y2, z2]
        b2.add_points(points2, dark_purple)
        b2.render()
        ax2.set_title('Control in excited state')

        plt.tight_layout()
        plt.show()

        # Combine the two projections into another figure
        fig_proj = plt.figure(figsize=(14, 7))

        # Get the limits for the projection plots based on the ground state
        x_min = min(X1.min(), X2.min())
        x_max = max(X1.max(), X2.max())
        y_min = min(Y1.min(), Y2.min())
        y_max = max(Y1.max(), Y2.max())

        # Projection on the XY plane for the first subplot (C-Ground)
        ax_proj1 = fig_proj.add_subplot(121)
        ax_proj1.scatter(X1, Y1, color=dark_navy, label='Ground')
        ax_proj1.set_xlabel('X')
        ax_proj1.set_ylabel('Y')
        ax_proj1.set_title('XY Plane')
        ax_proj1.axhline(0, color='grey', linewidth=0.5)
        ax_proj1.axvline(0, color='grey', linewidth=0.5)
        ax_proj1.grid(True)
        ax_proj1.set_xlim(x_min - 0.1, x_max + 0.1)
        ax_proj1.set_ylim(y_min - 0.1, y_max + 0.1)
        ax_proj1.legend()

        # Projection on the XY plane for the second subplot (C-Excited)
        ax_proj2 = fig_proj.add_subplot(122)
        ax_proj2.scatter(X2, Y2, color=dark_purple, label='Excited')
        ax_proj2.set_xlabel('X')
        ax_proj2.set_ylabel('Y')
        ax_proj2.set_title('XY Plane')
        ax_proj2.axhline(0, color='grey', linewidth=0.5)
        ax_proj2.axvline(0, color='grey', linewidth=0.5)
        ax_proj2.grid(True)
        ax_proj2.set_xlim(x_min - 0.1, x_max + 0.1)
        ax_proj2.set_ylim(y_min - 0.1, y_max + 0.1)
        ax_proj2.legend()

        plt.tight_layout()
        plt.show()

    def plot_leakage_to_control(self):
        args = {'start': self.start, 'stop': self.stop, 'step': self.step}

        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        def plot_specific_axis(data, label, color):
            plt.scatter(t, data, label=label, alpha=0.5, color=color)
            plt.plot(t, data, color=color)

        plt.figure(figsize=(20, 5))
        plt.title(f"ZZ interaction Leakage to Control Hamiltonian tomography - X axis")
        plot_specific_axis(data=self.result_control[:, 0, 0], label="Ground", color='#1f77b4')
        plot_specific_axis(data=self.result_control[:, 1, 0], label="Excited", color='#8B0000')

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<X>")
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.title(f"ZZ interaction Leakage to Control Hamiltonian tomography - Y axis")
        plot_specific_axis(data=self.result_control[:, 0, 1], label="Ground", color='#1f77b4')
        plot_specific_axis(data=self.result_control[:, 1, 1], label="Excited", color='#8B0000')

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<Y>")
        plt.legend()
        plt.show()

    def plot_blochsphere_leakage_to_control(self):
        # Define colors
        dark_navy = '#000080'
        dark_purple = '#800080'

        # Generate data for the first subplot  ----- C-Ground
        X1 = self.result_control[:, 0, 0]
        Y1 = self.result_control[:, 0, 1]

        # Generate data for the second subplot ----- C-Excited
        X2 = self.result_control[:, 1, 0]
        Y2 = self.result_control[:, 1, 1]

        # Combine the two Bloch spheres into one figure
        fig = plt.figure(figsize=(14, 7))

        # Add title to the entire figure
        fig.suptitle('Bloch Sphere of Control Qubit')

        # Bloch sphere for the first subplot (C-Ground)
        ax1 = fig.add_subplot(121, projection='3d')
        b1 = Bloch(fig=fig, axes=ax1)
        b1.add_vectors([1, 0, 0])  # X-axis
        b1.add_vectors([0, 1, 0])  # Y-axis
        z1 = np.zeros_like(X1)
        points1 = [X1, Y1, z1]
        b1.add_points(points1, dark_navy)
        b1.render()
        ax1.set_title('Control in ground state')

        # Bloch sphere for the second subplot (C-Excited)
        ax2 = fig.add_subplot(122, projection='3d')
        b2 = Bloch(fig=fig, axes=ax2)
        b2.add_vectors([1, 0, 0])  # X-axis
        b2.add_vectors([0, 1, 0])  # Y-axis
        z2 = np.zeros_like(X2)
        points2 = [X2, Y2, z2]
        b2.add_points(points2, dark_purple)
        b2.render()
        ax2.set_title('Control in excited state')

        plt.tight_layout()
        plt.show()

        # Combine the two projections into another figure
        fig_proj = plt.figure(figsize=(14, 7))

        # Get the limits for the projection plots based on the ground state
        x_min = min(X1.min(), X2.min())
        x_max = max(X1.max(), X2.max())
        y_min = min(Y1.min(), Y2.min())
        y_max = max(Y1.max(), Y2.max())

        # Projection on the XY plane for the first subplot (C-Ground)
        ax_proj1 = fig_proj.add_subplot(121)
        ax_proj1.scatter(X1, Y1, color=dark_navy, label='Ground')
        ax_proj1.set_xlabel('X')
        ax_proj1.set_ylabel('Y')
        ax_proj1.set_title('XY Plane')
        ax_proj1.axhline(0, color='grey', linewidth=0.5)
        ax_proj1.axvline(0, color='grey', linewidth=0.5)
        ax_proj1.grid(True)
        ax_proj1.set_xlim(x_min - 0.1, x_max + 0.1)
        ax_proj1.set_ylim(y_min - 0.1, y_max + 0.1)
        ax_proj1.legend()

        # Projection on the XY plane for the second subplot (C-Excited)
        ax_proj2 = fig_proj.add_subplot(122)
        ax_proj2.scatter(X2, Y2, color=dark_purple, label='Excited')
        ax_proj2.set_xlabel('X')
        ax_proj2.set_ylabel('Y')
        ax_proj2.set_title('XY Plane')
        ax_proj2.axhline(0, color='grey', linewidth=0.5)
        ax_proj2.axvline(0, color='grey', linewidth=0.5)
        ax_proj2.grid(True)
        ax_proj2.set_xlim(x_min - 0.1, x_max + 0.1)
        ax_proj2.set_ylim(y_min - 0.1, y_max + 0.1)
        ax_proj2.legend()

        plt.tight_layout()
        plt.show()

class ConditionalStarkTuneUpRepeatedGateXY(Experiment):

    @log_and_record
    def run(self, duts, amp_control, amp_target, frequency, phase=0, rise=0.03, axis='Y',
            echo=False, iz_control=0, iz_target=0, width=0, start_gate_number=0, gate_count=40):
        """
        Sweep time and find the initial guess of amplitude

        :return:
        """
        self.duts = duts
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase = phase
        self.width = width
        self.iz_control = iz_control
        self.iz_target = iz_target
        self.rise = rise
        self.start_gate_number = start_gate_number
        self.gate_count = gate_count

        c1_control = self.duts[0].get_default_c1()
        c1_target = self.duts[1].get_default_c1()

        c2 = prims.build_CZ_stark_from_parameters(
            control_q=self.duts[0],
            target_q=self.duts[1],
            amp_target=self.amp_target,
            amp_control=self.amp_control,
            frequency=self.frequency,
            rise=self.rise,
            width=self.width,
            phase_diff=self.phase,
            iz_control=self.iz_control,
            iz_target=self.iz_target,
            echo=echo,
            trunc=1.05,
            zz_interaction_positive=True  # It doesn't matter what to use here, after one tomography we will find out.
        )

        cs_pulse = c2.get_z_canceled_cs_pulse()

        lpb = cs_pulse

        lpb_flip_control = prims.SweepLPB([c1_control['I'], c1_control['X']])
        swp_flip = sweeper.from_sweep_lpb(lpb_flip_control)

        lpb_readout = prims.SweepLPB([c1_target['Yp'], c1_target['Xm']])
        swp_readout = sweeper.from_sweep_lpb(lpb_readout)

        self.pulse_train, self.result = self.run_repeated_gate_experiment(
            initial_lpb=c1_target['Ym'],
            initial_gate=lpb_flip_control,
            repeated_block=lpb,
            final_gate=lpb_readout,
            pulse_count=range(start_gate_number, start_gate_number + gate_count),
            swp_initial=swp_flip,
            swp_posterior=swp_readout,
            fit=False
        )

        self.analyze_results()

    def run_repeated_gate_experiment(self, initial_lpb, initial_gate, repeated_block, final_gate, pulse_count,
                                     swp_initial, swp_posterior, fit=True):
        """
        Function to run the repeated gate experiment based on inatial lpb, gate and pulse count.
        """
        int_target = initial_lpb
        ini_control = initial_gate
        rep = repeated_block
        fin = final_gate

        mprim_control = self.duts[0].get_measurement_prim_intlist(0)
        mprim_target = self.duts[1].get_measurement_prim_intlist(0)

        sequence_lpb = []
        results = []

        for n in pulse_count:
            sequence = LogicalPrimitiveBlockSerial(
                [int_target * ini_control] + [rep] * (n) + [fin + mprim_target * mprim_control])
            sequence_lpb.append(sequence)

        lpb = LogicalPrimitiveBlockSweep(sequence_lpb)
        swp = sweeper.from_sweep_lpb(lpb)

        swp_flip = swp_initial
        swp_readout = swp_posterior

        basic(lpb, swp=swp + swp_flip + swp_readout, basis="<z>")

        self.result = np.squeeze(mprim_target.result())
        self.result_control = np.squeeze(mprim_control.result())

        self.N = 1
        self.pulse_count = pulse_count

        if fit:
            self.fit()

        return lpb, self.result

    def analyze_results(self):
        print("Shape of result:", self.result.shape)

        t_start = self.start_gate_number
        t_stop = self.start_gate_number + self.gate_count
        t_step = 1

        t = np.arange(t_start, t_stop, t_step)

        self.fitting_2D = []
        for i in range(2):
            self.real_part = self.result[:, i, 0]
            self.imag_part = self.result[:, i, 1]

            self.complex_data = self.real_part + 1j * self.imag_part

            self.fit_result = fit_2d_freq_with_cov(self.complex_data, dt=t_step, freq_guess=0.125, use_freq_bound=True)
            self.fitting_2D.append(self.fit_result)
            # print(f"Fit Results for {i}: {self.fit_result}")

        self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1]['Frequency']) / 2
        self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1]['Frequency']) / 2

        print(f"IZ: {self.iz_rate: 0.5f} PGC, ZZ: {self.zz_rate: 0.5f} PGC (per gate count)")

        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
        }

    @register_browser_function(available_after=(run,))
    def plot(self):
        """
        Plot the results.
        """
        args = self.retrieve_args(self.run)

        t = np.arange(args['start_gate_number'], args['start_gate_number'] + args['gate_count'], 1)
        t_interpolate = np.arange(args['start_gate_number'], args['start_gate_number'] + args['gate_count'], 1 / 10)

        def plot_specific_axis(data, label, fit_params, use_imaginary_part):
            data = data.squeeze()

            plt.scatter(t, data, label=label, alpha=0.5)
            # plt.plot(t, data)

            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start_gate_number']
            o = fit_params['Offset_real'].nominal_value + 1j * fit_params['Offset_imag'].nominal_value

            # fit = a * np.exp(-decay * t_interpolate) * np.exp(1j * (2.0 * np.pi * f * t_interpolate + p)) + o
            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o

            plt.plot(t_interpolate, np.real(fit) if not use_imaginary_part else np.imag(fit))

        plt.figure(figsize=(20, 5))

        desired_num_ticks = 10  # Desired number of ticks
        step = max(1, len(t) // desired_num_ticks)
        xticks_subset = t[::step]
        plt.title(f"ZZ interaction repeated gate tomography - X axis")

        plot_specific_axis(data=self.result[:, 0, 0], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=False)
        plot_specific_axis(data=self.result[:, 1, 0], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=False)

        plt.xlabel("Pulse count")
        plt.ylabel("<X>")
        plt.legend()
        plt.xticks(xticks_subset)

        plt.figure(figsize=(20, 5))
        plt.title(f"ZZ interaction repeated gate tomography - Y axis")

        plot_specific_axis(data=self.result[:, 0, 1], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=True)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=True)

        plt.xlabel("Pulse count")
        plt.ylabel("<Y>")
        plt.legend()
        plt.xticks(xticks_subset)
        plt.show()

        # self.plot_rescaled_after_fit()

    def plot_rescaled_after_fit(self):
        args = self.retrieve_args(self.run)

        t = np.arange(args['start_gate_number'], args['start_gate_number'] + args['gate_count'], 1)
        t_interpolate = np.arange(args['start_gate_number'], args['start_gate_number'] + args['gate_count'], 1 / 10)

        def rescale_and_center(data):
            data_centered = data - np.mean(data)
            data_min = np.min(data_centered)
            data_max = np.max(data_centered)
            data_rescaled = 2 * (data_centered - data_min) / (data_max - data_min) - 1
            return data_rescaled

        def plot_specific_axis(data, label, fit_params, use_imaginary_part=False, color='blue'):
            data = data.squeeze()
            data_rescaled = rescale_and_center(data)
            plt.scatter(t, data_rescaled, label=label, alpha=0.5, color=color)

            f = fit_params['Frequency'].nominal_value
            a = 1  # Fixed amplitude
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start_gate_number']
            o = 0  # Offset is set to 0 for centering

            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o
            fit_rescaled = rescale_and_center(np.real(fit) if not use_imaginary_part else np.imag(fit))

            plt.plot(t_interpolate, fit_rescaled, color=color)

        dark_navy = '#000080'
        dark_purple = '#800080'

        plt.figure(figsize=(20, 5))

        desired_num_ticks = 10  # Desired number of ticks
        step = max(1, len(t) // desired_num_ticks)
        xticks_subset = t[::step]
        plt.title(f"ZZ interaction rescaled repeated gate tomography - X axis")

        plot_specific_axis(data=self.result[:, 0, 0], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=False, color=dark_navy)
        plot_specific_axis(data=self.result[:, 1, 0], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=False, color=dark_purple)

        plt.xlabel("Pulse count")
        plt.ylabel("<X>")
        plt.legend()
        plt.xticks(xticks_subset)

        plt.figure(figsize=(20, 5))
        plt.title(f"ZZ interaction rescaled repeated gate tomography - Y axis")

        plot_specific_axis(data=self.result[:, 0, 1], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=True)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=True)

class ConditionalStarkEchoTuneUp(Experiment):

    @log_and_record
    def run(self, duts, params=None, frequency=None, amp_control=None, phase_diff=0, rise=0.01, trunc=1.0,
            t_start=0, t_stop=20, sweep_points=30,
            n_start=0, n_stop=32, update_iz=False, update_zz=True, n_max_iteration=20
            ):
        self.duts = duts
        self.n_max_iteration = n_max_iteration

        assert update_iz == False

        if params is None:
            amp_rabi_control = duts[0].get_c1('f01')['X'].amp
            amp_rabi_target = duts[1].get_c1('f01')['X'].amp

            area_control = amp_rabi_control * duts[0].get_c1('f01')['X'].width
            area_target = amp_rabi_target * duts[1].get_c1('f01')['X'].width


            params = {
                'iz_control': 0,
                'iz_target': 0,
                'frequency': frequency,
                'amp_control': amp_control,
                'amp_target': amp_control * area_target / area_control,
                # 'amp_target': amp_control,
                'rise': rise,
                'trunc': trunc,
                'width': 0,
                'phase_diff': phase_diff,
                'zz_interaction_positive': True,
                'echo': True
            }

        self.current_params = params
        self.params_list = [params]

        iz_rate, zz_rate = self.run_sizzel_xy_hamiltonian_tomography(t_start=t_start, t_stop=t_stop,
                                                                     sweep_points=sweep_points)

        self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

        self.run_repeated_gate_hamiltonian_tomography(duts=self.duts, zz_rate=zz_rate, n_start=n_start, n_stop=n_stop,
                                                      update_iz=False, update_zz=True)

    def run_sizzel_xy_hamiltonian_tomography(self, t_start, t_stop, sweep_points=60):

        t_step = (t_stop - t_start) / sweep_points

        sizzel_xy = ConditionalStarkTuneUpRabiXY(
            qubits=self.duts,
            frequency=self.current_params['frequency'],
            amp_control=self.current_params['amp_control'],
            amp_target=self.current_params['amp_target'],
            rise=self.current_params['rise'],
            start=t_start,
            stop=t_stop,
            step=t_step,
            phase_diff=self.current_params['phase_diff'],
            iz_rate_cancel=0,
            iz_rise_drop=0,
            echo=True)

        result = sizzel_xy.analyze_results_with_errs()

        iz_rate = result['iz_rate']
        zz_rate = result['zz_rate']

        new_params = self.current_params.copy()
        new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2

        print(f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]} us')

        self.params_list.append(new_params)
        self.current_params = new_params

        return iz_rate, zz_rate

    def run_repeated_gate_hamiltonian_tomography(self, duts, zz_rate, n_start=0, n_stop=32, update_iz=False,
                                                 update_zz=True):

        iz_target = self.current_params['iz_target']
        width = self.current_params['width']
        iz_check_pass = False
        zz_check_pass = False

        measured_iz_list = []
        measured_zz_list = []

        estimated_iz_list = []
        estimated_zz_list = []

        kalman_iz = None
        kalman_zz = None

        for i in range(self.n_max_iteration):
            repeated_gate = ConditionalStarkTuneUpRepeatedGateXY(
                duts=self.duts,
                iz_control=0,
                iz_target=iz_target,
                frequency=self.current_params['frequency'],
                amp_control=self.current_params['amp_control'],
                amp_target=self.current_params['amp_target'],
                rise=self.current_params['rise'],
                width=width,
                start_gate_number=n_start,
                gate_count=n_stop,
                echo=True,
            )

            iz_target_measured = iz_target + repeated_gate.iz_rate.nominal_value * np.pi * 2
            zz_measured = repeated_gate.zz_rate.nominal_value

            measured_iz_list.append(iz_target_measured)
            measured_zz_list.append(zz_measured)

            if kalman_iz is None:
                kalman_iz = KalmanFilter1D(initial_position=iz_target_measured,
                                           position_variance=(repeated_gate.iz_rate.std_dev * np.pi * 2) ** 2)
                kalman_zz = KalmanFilter1D(initial_position=zz_measured,
                                           position_variance=(repeated_gate.zz_rate.std_dev * np.pi * 2) ** 2)
            else:
                kalman_iz.update(measurement=iz_target_measured,
                                 measurement_variance=(repeated_gate.iz_rate.std_dev * np.pi * 2) ** 2)

                kalman_zz.update(measurement=zz_measured,
                                 measurement_variance=(repeated_gate.zz_rate.std_dev * np.pi * 2) ** 2)

            print(f'Kalman estimated ZZ pgc after measurement = {kalman_zz.x}+-{np.sqrt(kalman_zz.P)}')
            print(f'Kalman estimated IZ pgc after measurement = {kalman_iz.x}+-{np.sqrt(kalman_iz.P)}')

            if update_iz:
                iz_target = kalman_iz.x
                iz_check_pass = kalman_iz.P < 1e-2
                if not update_zz:
                    estimated_iz_list.append(unc.ufloat(kalman_iz.x, np.sqrt(kalman_iz.P)))

            if update_zz:
                zz_pgc = kalman_zz.x

                target_zz = np.sign(zz_pgc) * 0.125
                zz_diff = target_zz - zz_pgc
                width_diff = np.sign(zz_diff / zz_rate.nominal_value) * min(np.abs(zz_diff / zz_rate.nominal_value / 2),
                                                                            0.05 * self.current_params['width'])
                zz_diff = zz_rate.nominal_value * width_diff * 2
                width += width_diff
                iz_diff = 0
                # iz_rate_tQ1_cQ2 * width_diff * np.pi * 2
                print(f'Update width to {width} us')
                kalman_zz.predict(movement=zz_diff,
                                  position_variance=(zz_rate.std_dev * width_diff * np.pi * 2) ** 2)
                kalman_iz.predict(movement=iz_diff,
                                  position_variance=(zz_rate.std_dev * width_diff * np.pi * 2) ** 2)

                estimated_iz_list.append(unc.ufloat(kalman_iz.x, np.sqrt(kalman_iz.P)))
                estimated_zz_list.append(unc.ufloat(kalman_zz.x, np.sqrt(kalman_zz.P)))

            zz_accuracy_check = np.abs(target_zz - zz_pgc) < 1e-3
            zz_uncertainty_check = np.sqrt(kalman_zz.P) < 1e-3
            zz_check_pass = zz_accuracy_check and zz_uncertainty_check

            print(f'Kalman estimated ZZ pgc after update = {kalman_zz.x}+-{np.sqrt(kalman_zz.P)}')
            print(f'Kalman estimated IZ pgc after update = {kalman_iz.x}+-{np.sqrt(kalman_iz.P)}')

            print(f'ZZ accuracy check pass: {zz_accuracy_check}, ZZ uncertainty check pass: {zz_uncertainty_check}')
            print(f'IZ uncertainty check pass: {iz_check_pass} ')

            if (iz_check_pass or not update_iz) and (zz_check_pass or not update_zz):
                break

        new_params = self.current_params.copy()

        new_params['iz_target'] = iz_target
        new_params['width'] = width

        print(f'Estimated IZ = {iz_target}, ZZ = {zz_pgc}, width = {width}')

        self.params_list.append(new_params)
        self.current_params = new_params

        self.estimated_iz_list = estimated_iz_list
        self.estimated_zz_list = estimated_zz_list

class ConditionalStarkSpectroscopyDiffAmpFreq(experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the frequency and amplitude parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], freq_start: float = 4100, freq_stop: float = 4144, freq_step: float = 1,
            amp_start: float = 0, amp_stop: float = 0.2, amp_step: float = 0.02,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the amplitude and frequency and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            freq_start (float): Starting frequency for the sweep (MHz).
            freq_stop (float): Stopping frequency for the sweep (MHz).
            freq_step (float): Step size for the frequency sweep (MHz).
            amp_start (float): Starting amplitude for the sweep.
            amp_stop (float): Stopping amplitude for the sweep.
            amp_step (float): Step size for the amplitude sweep.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """
        # Clone the control pulse from each DUT for manipulation.
        cs_pulses = [dut.get_c1('f01')['X'].clone() for dut in duts]

        # Get the measurement primitives from each DUT.
        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]

        # Get the default control pulse for each DUT.
        c1s = [dut.get_default_c1() for dut in duts]

        # Flip both`
        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        # Update the pulse parameters for all cloned pulses.
        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.update_pulse_args(shape='soft_square', amp=0, phase=0, width=width if not echo else width / 2,
                                       rise=rise, trunc=trunc)

        # Create amplitude sweeper.
        swp_amp = sweeper(np.arange, n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
                          params=[sparam.func(cs_pulse.update_pulse_args, {}, 'amp') for cs_pulse in cs_pulses])

        # Create frequency sweeper.
        swp_freq = sweeper(np.arange, n_kwargs={'start': freq_start, 'stop': freq_stop, 'step': freq_step},
                           params=[sparam.func(cs_pulse.update_freq, {}, 'freq') for cs_pulse in cs_pulses])

        # Set up additional pulse sequences and sweep.
        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = prims.ParallelLPB(cs_pulses)
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        # lpb = flip_sweep_lpb + c1s[1]['Xp'] + lpb_zz + c1s[1]['Ym'] + prims.ParallelLPB(mprims)

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        # Execute the basic spectroscopy sequence with all sweeps combined.
        basic(lpb, swp=swp_amp + swp_freq + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Generate the heatmap. RdBu or viridis are good color scales.
        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
            ))

        # Set plot titles.
        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving amplitude (a.u)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Generate the heatmap. RdBu or viridis are good color scales.
        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
            ))

        # Set plot titles.
        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving amplitude (a.u)",
        )

        return fig

    # @register_browser_function(available_after=(run,))
    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Create a subplot figure with 1 row and 2 columns
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectroscopy Difference in Y axis on Target Qubit",
                                                            "Spectroscopy Difference in Y axis on Control Qubit"))

        # Generate the heatmap for the main result
        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),  # Adjust x to position the colorbar
        )

        # Generate the heatmap for the control result
        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),  # Adjust x to position the colorbar
        )

        # Add heatmaps to the figure
        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        # Update layout for the figure
        fig.update_layout(
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving Amplitude (a.u)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),  # Add margins to make space for titles
            width=1600,  # Increase the width for better resolution
            height=600,  # Increase the height for better resolution
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        # Customize x-axis and y-axis for both subplots
        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Frequency (MHz)', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)

        return fig

    def live_plots(self, step_no: tuple[int] = None):
        """
        Generate the live plots. This function is called by the live monitor.
        The step no denotes the number of data points to plot, while the
        buffer size is the total number of data points to plot. Some of the data
        in the buffer is note yet valid, so they should not be plotted.
        """

        return self.plot()

class ConditionalStarkSpectroscopyDiffAmpTargetFreq(experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the frequency and amplitude parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], freq_start: float = 4100, freq_stop: float = 4144, freq_step: float = 1,
            amp_control_fixed = 0.2,
            amp_start: float = 0, amp_stop: float = 0.2, amp_step: float = 0.02,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the amplitude and frequency and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            freq_start (float): Starting frequency for the sweep (MHz).
            freq_stop (float): Stopping frequency for the sweep (MHz).
            freq_step (float): Step size for the frequency sweep (MHz).
            amp_start (float): Starting amplitude for the sweep.
            amp_stop (float): Stopping amplitude for the sweep.
            amp_step (float): Step size for the amplitude sweep.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """

        self.amp_control_fixed = amp_control_fixed

        # Clone the control pulse from each DUT for manipulation.
        cs_pulses = [dut.get_c1('f01')['X'].clone() for dut in duts]

        # Get the measurement primitives from each DUT.
        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]

        # Get the default control pulse for each DUT.
        c1s = [dut.get_default_c1() for dut in duts]

        # Flip both
        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        # Update the pulse parameters for all cloned pulses.
        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.update_pulse_args(shape='soft_square', amp=self.amp_control_fixed, phase=0, width=width if not echo else width / 2,
                                       rise=rise, trunc=trunc)

        # # Create amplitude sweeper.
        # swp_amp = sweeper(np.arange, n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
        #                   params=[sparam.func(cs_pulse.update_pulse_args, {}, 'amp') for cs_pulse in cs_pulses])

        # Create amplitude sweeper and apply only to the second cs_pulse only
        swp_amp = sweeper(
            np.arange,
            n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
            params=[sparam.func(cs_pulses[1].update_pulse_args, {}, 'amp')]  # Target only the second pulse
        )

        # Create frequency sweeper.
        swp_freq = sweeper(np.arange, n_kwargs={'start': freq_start, 'stop': freq_stop, 'step': freq_step},
                           params=[sparam.func(cs_pulse.update_freq, {}, 'freq') for cs_pulse in cs_pulses])

        # Set up additional pulse sequences and sweep.
        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = prims.ParallelLPB(cs_pulses)
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        # lpb = flip_sweep_lpb + c1s[1]['Xp'] + lpb_zz + c1s[1]['Ym'] + prims.ParallelLPB(mprims)

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        # Execute the basic spectroscopy sequence with all sweeps combined.
        basic(lpb, swp=swp_amp + swp_freq + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Generate the heatmap. RdBu or viridis are good color scales.
        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
            ))

        # Set plot titles.
        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit with Control Fixed Amplitude",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Target Driving amplitude (a.u)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Generate the heatmap. RdBu or viridis are good color scales.
        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
            ))

        # Set plot titles.
        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control with Control Fixed Amplitude",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Target Driving amplitude (a.u)",
        )

        return fig

    # @register_browser_function(available_after=(run,))
    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Create a subplot figure with 1 row and 2 columns
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectroscopy Difference in Y axis on Target Qubit with Control Fixed Amplitude",
                                                            "Spectroscopy Difference in Y axis on Control Qubit with Control Fixed Amplitude"))

        # Generate the heatmap for the main result
        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),  # Adjust x to position the colorbar
        )

        # Generate the heatmap for the control result
        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),  # Adjust x to position the colorbar
        )

        # Add heatmaps to the figure
        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        # Update layout for the figure
        fig.update_layout(
            xaxis_title="Frequency (MHz)",
            yaxis_title="Target Driving Amplitude (a.u)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),  # Add margins to make space for titles
            width=1600,  # Increase the width for better resolution
            height=600,  # Increase the height for better resolution
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        # Customize x-axis and y-axis for both subplots
        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Frequency (MHz)', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)

        return fig

    def live_plots(self, step_no: tuple[int] = None):
        """
        Generate the live plots. This function is called by the live monitor.
        The step no denotes the number of data points to plot, while the
        buffer size is the total number of data points to plot. Some of the data
        in the buffer is note yet valid, so they should not be plotted.
        """

        return self.plot()

class ConditionalStarkSpectroscopyDiffPhaseFreq(experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the frequency and phase parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], freq_start: float = 4100, freq_stop: float = 4144, freq_step: float = 1,
            phase_diff_start: float = 0, phase_diff_stop: float = np.pi, phase_diff_step: float = np.pi / 10,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, amp=0.2, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the phase and frequency and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            freq_start (float): Starting frequency for the sweep (MHz).
            freq_stop (float): Stopping frequency for the sweep (MHz).
            freq_step (float): Step size for the frequency sweep (MHz).
            phase_diff_start (float): Starting phase difference for the sweep.
            phase_diff_stop (float): Stopping phase difference for the sweep.
            phase_diff_step (float): Step size for the phase difference sweep.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            amp (float): Amplitude for the control and target pulses.
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """
        self.duts = duts
        self.frequency = freq_start
        self.amp_control = amp
        self.amp_target = amp
        self.width = width
        self.phase_diff = phase_diff_stop

        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]
        c1s = [dut.get_default_c1() for dut in duts]

        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        # Clone the control pulse from each DUT for manipulation.
        cs_pulses = [dut.get_c1('f01')['X'].clone() for dut in duts]

        # Update the pulse parameters for all cloned pulses.
        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.update_pulse_args(shape='soft_square', amp=0, phase=phase_diff_start, width=width if not echo else width / 2,
                                       rise=rise, trunc=trunc)

        # Create phase sweeper.
        # swp_phase = sweeper(np.arange, n_kwargs={'start': phase_diff_start, 'stop': phase_diff_stop, 'step': phase_diff_step},
        #                   params=[sparam.func(cs_pulse.update_pulse_args, {}, 'phase') for cs_pulse in cs_pulses])

        # Create amplitude sweeper and apply only to the second cs_pulse.
        swp_phase = sweeper(
            np.arange,
            n_kwargs={'start': phase_diff_start, 'stop': phase_diff_stop, 'step': phase_diff_step},
            params=[sparam.func(cs_pulses[1].update_pulse_args, {}, 'phase')]  # Target only the second pulse
        )

        # Create frequency sweeper.
        swp_freq = sweeper(np.arange, n_kwargs={'start': freq_start, 'stop': freq_stop, 'step': freq_step},
                           params=[sparam.func(cs_pulse.update_freq, {}, 'freq') for cs_pulse in cs_pulses])


        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = cs_pulse
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        basic(lpb, swp=swp_phase + swp_freq + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Phase Difference (radians)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Phase Difference (radians)",
        )

        return fig

    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectroscopy Difference in Y axis on Target Qubit",
                                                            "Spectroscopy Difference in Y axis on Control Qubit"))

        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),
        )

        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),
        )

        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        fig.update_layout(
            xaxis_title="Frequency (MHz)",
            yaxis_title="Phase Difference (radians)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),
            width=1600,
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Frequency (MHz)', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2,
                                       linecolor='black',
                                       mirror=True)

        return fig

class ConditionalStarkSpectroscopyDiffAmpPhase(experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the amplitude and phase parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], amp_start: float = 0, amp_stop: float = 0.2, amp_step: float = 0.02,
            phase_diff_start: float = 0, phase_diff_stop: float = np.pi, phase_diff_step: float = np.pi / 10,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, frequency=4700, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the amplitude and phase and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            amp_start (float): Starting amplitude for the sweep.
            amp_stop (float): Stopping amplitude for the sweep.
            amp_step (float): Step size for the amplitude sweep.
            phase_diff_start (float): Starting phase difference for the sweep.
            phase_diff_stop (float): Stopping phase difference for the sweep.
            phase_diff_step (float): Step size for the phase difference sweep.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            frequency (float): Frequency for the control and target pulses (MHz).
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """
        self.duts = duts
        self.frequency = frequency
        self.amp_control = amp_start
        self.amp_target = amp_start
        self.width = width
        self.phase_diff = phase_diff_step

        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]
        c1s = [dut.get_default_c1() for dut in duts]

        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        # Clone the control pulse from each DUT for manipulation.
        cs_pulses = [dut.get_c1('f01')['X'].clone() for dut in duts]

        # Update the pulse parameters for all cloned pulses.
        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.update_pulse_args(shape='soft_square', amp=0, phase=phase_diff_start, width=width if not echo else width / 2,
                                       rise=rise, trunc=trunc)

        # Create amp sweeper.
        swp_amp = sweeper(np.arange, n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
                              params=[sparam.func(cs_pulse.update_pulse_args, {}, 'amp') for cs_pulse in cs_pulses])

        # Create phase sweeper.
        # swp_phase = sweeper(np.arange, n_kwargs={'start': phase_diff_start, 'stop': phase_diff_stop, 'step': phase_diff_step},
        #                     params=[sparam.func(cs_pulse.update_pulse_args, {}, 'phase') for cs_pulse in cs_pulses])

        # Create phase sweeper and apply only to the second cs_pulse.
        swp_phase = sweeper(
            np.arange,
            n_kwargs={'start': phase_diff_start, 'stop': phase_diff_stop, 'step': phase_diff_step},
            params=[sparam.func(cs_pulses[1].update_pulse_args, {}, 'phase')]  # Target only the second pulse
        )

        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = cs_pulse
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        basic(lpb, swp=swp_amp + swp_phase + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit",
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control Qubit",
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
        )

        return fig

    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectroscopy Difference in Y axis on Target Qubit",
                                                            "Spectroscopy Difference in Y axis on Control Qubit"))

        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),
        )

        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),
        )

        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        fig.update_layout(
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),
            width=1600,
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Phase Difference (radians)', showline=True, linewidth=2,
                                       linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)

        return fig

class ConditionalStarkSizzleSpectroscopyAmpFreq(experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the frequency and amplitude parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], freq_start: float = 4100, freq_stop: float = 4144, freq_step: float = 1,
            amp_start: float = 0, amp_stop: float = 0.2, amp_step: float = 0.02,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, phase_diff = 0, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the amplitude and frequency and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            freq_start (float): Starting frequency for the sweep (MHz).
            freq_stop (float): Stopping frequency for the sweep (MHz).
            freq_step (float): Step size for the frequency sweep (MHz).
            amp_start (float): Starting amplitude for the sweep.
            amp_stop (float): Stopping amplitude for the sweep.
            amp_step (float): Step size for the amplitude sweep.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """
        # Clone the control pulse from each DUT for manipulation.
        # cs_pulses = [dut.get_c1('f01')['X'].clone() for dut in duts]

        self.duts = duts
        self.frequency = freq_start
        self.amp_control = amp_start
        self.amp_target = amp_start
        self.width = width
        self.phase_diff = phase_diff


        # Get the measurement primitives from each DUT.
        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]

        # Get the default control pulse for each DUT.
        c1s = [dut.get_default_c1() for dut in duts]

        # Flip both
        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        # Update the pulse parameters for all cloned pulses.
        # for i, cs_pulse in enumerate(cs_pulses):
        #     cs_pulse.update_pulse_args(shape='soft_square', amp=0, phase=0, width=width if not echo else width / 2,
        #                                rise=rise, trunc=trunc)

        c2 = prims.build_CZ_stark_from_parameters(control_q=self.duts[0], target_q=self.duts[1],
                                                  amp_target=self.amp_target, amp_control=self.amp_control,
                                                  frequency=self.frequency,
                                                  rise=rise, width=self.width,
                                                  phase_diff=self.phase_diff,
                                                  iz_control=0,
                                                  iz_target=0,
                                                  echo=False,
                                                  trunc=1.0, zz_interaction_positive=True)
        cs_pulse = c2.get_stark_drive_pulses()
        stark_drive_target_pulse = c2['stark_drive_target']
        stark_drive_control_pulse = c2['stark_drive_control']

        swpparams_amp = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'amp_target'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'amp_control'),
        ]
        # Create amplitude sweeper.
        swp_amp = sweeper(np.arange, n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
                          params=swpparams_amp)

        swpparams_freq = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'frequency'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'frequency'),
        ]
        # Create frequency sweeper.
        swp_freq = sweeper(np.arange, n_kwargs={'start': freq_start, 'stop': freq_stop, 'step': freq_step},
                           params=swpparams_freq)

        # Set up additional pulse sequences and sweep.
        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = cs_pulse
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        # lpb = flip_sweep_lpb + c1s[1]['Xp'] + lpb_zz + c1s[1]['Ym'] + prims.ParallelLPB(mprims)

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        # Execute the basic spectroscopy sequence with all sweeps combined.
        basic(lpb, swp=swp_amp + swp_freq + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Generate the heatmap. RdBu or viridis are good color scales.
        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
            ))

        # Set plot titles.
        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving amplitude (a.u)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Generate the heatmap. RdBu or viridis are good color scales.
        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
            ))

        # Set plot titles.
        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving amplitude (a.u)",
        )

        return fig

    # @register_browser_function(available_after=(run,))
    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Create a subplot figure with 1 row and 2 columns
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectroscopy Difference in Y axis on Target Qubit",
                                                            "Spectroscopy Difference in Y axis on Control Qubit"))

        # Generate the heatmap for the main result
        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),  # Adjust x to position the colorbar
        )

        # Generate the heatmap for the control result
        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),  # Adjust x to position the colorbar
        )

        # Add heatmaps to the figure
        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        # Update layout for the figure
        fig.update_layout(
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving Amplitude (a.u)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),  # Add margins to make space for titles
            width=1600,  # Increase the width for better resolution
            height=600,  # Increase the height for better resolution
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        # Customize x-axis and y-axis for both subplots
        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Frequency (MHz)', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)

        return fig

    def live_plots(self, step_no: tuple[int] = None):
        """
        Generate the live plots. This function is called by the live monitor.
        The step no denotes the number of data points to plot, while the
        buffer size is the total number of data points to plot. Some of the data
        in the buffer is note yet valid, so they should not be plotted.
        """

        return self.plot()

class ConditionalStarkSizzleSpectroscopyPhaseFreq(experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the frequency and phase parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], freq_start: float = 4100, freq_stop: float = 4144, freq_step: float = 1,
            phase_diff_start: float = 0, phase_diff_stop: float = np.pi, phase_diff_step: float = np.pi / 10,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, amp=0.2, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the phase and frequency and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            freq_start (float): Starting frequency for the sweep (MHz).
            freq_stop (float): Stopping frequency for the sweep (MHz).
            freq_step (float): Step size for the frequency sweep (MHz).
            phase_diff_start (float): Starting phase difference for the sweep.
            phase_diff_stop (float): Stopping phase difference for the sweep.
            phase_diff_step (float): Step size for the phase difference sweep.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            amp (float): Amplitude for the control and target pulses.
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """
        self.duts = duts
        self.frequency = freq_start
        self.amp_control = amp
        self.amp_target = amp
        self.width = width
        self.phase_diff = phase_diff_stop

        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]
        c1s = [dut.get_default_c1() for dut in duts]

        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        c2 = prims.build_CZ_stark_from_parameters(control_q=self.duts[0], target_q=self.duts[1],
                                                  amp_target=self.amp_target, amp_control=self.amp_control,
                                                  frequency=self.frequency,
                                                  rise=rise, width=self.width,
                                                  phase_diff=self.phase_diff,
                                                  iz_control=0,
                                                  iz_target=0,
                                                  echo=False,
                                                  trunc=1.0, zz_interaction_positive=True)
        cs_pulse = c2.get_stark_drive_pulses()
        stark_drive_target_pulse = c2['stark_drive_target']
        stark_drive_control_pulse = c2['stark_drive_control']

        swpparams_phase = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'phase_diff'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'phase_diff'),
        ]
        swp_phase = sweeper(np.arange,
                            n_kwargs={'start': phase_diff_start, 'stop': phase_diff_stop, 'step': phase_diff_step},
                            params=swpparams_phase)

        swpparams_freq = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'frequency'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'frequency'),
        ]

        swp_freq = sweeper(np.arange, n_kwargs={'start': freq_start, 'stop': freq_stop, 'step': freq_step},
                           params=swpparams_freq)

        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = cs_pulse
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        basic(lpb, swp=swp_phase + swp_freq + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Phase Difference (radians)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control Qubit",
            xaxis_title="Frequency (MHz)",
            yaxis_title="Phase Difference (radians)",
        )

        return fig

    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectroscopy Difference in Y axis on Target Qubit",
                                                            "Spectroscopy Difference in Y axis on Control Qubit"))

        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),
        )

        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),
        )

        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        fig.update_layout(
            xaxis_title="Frequency (MHz)",
            yaxis_title="Phase Difference (radians)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),
            width=1600,
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Frequency (MHz)', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2,
                                       linecolor='black',
                                       mirror=True)

        return fig

class ConditionalStarkSizzleSpectroscopyAmpPhase(experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the amplitude and phase parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], amp_start: float = 0, amp_stop: float = 0.2, amp_step: float = 0.02,
            phase_diff_start: float = 0, phase_diff_stop: float = np.pi, phase_diff_step: float = np.pi / 10,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, frequency=4700, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the amplitude and phase and observing the difference in measuring Y axis.

        Args:
            duts (List[Any]): List of device under test instances.
            amp_start (float): Starting amplitude for the sweep.
            amp_stop (float): Stopping amplitude for the sweep.
            amp_step (float): Step size for the amplitude sweep.
            phase_diff_start (float): Starting phase difference for the sweep.
            phase_diff_stop (float): Stopping phase difference for the sweep.
            phase_diff_step (float): Step size for the phase difference sweep.
            rise (float): Rise time for the pulse shape.
            trunc (float): Truncation factor for the pulse shape.
            width (float): Width of the pulse shape.
            frequency (float): Frequency for the control and target pulses (MHz).
            echo (bool): Whether to include an echo pulse in the sequence.

        Returns:
            None
        """
        self.duts = duts
        self.frequency = frequency
        self.amp_control = amp_start
        self.amp_target = amp_start
        self.width = width
        self.phase_diff = phase_diff_step

        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]
        c1s = [dut.get_default_c1() for dut in duts]

        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        c2 = prims.build_CZ_stark_from_parameters(control_q=self.duts[0], target_q=self.duts[1],
                                                  amp_target=self.amp_target, amp_control=self.amp_control,
                                                  frequency=self.frequency,
                                                  rise=rise, width=self.width,
                                                  phase_diff=self.phase_diff,
                                                  iz_control=0,
                                                  iz_target=0,
                                                  echo=False,
                                                  trunc=1.0, zz_interaction_positive=True)
        cs_pulse = c2.get_stark_drive_pulses()
        stark_drive_target_pulse = c2['stark_drive_target']
        stark_drive_control_pulse = c2['stark_drive_control']

        swpparams_amp = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'amp_target'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'amp_control'),
        ]
        # Create amplitude sweeper.
        swp_amp = sweeper(np.arange, n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
                          params=swpparams_amp)

        swpparams_phase = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'phase_diff'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'phase_diff'),
        ]
        swp_phase = sweeper(np.arange,
                            n_kwargs={'start': phase_diff_start, 'stop': phase_diff_stop, 'step': phase_diff_step},
                            params=swpparams_phase)

        flip_sweep_lpb = prims.SweepLPB([c1s[0]['I'], c1s[0]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = cs_pulse
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        lpb = c1s[1]['Ym'] * flip_sweep_lpb + lpb_zz + c1s[1]['Xm'] + prims.ParallelLPB(mprims)

        self.mp_control = mprims[0]
        self.mp_target = mprims[1]

        basic(lpb, swp=swp_amp + swp_phase + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp_target.result())
        self.result_control = np.squeeze(self.mp_control.result())

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Target Qubit",
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_leakage_to_control(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]), x=ys, y=xs,
            colorscale='RdBu'
        ))

        fig.update_layout(
            title="Spectroscopy Difference in Y axis on Control Qubit",
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
        )

        return fig

    def plot_high_resolution(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as heatmaps.

        Returns:
            go.Figure: A figure with two high-resolution heatmaps, one for the main result and one for the control.
        """
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['phase_diff_start'], stop=args['phase_diff_stop'], step=args['phase_diff_step'])

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectroscopy Difference in Y axis on Target Qubit",
                                                            "Spectroscopy Difference in Y axis on Control Qubit"))

        heatmap_main = go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="", titleside="right", x=0.45),
        )

        heatmap_control = go.Heatmap(
            z=(self.result_control[:, :, 0] - self.result_control[:, :, 1]),
            x=ys,
            y=xs,
            colorscale='RdBu',
            colorbar=dict(title="Difference in Y axis", titleside="right", x=1.05),
        )

        fig.add_trace(heatmap_main, row=1, col=1)
        fig.add_trace(heatmap_control, row=1, col=2)

        fig.update_layout(
            xaxis_title="Phase Difference (radians)",
            yaxis_title="Amplitude (a.u)",
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(size=18, family="Arial, sans-serif"),
            margin=dict(l=80, r=20, t=40, b=80),
            width=1600,
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )

        fig['layout']['xaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['yaxis'].update(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig['layout']['xaxis2'].update(title='Phase Difference (radians)', showline=True, linewidth=2,
                                       linecolor='black',
                                       mirror=True)
        fig['layout']['yaxis2'].update(title='', showline=True, linewidth=2, linecolor='black',
                                       mirror=True)

        return fig

class ConditionalStarkFinePhaseTuneUp(Experiment):
    @log_and_record
    def run(self, duts, params=None, frequency=None, amp_control=None, phase_diff=0, rise=0.0, trunc=1.0,
            t_start=0, t_stop=20, sweep_points=30,
            phase_diff_start: float = 0, phase_diff_stop: float = 2*np.pi, phase_diff_step: float = np.pi / 10,
            n_start=0, n_stop=32, update_iz=False, update_zz=True
            ):
        self.duts = duts

        assert update_iz == False

        if params is None:
            amp_rabi_control = duts[0].get_c1('f01')['X'].amp
            amp_rabi_target = duts[1].get_c1('f01')['X'].amp

            area_control = amp_rabi_control * duts[0].get_c1('f01')['X'].width
            area_target = amp_rabi_target * duts[1].get_c1('f01')['X'].width

            params = {
                'iz_control': 0,
                'iz_target': 0,
                'frequency': frequency,
                'amp_control': amp_control,
                'amp_target': amp_control * area_target / area_control,
                'rise': rise,
                'trunc': trunc,
                'width': 0,
                'phase_diff': phase_diff,
                'zz_interaction_positive': True,
                'echo': True
            }

        self.current_params = params
        self.params_list = [params]
        self.results = []

        for phase in np.arange(phase_diff_start, phase_diff_stop, phase_diff_step):
            print(f"phase: {phase:.3f} radians")
            self.current_params['phase_diff'] = phase
            iz_rate, zz_rate, result, result_target, result_control = self.run_sizzel_xy_hamiltonian_tomography(
                t_start=t_start, t_stop=t_stop, sweep_points=sweep_points
            )

            self.results.append({
                'phase_diff': phase,
                'iz_rate': iz_rate,
                'zz_rate': zz_rate,
                'result': result,
                'result_target': result_target,
                'result_control': result_control,
                't_start': t_start,
                't_stop': t_stop,
                'sweep_points': sweep_points
            })

            self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

    def run_sizzel_xy_hamiltonian_tomography(self, t_start, t_stop, sweep_points=60):
        t_step = (t_stop - t_start) / sweep_points

        sizzel_xy = ConditionalStarkTuneUpRabiXY(
            qubits=self.duts,
            frequency=self.current_params['frequency'],
            amp_control=self.current_params['amp_control'],
            amp_target=self.current_params['amp_target'],
            rise=self.current_params['rise'],
            start=t_start,
            stop=t_stop,
            step=t_step,
            phase_diff=self.current_params['phase_diff'],
            iz_rate_cancel=0,
            iz_rise_drop=0,
            echo=True
        )

        result = sizzel_xy.analyze_results_with_errs()
        iz_rate = result['iz_rate']
        zz_rate = result['zz_rate']

        new_params = self.current_params.copy()
        new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2

        print(f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]} us')

        self.params_list.append(new_params)
        self.current_params = new_params

        self.result_target = sizzel_xy.result  # Store the result of the current iteration
        self.result_control = sizzel_xy.result_control  # Assuming result_control is accessible from sizzel_xy

        return iz_rate, zz_rate, result, self.result_target, self.result_control

    @register_browser_function(available_after=(run,))
    def plot_results(self, plot_size_3d=(800, 2000), fig_size=(600, 2000)):
        # Colors for the plot
        dark_navy = '#000080'
        dark_purple = '#800080'
        light_black = '#D3D3D3'

        # Prepare data for 3D plots
        phases = [res['phase_diff'] for res in self.results]
        times = [np.linspace(res['t_start'], res['t_stop'], res['sweep_points']) for res in self.results]

        # Extract the necessary data for plotting
        results_ground_x = [np.array(res['result_target'])[:, 0, 0] for res in self.results]
        results_excited_x = [np.array(res['result_target'])[:, 1, 0] for res in self.results]

        results_ground_y = [np.array(res['result_target'])[:, 0, 1] for res in self.results]
        results_excited_y = [np.array(res['result_target'])[:, 1, 1] for res in self.results]

        results_control_ground_x = [np.array(res['result_control'])[:, 0, 0] for res in self.results]
        results_control_excited_x = [np.array(res['result_control'])[:, 1, 0] for res in self.results]

        results_control_ground_y = [np.array(res['result_control'])[:, 0, 1] for res in self.results]
        results_control_excited_y = [np.array(res['result_control'])[:, 1, 1] for res in self.results]

        iz_rates = [res['iz_rate'].nominal_value for res in self.results]
        zz_rates = [res['zz_rate'].nominal_value for res in self.results]

        # Create the figure with 3D plots for x data
        fig_3d_x = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - X axis", "Control Qubits - X axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D phase vs time vs result X for ground and excited states
        for i, phase in enumerate(phases):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_ground_x[i], mode='lines',
                             name='Ground X', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_excited_x[i], mode='lines',
                             name='Excited X', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D phase vs time vs result control X for ground and excited states
        for i, phase in enumerate(phases):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_control_ground_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_control_excited_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D X figure
        fig_3d_x.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            # title_text="Conditional Stark Fine Phase Tune-Up Results - 3D X",
            scene=dict(
                xaxis=dict(title='Phase', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target X', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Phase', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control X', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the figure with 3D plots for y data
        fig_3d_y = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - Y axis", "Control Qubits - Y axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D phase vs time vs result Y for ground and excited states
        for i, phase in enumerate(phases):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_ground_y[i], mode='lines',
                             name='Ground Y', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_excited_y[i], mode='lines',
                             name='Excited Y', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D phase vs time vs result control Y for ground and excited states
        for i, phase in enumerate(phases):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_control_ground_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[phase] * len(times[i]), y=times[i], z=results_control_excited_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D Y figure
        fig_3d_y.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            # title_text="Conditional Stark Fine Phase Tune-Up Results - 3D Y",
            scene=dict(
                xaxis=dict(title='Phase', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target Y', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Phase', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control Y', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the third figure with 2D plots
        fig_2d = make_subplots(rows=1, cols=2,
                               subplot_titles=("ZZ vs Phase", "IZ vs Phase"),
                               specs=[[{'type': 'scatter'}, {'type': 'scatter'}]])

        # Plot 2D zz vs phase
        fig_2d.add_trace(
            go.Scatter(x=phases, y=zz_rates, mode='lines+markers', name='ZZ vs Phase', line=dict(color=dark_navy)),
            row=1, col=1)

        # Plot 2D iz vs phase
        fig_2d.add_trace(
            go.Scatter(x=phases, y=iz_rates, mode='lines+markers', name='IZ vs Phase', line=dict(color=dark_purple)),
            row=1, col=2)

        # Update layout for 2D figure
        fig_2d.update_layout(
            height=fig_size[0], width=fig_size[1],
            # title_text="Conditional Stark Fine Phase Tune-Up Results - 2D",
            plot_bgcolor='white',
            xaxis=dict(gridcolor=light_black),
            yaxis=dict(gridcolor=light_black)
        )

        # Customize 2D plots
        fig_2d.update_xaxes(title_text="Phase", row=1, col=1)
        fig_2d.update_yaxes(title_text="ZZ Rate (MHz)", row=1, col=1)

        fig_2d.update_xaxes(title_text="Phase", row=1, col=2)
        fig_2d.update_yaxes(title_text="IZ Rate (MHz)", row=1, col=2)

        return fig_3d_x.show(), fig_3d_y.show(), fig_2d.show()

class ConditionalStarkFineRiseTuneUp(Experiment):
    @log_and_record
    def run(self, duts, params=None, frequency=None, amp_control=None, phase_diff=0, trunc=1.0,
            rise_start=0.01, rise_stop=0.1, rise_step=0.01,
            t_start=0, t_stop=20, sweep_points=30,
            n_start=0, n_stop=32, update_iz=False, update_zz=True):
        self.duts = duts

        assert update_iz == False

        if params is None:
            amp_rabi_control = duts[0].get_c1('f01')['X'].amp
            amp_rabi_target = duts[1].get_c1('f01')['X'].amp

            area_control = amp_rabi_control * duts[0].get_c1('f01')['X'].width
            area_target = amp_rabi_target * duts[1].get_c1('f01')['X'].width

            params = {
                'iz_control': 0,
                'iz_target': 0,
                'frequency': frequency,
                'amp_control': amp_control,
                'amp_target': amp_control * area_target / area_control,
                'rise': rise_start,
                'trunc': trunc,
                'width': 0,
                'phase_diff': phase_diff,
                'zz_interaction_positive': True,
                'echo': True
            }

        self.current_params = params
        self.params_list = [params]
        self.results = []

        for rise in np.arange(rise_start, rise_stop, rise_step):
            print(f"rise: {rise}")
            self.current_params['rise'] = rise
            iz_rate, zz_rate, result, result_target, result_control = self.run_sizzel_xy_hamiltonian_tomography(
                t_start=t_start, t_stop=t_stop, sweep_points=sweep_points
            )

            self.results.append({
                'rise': rise,
                'iz_rate': iz_rate,
                'zz_rate': zz_rate,
                'result': result,
                'result_target': result_target,
                'result_control': result_control,
                't_start': t_start,
                't_stop': t_stop,
                'sweep_points': sweep_points
            })

            self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

    def run_sizzel_xy_hamiltonian_tomography(self, t_start, t_stop, sweep_points=60):
        t_step = (t_stop - t_start) / sweep_points

        sizzel_xy = ConditionalStarkTuneUpRabiXY(
            qubits=self.duts,
            frequency=self.current_params['frequency'],
            amp_control=self.current_params['amp_control'],
            amp_target=self.current_params['amp_target'],
            rise=self.current_params['rise'],
            start=t_start,
            stop=t_stop,
            step=t_step,
            phase_diff=self.current_params['phase_diff'],
            iz_rate_cancel=0,
            iz_rise_drop=0,
            echo=True
        )

        result = sizzel_xy.analyze_results_with_errs()
        iz_rate = result['iz_rate']
        zz_rate = result['zz_rate']

        new_params = self.current_params.copy()
        new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2

        print(f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]} us')

        self.params_list.append(new_params)
        self.current_params = new_params

        self.result_target = sizzel_xy.result  # Store the result of the current iteration
        self.result_control = sizzel_xy.result_control  # Assuming result_control is accessible from sizzel_xy

        return iz_rate, zz_rate, result, self.result_target, self.result_control

    @register_browser_function(available_after=(run,))
    def plot_results(self, plot_size_3d=(800, 2000), fig_size=(600, 2000)):
        # Colors for the plot
        dark_navy = '#000080'
        dark_purple = '#800080'
        light_black = '#D3D3D3'

        # Prepare data for 3D plots
        rises = [res['rise'] for res in self.results]
        times = [np.linspace(res['t_start'], res['t_stop'], res['sweep_points']) for res in self.results]

        # Extract the necessary data for plotting
        results_ground_x = [np.array(res['result_target'])[:, 0, 0] for res in self.results]
        results_excited_x = [np.array(res['result_target'])[:, 1, 0] for res in self.results]

        results_ground_y = [np.array(res['result_target'])[:, 0, 1] for res in self.results]
        results_excited_y = [np.array(res['result_target'])[:, 1, 1] for res in self.results]

        results_control_ground_x = [np.array(res['result_control'])[:, 0, 0] for res in self.results]
        results_control_excited_x = [np.array(res['result_control'])[:, 1, 0] for res in self.results]

        results_control_ground_y = [np.array(res['result_control'])[:, 0, 1] for res in self.results]
        results_control_excited_y = [np.array(res['result_control'])[:, 1, 1] for res in self.results]

        iz_rates = [res['iz_rate'].nominal_value for res in self.results]
        zz_rates = [res['zz_rate'].nominal_value for res in self.results]

        # Create the figure with 3D plots for x data
        fig_3d_x = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - X axis", "Control Qubits - X axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D rise vs time vs result X for ground and excited states
        for i, rise in enumerate(rises):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_ground_x[i], mode='lines',
                             name='Ground X', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_excited_x[i], mode='lines',
                             name='Excited X', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D rise vs time vs result control X for ground and excited states
        for i, rise in enumerate(rises):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_control_ground_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_control_excited_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D X figure
        fig_3d_x.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Rise', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target X', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Rise', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control X', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the figure with 3D plots for y data
        fig_3d_y = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - Y axis", "Control Qubits - Y axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D rise vs time vs result Y for ground and excited states
        for i, rise in enumerate(rises):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_ground_y[i], mode='lines',
                             name='Ground Y', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_excited_y[i], mode='lines',
                             name='Excited Y', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D rise vs time vs result control Y for ground and excited states
        for i, rise in enumerate(rises):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_control_ground_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[rise] * len(times[i]), y=times[i], z=results_control_excited_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D Y figure
        fig_3d_y.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Rise', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target Y', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Rise', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control Y', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the third figure with 2D plots
        fig_2d = make_subplots(rows=1, cols=2,
                               subplot_titles=("ZZ vs Rise", "IZ vs Rise"),
                               specs=[[{'type': 'scatter'}, {'type': 'scatter'}]])

        # Plot 2D zz vs rise
        fig_2d.add_trace(
            go.Scatter(x=rises, y=zz_rates, mode='lines+markers', name='ZZ vs Rise', line=dict(color=dark_navy)),
            row=1, col=1)

        # Plot 2D iz vs rise
        fig_2d.add_trace(
            go.Scatter(x=rises, y=iz_rates, mode='lines+markers', name='IZ vs Rise', line=dict(color=dark_purple)),
            row=1, col=2)

        # Update layout for 2D figure
        fig_2d.update_layout(
            height=fig_size[0], width=fig_size[1],
            plot_bgcolor='white',
            xaxis=dict(gridcolor=light_black),
            yaxis=dict(gridcolor=light_black)
        )

        # Customize 2D plots
        fig_2d.update_xaxes(title_text="Rise", row=1, col=1)
        fig_2d.update_yaxes(title_text="ZZ Rate (MHz)", row=1, col=1)

        fig_2d.update_xaxes(title_text="Rise", row=1, col=2)
        fig_2d.update_yaxes(title_text="IZ Rate (MHz)", row=1, col=2)

        return fig_3d_x.show(), fig_3d_y.show(), fig_2d.show()

class ConditionalStarkFineTruncTuneUp(Experiment):
    @log_and_record
    def run(self, duts, params=None, frequency=None, amp_control=None, phase_diff=0,
            trunc_start=0.5, trunc_stop=2.0, trunc_step=0.1,
            rise=0.01, t_start=0, t_stop=20, sweep_points=30,
            n_start=0, n_stop=32, update_iz=False, update_zz=True):
        self.duts = duts

        assert update_iz == False

        if params is None:
            amp_rabi_control = duts[0].get_c1('f01')['X'].amp
            amp_rabi_target = duts[1].get_c1('f01')['X'].amp

            area_control = amp_rabi_control * duts[0].get_c1('f01')['X'].width
            area_target = amp_rabi_target * duts[1].get_c1('f01')['X'].width

            params = {
                'iz_control': 0,
                'iz_target': 0,
                'frequency': frequency,
                'amp_control': amp_control,
                'amp_target': amp_control * area_target / area_control,
                'rise': rise,
                'trunc': trunc_start,
                'width': 0,
                'phase_diff': phase_diff,
                'zz_interaction_positive': True,
                'echo': True
            }

        self.current_params = params
        self.params_list = [params]
        self.results = []

        for trunc in np.arange(trunc_start, trunc_stop, trunc_step):
            print(f"trunc: {trunc}")
            self.current_params['trunc'] = trunc
            iz_rate, zz_rate, result, result_target, result_control = self.run_sizzel_xy_hamiltonian_tomography(
                t_start=t_start, t_stop=t_stop, sweep_points=sweep_points
            )

            self.results.append({
                'trunc': trunc,
                'iz_rate': iz_rate,
                'zz_rate': zz_rate,
                'result': result,
                'result_target': result_target,
                'result_control': result_control,
                't_start': t_start,
                't_stop': t_stop,
                'sweep_points': sweep_points
            })

            self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

    def run_sizzel_xy_hamiltonian_tomography(self, t_start, t_stop, sweep_points=60):
        t_step = (t_stop - t_start) / sweep_points

        sizzel_xy = ConditionalStarkTuneUpRabiXY(
            qubits=self.duts,
            frequency=self.current_params['frequency'],
            amp_control=self.current_params['amp_control'],
            amp_target=self.current_params['amp_target'],
            rise=self.current_params['rise'],
            start=t_start,
            stop=t_stop,
            step=t_step,
            phase_diff=self.current_params['phase_diff'],
            iz_rate_cancel=0,
            iz_rise_drop=0,
            echo=True
        )

        result = sizzel_xy.analyze_results_with_errs()
        iz_rate = result['iz_rate']
        zz_rate = result['zz_rate']

        new_params = self.current_params.copy()
        new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2

        print(f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]} us')

        self.params_list.append(new_params)
        self.current_params = new_params

        self.result_target = sizzel_xy.result  # Store the result of the current iteration
        self.result_control = sizzel_xy.result_control  # Assuming result_control is accessible from sizzel_xy

        return iz_rate, zz_rate, result, self.result_target, self.result_control

    @register_browser_function(available_after=(run,))
    def plot_results(self, plot_size_3d=(800, 2000), fig_size=(600, 2000)):
        # Colors for the plot
        dark_navy = '#000080'
        dark_purple = '#800080'
        light_black = '#D3D3D3'

        # Prepare data for 3D plots
        truncs = [res['trunc'] for res in self.results]
        times = [np.linspace(res['t_start'], res['t_stop'], res['sweep_points']) for res in self.results]

        # Extract the necessary data for plotting
        results_ground_x = [np.array(res['result_target'])[:, 0, 0] for res in self.results]
        results_excited_x = [np.array(res['result_target'])[:, 1, 0] for res in self.results]

        results_ground_y = [np.array(res['result_target'])[:, 0, 1] for res in self.results]
        results_excited_y = [np.array(res['result_target'])[:, 1, 1] for res in self.results]

        results_control_ground_x = [np.array(res['result_control'])[:, 0, 0] for res in self.results]
        results_control_excited_x = [np.array(res['result_control'])[:, 1, 0] for res in self.results]

        results_control_ground_y = [np.array(res['result_control'])[:, 0, 1] for res in self.results]
        results_control_excited_y = [np.array(res['result_control'])[:, 1, 1] for res in self.results]

        iz_rates = [res['iz_rate'].nominal_value for res in self.results]
        zz_rates = [res['zz_rate'].nominal_value for res in self.results]

        # Create the figure with 3D plots for x data
        fig_3d_x = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - X axis", "Control Qubits - X axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D trunc vs time vs result X for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_ground_x[i], mode='lines',
                             name='Ground X', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_excited_x[i], mode='lines',
                             name='Excited X', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D trunc vs time vs result control X for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_ground_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_x.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_excited_x[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D X figure
        fig_3d_x.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target X', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control X', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the figure with 3D plots for y data
        fig_3d_y = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Target Qubit - Y axis", "Control Qubits - Y axis"),
                                 specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

        # Plot 3D trunc vs time vs result Y for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_ground_y[i], mode='lines',
                             name='Ground Y', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=1)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_excited_y[i], mode='lines',
                             name='Excited Y', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=1)

        # Plot 3D trunc vs time vs result control Y for ground and excited states
        for i, trunc in enumerate(truncs):
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_ground_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_navy)),
                row=1, col=2)
            fig_3d_y.add_trace(
                go.Scatter3d(x=[trunc] * len(times[i]), y=times[i], z=results_control_excited_y[i], mode='lines',
                             name='', showlegend=(i == 0), line=dict(color=dark_purple)),
                row=1, col=2)

        # Update layout for 3D Y figure
        fig_3d_y.update_layout(
            height=plot_size_3d[0], width=plot_size_3d[1],
            scene=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Target Y', backgroundcolor='white', gridcolor=light_black)
            ),
            scene2=dict(
                xaxis=dict(title='Trunc', backgroundcolor='white', gridcolor=light_black),
                yaxis=dict(title='Time', backgroundcolor='white', gridcolor=light_black),
                zaxis=dict(title='Result Control Y', backgroundcolor='white', gridcolor=light_black)
            )
        )

        # Create the third figure with 2D plots
        fig_2d = make_subplots(rows=1, cols=2,
                               subplot_titles=("ZZ vs Trunc", "IZ vs Trunc"),
                               specs=[[{'type': 'scatter'}, {'type': 'scatter'}]])

        # Plot 2D zz vs trunc
        fig_2d.add_trace(
            go.Scatter(x=truncs, y=zz_rates, mode='lines+markers', name='ZZ vs Trunc', line=dict(color=dark_navy)),
            row=1, col=1)

        # Plot 2D iz vs trunc
        fig_2d.add_trace(
            go.Scatter(x=truncs, y=iz_rates, mode='lines+markers', name='IZ vs Trunc', line=dict(color=dark_purple)),
            row=1, col=2)

        # Update layout for 2D figure
        fig_2d.update_layout(
            height=fig_size[0], width=fig_size[1],
            plot_bgcolor='white',
            xaxis=dict(gridcolor=light_black),
            yaxis=dict(gridcolor=light_black)
        )

        # Customize 2D plots
        fig_2d.update_xaxes(title_text="Trunc", row=1, col=1)
        fig_2d.update_yaxes(title_text="ZZ Rate (MHz)", row=1, col=1)

        fig_2d.update_xaxes(title_text="Trunc", row=1, col=2)
        fig_2d.update_yaxes(title_text="IZ Rate (MHz)", row=1, col=2)

        return fig_3d_x.show(), fig_3d_y.show(), fig_2d.show()

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Class for drawing a 3D arrow.

        :param xs:
        :type xs:
        :param ys:
        :type ys:
        :param zs:
        :type zs:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

class BlochSphere:
    def __init__(self,
                 figsize=None,
                 rotation_angle=45,
                 label_fontsize=35,
                 tick_label_fontsize=20,
                 point_size=30,
                 point_alpha=1.0,
                 point_edgecolor='k',
                 vector_linewdith=3,
                 vector_arrowhead_scale=35,
                 show_background_grid=True,
                 show_background=True,
                 xy_projection=False,
                 yz_projection=False,
                 zx_projection=False,
                 show_3d_projection=False,
                 plot_2d_slice=False):
        """
        Class for plotting points and vectors on the Bloch Sphere.

        :param figsize: figure size for Bloch Sphere (default: None)
        :type figsize: tuple | None
        :param rotation_angle: angle about the z-axis to rotate the Bloch sphere for viewing
        :type rotation_angle: int
        :param label_fontsize: fontsize for x-, y-, z-labels (default: 35)
        :type label_fontsize: int
        :param tick_label_fontsize:  fontsize for x-, y-, z-ticks (default: 20)
        :type tick_label_fontsize: int
        :param point_size: point size for scatter plots
        :type point_size: int
        :param point_alpha: opacity for points in scatter plots
        :type point_alpha: float
        :param vector_linewdith: linewidth of vector in Bloch sphere
        :type vector_linewdith: int
        :param vector_arrowhead_scale: mutation scale of vector arrowhead
        :type vector_arrowhead_scale: int
        :param show_background_grid: display x, y, z grids behind Bloch sphere
        :type show_background_grid: bool
        :param show_background: display background behind Bloch sphere
        :type show_background: bool
        :param xy_projection: plot a projection of the data on the XY plane (default: False)
        :type xy_projection: bool
        :param yz_projection: plot a projection of the data on the YZ plane (default: False)
        :type yz_projection: bool
        :param zx_projection: plot a projection of the data on the zx plane (default: False)
        :type zx_projection: bool
        :param show_3d_projection: plot the projection onto a 2D plane behind the 3D Bloch sphere (default: False)
        :type show_3d_projection: bool
        :param plot_2d_slice: plot the projection as slice on a separate 2D graph (default: False)
        :type plot_2d_slice: bool

        Example of typical usage:

            b = BlochSphere(point_alpha=0.7,
                            xy_projection=True,
                            xz_projection=True,
                            yz_projection=True,
                            show_3d_projection=True,
                            plot_2d_slice=True)
            b.add_vector([x1, y1, z1], color='k', label='Vector 1')
            b.add_vector([x2, y2, z1], color='b', label='Vector 2')
            b.add_points([x_array, y_array, z_array], color='orange', label='A bunch of scatter points')
            b.show(save=True, save_pdf=True, directory='../data/Figures/', filename='Tomography_Q6_K25_')
        """

        self.figsize = figsize
        self.label_fontsize = label_fontsize
        self.tick_label_fontsize = tick_label_fontsize
        self.point_size = point_size
        self.point_alpha = point_alpha
        self.rotation_angle = rotation_angle % 360  # [0, 360)
        self.point_edgecolor = point_edgecolor
        self.vector_linewdith = vector_linewdith
        self.vector_arrowhead_scale = vector_arrowhead_scale
        self.show_background_grid = show_background_grid
        self.show_background = show_background
        self.xy_projection = xy_projection
        self.yz_projection = yz_projection
        self.zx_projection = zx_projection
        self.show_3d_projection = show_3d_projection
        self.plot_2d_slice = plot_2d_slice

        self.fig = None
        self.ax = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.zorder = 1  # enforce the order of plotting
        self.include_legend = False

        if 90 < self.rotation_angle < 270:
            self.sign_yz = -1
        else:
            self.sign_yz = 1

        if 180 < self.rotation_angle:
            self.sign_zx = -1
        else:
            self.sign_zx = 1

    def draw_bloch_sphere(self):
        """Draws an empty Bloch sphere."""
        phi = np.linspace(0, 2 * np.pi, 50)
        theta = np.linspace(0, np.pi, 50)
        PHI, THETA = np.meshgrid(phi, theta)

        x_sphere = np.sin(PHI) * np.cos(THETA)
        y_sphere = np.sin(PHI) * np.sin(THETA)
        z_sphere = np.cos(PHI)

        num_subplots = 1
        if self.plot_2d_slice is True:
            if self.xy_projection is True:
                num_subplots += 1
            if self.yz_projection is True:
                num_subplots += 1
            if self.zx_projection is True:
                num_subplots += 1

        if num_subplots < 4:
            # figsize = (num_subplots * figsize[0], figsize[1])
            # subplots[1] = num_subplots
            figsize = (num_subplots * 10, 10)
            subplots = (1, num_subplots, 1)
        elif num_subplots == 4:
            figsize = (20, 20)
            subplots = (2, 2, 1)
        rows, cols, subplot = subplots

        self.fig = plt.figure(figsize=self.figsize if self.figsize is not None else figsize)

        # Main Bloch sphere plot
        self.ax = self.fig.add_subplot(rows, cols, subplot, projection='3d')  # plt.axes(projection='3d')
        self.ax.plot_wireframe(x_sphere, y_sphere, z_sphere, rstride=1, cstride=1, color='k', alpha=0.1, lw=1)
        self.ax.plot([-1, 1], [0, 0], [0, 0], c='k', alpha=0.5)
        self.ax.plot([0, 0], [-1, 1], [0, 0], c='k', alpha=0.5)
        self.ax.plot([0, 0], [0, 0], [-1, 1], c='k', alpha=0.5)
        self.ax.plot(np.cos(phi), np.sin(phi), 0, c='k', alpha=0.5)
        self.ax.plot(np.zeros(50), np.sin(phi), np.cos(phi), c='k', alpha=0.5)
        self.ax.plot(np.sin(phi), np.zeros(50), np.cos(phi), c='k', alpha=0.5)
        self.ax.set_xlabel(r'$\langle x \rangle$', fontsize=self.label_fontsize)
        self.ax.set_ylabel(r'$\langle y \rangle$', fontsize=self.label_fontsize)
        self.ax.set_zlabel(r'$\langle z \rangle$', fontsize=self.label_fontsize)
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xticklabels(['-1', '', '', '', '', '', '', '', '1'], fontsize=self.tick_label_fontsize)
        self.ax.set_yticklabels(['-1', '', '', '', '', '', '', '', '1'], fontsize=self.tick_label_fontsize)
        self.ax.set_zticklabels(['-1', '', '', '', '', '', '', '', '1'], fontsize=self.tick_label_fontsize)
        self.ax.set_facecolor('white')
        self.ax.grid(self.show_background_grid, color='k')
        if self.show_background is False:
            self.ax.set_axis_off()
        if self.rotation_angle is not None:
            self.ax.view_init(30, self.rotation_angle)

        if self.xy_projection is True:
            subplot += 1

            if self.show_3d_projection is True:
                circle = Circle((0, 0), 1, color='grey', fill=False)
                self.ax.add_patch(circle)
                art3d.pathpatch_2d_to_3d(circle, z=-1, zdir='z')

            if self.plot_2d_slice is True:
                circle = plt.Circle((0, 0), 1, color='grey', lw=5, fill=False)
                self.ax1 = self.fig.add_subplot(rows, cols, subplot)
                self.ax1.add_artist(circle)
                self.ax1.set_xlabel(r'$\langle x \rangle$', fontsize=self.label_fontsize)
                self.ax1.set_ylabel(r'$\langle y \rangle$', fontsize=self.label_fontsize)
                self.ax1.set_xlim(-1.02, 1.02)
                self.ax1.set_ylim(-1.02, 1.02)
                self.ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
                self.ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
                self.ax1.set_xticklabels([-1, -0.5, 0, 0.5, 1], fontsize=self.tick_label_fontsize)
                self.ax1.set_yticklabels([-1, -0.5, 0, 0.5, 1], fontsize=self.tick_label_fontsize)

        if self.yz_projection is True:
            subplot += 1

            if self.show_3d_projection is True:
                circle = Circle((0, 0), 1, color='grey', fill=False)
                self.ax.add_patch(circle)
                art3d.pathpatch_2d_to_3d(circle, z=-1 * self.sign_yz, zdir='x')

            if self.plot_2d_slice is True:
                circle = plt.Circle((0, 0), 1, color='grey', lw=5, fill=False)
                self.ax3 = self.fig.add_subplot(rows, cols, subplot)
                self.ax3.add_artist(circle)
                self.ax3.set_xlabel(r'$\langle y \rangle$', fontsize=self.label_fontsize)
                self.ax3.set_ylabel(r'$\langle z \rangle$', fontsize=self.label_fontsize)
                self.ax3.set_xlim(-1.02, 1.02)
                self.ax3.set_ylim(-1.02, 1.02)
                self.ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
                self.ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
                self.ax3.set_xticklabels([-1, -0.5, 0, 0.5, 1], fontsize=self.tick_label_fontsize)
                self.ax3.set_yticklabels([-1, -0.5, 0, 0.5, 1], fontsize=self.tick_label_fontsize)

        if self.zx_projection is True:
            subplot += 1

            if self.show_3d_projection is True:
                circle = Circle((0, 0), 1, color='grey', fill=False)
                self.ax.add_patch(circle)
                art3d.pathpatch_2d_to_3d(circle, z=-1 * self.sign_zx, zdir='y')

            if self.plot_2d_slice is True:
                circle = plt.Circle((0, 0), 1, color='grey', lw=5, fill=False)
                self.ax2 = self.fig.add_subplot(rows, cols, subplot)
                self.ax2.add_artist(circle)
                self.ax2.set_xlabel(r'$\langle x \rangle$', fontsize=self.label_fontsize)
                self.ax2.set_ylabel(r'$\langle z \rangle$', fontsize=self.label_fontsize)
                self.ax2.set_xlim(-1.02, 1.02)
                self.ax2.set_ylim(-1.02, 1.02)
                self.ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
                self.ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
                self.ax2.set_xticklabels([-1, -0.5, 0, 0.5, 1], fontsize=self.tick_label_fontsize)
                self.ax2.set_yticklabels([-1, -0.5, 0, 0.5, 1], fontsize=self.tick_label_fontsize)

    def add_points(self, points, color=None, label=None, linewidth=1.5, size=None):
        """Adds points to the Bloch sphere.

        :param points: [x, y, z] coordinates for a point
            Each can be an individual list of multiple coordinates for multiple points.
        :type points: list|np.array
        :param color: color of points for scatter point (default: None)
        :type color: None|str|RGB
        :param label: label of the points for the legend (default: None)
        :type label: None|str
        :param linewidth: width of the edgecolor around the points
        :type linewidth: int|float
        :param size: size of the scatter points
        :type size: int|float
        """
        if self.fig is None:
            self.draw_bloch_sphere()

        if label is not None:
            self.include_legend = True

        x, y, z = points
        if color is None:
            sc = self.ax.scatter3D(x, y, z, alpha=self.point_alpha, edgecolor=self.point_edgecolor,
                                   lw=linewidth, label=label, s=self.point_size if size is None else size)
            color = sc.get_facecolor()
        else:
            self.ax.scatter3D(x, y, z, color=color, alpha=self.point_alpha, edgecolor=self.point_edgecolor,
                              lw=linewidth, label=label, s=self.point_size if size is None else size)

        if self.xy_projection is True:

            if self.show_3d_projection is True:
                self.ax.scatter(x, y, zs=-1, zdir='z', color=color, alpha=self.point_alpha,
                                edgecolor=self.point_edgecolor, lw=linewidth, zorder=self.zorder,
                                s=self.point_size if size is None else size)

            if self.plot_2d_slice is True:
                self.ax1.scatter(x, y, color=color, alpha=self.point_alpha,
                                 edgecolor=self.point_edgecolor, lw=linewidth,
                                 s=self.point_size if size is None else size)

        if self.yz_projection is True:

            if self.show_3d_projection is True:
                self.ax.scatter(y, z, zs=-1 * self.sign_yz, zdir='x', color=color,
                                alpha=self.point_alpha, edgecolor=self.point_edgecolor, lw=linewidth,
                                zorder=self.zorder, s=self.point_size if size is None else size)

            if self.plot_2d_slice is True:
                self.ax3.scatter(y, z, color=color, alpha=self.point_alpha,
                                 edgecolor=self.point_edgecolor, lw=linewidth,
                                 s=self.point_size if size is None else size)

        if self.zx_projection is True:

            if self.show_3d_projection is True:
                self.ax.scatter(x, z, zs=-1 * self.sign_zx, zdir='y', color=color,
                                alpha=self.point_alpha, edgecolor=self.point_edgecolor, lw=linewidth,
                                zorder=self.zorder, s=self.point_size if size is None else size)

            if self.plot_2d_slice is True:
                self.ax2.scatter(x, z, color=color, alpha=self.point_alpha,
                                 edgecolor=self.point_edgecolor, lw=linewidth,
                                 s=self.point_size if size is None else size)

        self.zorder += 1

    def add_trajectory(self, points, color=None, label=None, linestyle='-', linewidth=2.0, marker='o', ms=None):
        """Adds a trajectory to the Bloch sphere.

        :param points: [x, y, z] coordinates for a point
            Each can be an individual list of multiple coordinates for multiple points.
        :type points: list|np.array
        :param color: color of points for scatter point (default: None)
        :type color: None|str|RGB
        :param label: label of the points for the legend (default: None)
        :type label: None|str
        """
        if self.fig is None:
            self.draw_bloch_sphere()

        if label is not None:
            self.include_legend = True

        x, y, z = points
        if color is None:
            p = self.ax.plot(x, y, z, alpha=self.point_alpha, ls=linestyle, label=label, lw=linewidth, marker=marker,
                             ms=self.point_size if ms is None else ms)
            color = p[0].get_color()
        else:
            self.ax.plot(x, y, z, alpha=self.point_alpha, color=color, label=label, ls=linestyle, lw=linewidth,
                         marker=marker, ms=self.point_size if ms is None else ms)

        if self.xy_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot(x, y, zs=-1, zdir='z', alpha=self.point_alpha, color=color, ls=linestyle, lw=linewidth,
                             marker=marker, ms=self.point_size if ms is None else ms, zorder=self.zorder)

            if self.plot_2d_slice is True:
                self.ax1.plot(x, y, alpha=self.point_alpha, color=color, ls=linestyle, lw=linewidth, marker=marker,
                              ms=self.point_size if ms is None else ms)

        if self.yz_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot(y, z, zs=-1 * self.sign_yz, zdir='x', alpha=self.point_alpha, color=color, ls=linestyle,
                             lw=linewidth, marker=marker, ms=self.point_size if ms is None else ms, zorder=self.zorder)

            if self.plot_2d_slice is True:
                self.ax3.plot(y, z, alpha=self.point_alpha, color=color, ls=linestyle, lw=linewidth, marker=marker,
                              ms=self.point_size if ms is None else ms)

        if self.zx_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot(x, z, zs=-1 * self.sign_zx, zdir='y', alpha=self.point_alpha, color=color, ls=linestyle,
                             lw=linewidth, marker=marker, ms=self.point_size if ms is None else ms, zorder=self.zorder)

            if self.plot_2d_slice is True:
                self.ax2.plot(x, z, alpha=self.point_alpha, color=color, ls=linestyle, lw=linewidth, marker=marker,
                              ms=self.point_size if ms is None else ms)

        self.zorder += 1

    def add_vector(self, vector, color=None, label=None):
        """Add a vector to the Bloch sphere.

        :param vector: [x, y, z] coordinates for the tip of a vector
        :type vector: list|np.array
        :param color: color of vector (default: None)
        :type color: None|str|RGB
        :param label: label of the vector for the legend (default: None)
        :type label: None|str
        """
        if self.fig is None:
            self.draw_bloch_sphere()

        if label is not None:
            self.include_legend = True

        x, y, z = vector
        if color is None:
            p = self.ax.plot([0, x], [0, y], [0, z], lw=self.vector_linewdith, label=label)
            color = p[0].get_color()
        else:
            self.ax.plot([0, x], [0, y], [0, z], lw=self.vector_linewdith, color=color, label=label)
        a = Arrow3D([0, x], [0, y], [0, z], mutation_scale=self.vector_arrowhead_scale, arrowstyle='-|>', color=color)
        self.ax.add_artist(a)

        if self.xy_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot([0, x], [0, y], zs=-1, zdir='z', color=color, lw=self.vector_linewdith - 2,
                             zorder=self.zorder + 1)
                a = Arrow3D([0, x], [0, y], [-1, -1], mutation_scale=self.vector_arrowhead_scale - 10, arrowstyle='-|>',
                            color=color, zorder=self.zorder + 2)
                self.ax.add_artist(a)

            if self.plot_2d_slice is True:
                self.ax1.arrow(0, 0, x, y, color=color, lw=self.vector_linewdith, head_width=0.04, head_length=0.04,
                               length_includes_head=True)

        if self.yz_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot([0, y], [0, z], zs=-1 * self.sign_yz, zdir='x', color=color,
                             lw=self.vector_linewdith - 2, zorder=self.zorder + 1)
                a = Arrow3D([-1 * self.sign_yz, -1 * self.sign_yz], [0, y], [0, z],
                            mutation_scale=self.vector_arrowhead_scale - 10, arrowstyle='-|>', color=color,
                            zorder=self.zorder + 2)
                self.ax.add_artist(a)

            if self.plot_2d_slice is True:
                self.ax3.arrow(0, 0, y, z, color=color, lw=self.vector_linewdith, head_width=0.04, head_length=0.04,
                               length_includes_head=True)

        if self.zx_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot([0, x], [0, z], zs=-1 * self.sign_zx, zdir='y', color=color,
                             lw=self.vector_linewdith - 2, zorder=self.zorder + 1)
                a = Arrow3D([0, x], [-1 * self.sign_zx, -1 * self.sign_zx], [0, z],
                            mutation_scale=self.vector_arrowhead_scale - 10, arrowstyle='-|>', color=color,
                            zorder=self.zorder + 2)
                self.ax.add_artist(a)

            if self.plot_2d_slice is True:
                self.ax2.arrow(0, 0, x, z, color=color, lw=self.vector_linewdith, head_width=0.04, head_length=0.04,
                               length_includes_head=True)

        self.zorder += 1

    def show(self, save=False, save_pdf=False, save_svg=False, directory=None, filename=None):
        """Plot the Bloch Sphere in a figure.

        :param save: save the figure (default: False)
        :type save: bool
        :param save_svg: save the figure as an svg (default: False)
        :type save_svg: bool
        :param directory: directory in which the save the figure (default: None)
            If None, it will save in the current directory.
        :type directory: None|str
        :param filename: string to prepend in front for 'Bloch_sphere.png' for a filename
        :type filename: None|str
        """
        if self.fig is None:
            self.draw_bloch_sphere()

        if self.include_legend is True:
            self.ax.legend(loc=0, fontsize=20)

        # plt.tight_layout()  # Creates issues with legend
        if save is True:
            plt.savefig(f'{directory}{filename}Bloch_sphere.png', dpi=300)
            if save_pdf is True:
                plt.savefig(f'{directory}{filename}Bloch_sphere.pdf', dpi=300)
            if save_svg is True:
                plt.savefig(f'{directory}{filename}Bloch_sphere.svg', dpi=300)
        plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class ConditionalStarkSpectroscopyEchoNoFitting(experiment):
    @log_and_record
    def run(self, duts, freq_start=4100, freq_stop=4144, freq_step=1, amp_start=0, amp_stop=0.2, amp_step=0.02,
            rise=0.01, trunc=1.2,
            width=0.2):
        cs_pulses = [dut.get_c1('f01')['X'].clone() for dut in duts]

        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]

        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.set_pulse_shapes(prims.SoftSquare, amp=0, phase=0, width=width, rise=rise, trunc=trunc)

        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.update_pulse_args(shape='soft_square', amp=0, phase=0, width=width, rise=rise, trunc=trunc)

        swp_amp = sweeper(np.arange, n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
                          params=[sparam.func(cs_pulse.update_pulse_args, {}, 'amp') for cs_pulse in cs_pulses])

        swp_freq = sweeper(np.arange, n_kwargs={'start': freq_start, 'stop': freq_stop, 'step': freq_step},
                           params=[sparam.func(cs_pulse.update_freq, {}, 'freq') for cs_pulse in cs_pulses])

        c1s = [dut.get_default_c1() for dut in duts]

        flip_both = prims.ParallelLPB([c1s[0]['Y'], c1s[1]['Y']])

        lpb_zz = prims.ParallelLPB(cs_pulses) + flip_both + prims.ParallelLPB(cs_pulses) + flip_both

        lpb = c1s[0]['Xp'] + lpb_zz + c1s[0]['Ym'] + prims.ParallelLPB(mprims)

        basic(lpb, sweep=swp_amp + swp_freq, basis="<z>")

        self.result = mprims[0].result()

        pass

    @register_browser_function(available_after=(run,))
    def plot(self):
        args = self.retrieve_args(self.run)

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        fig = go.Figure(data=go.Heatmap(
            z=self.result.T, x=ys, y=xs,
            colorscale='Viridis'),
        )

        fig.update_layout(
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving amplitude",
        )

        return fig


class ConditionalStarkSpectroscopyFreqAmp(experiment):
    @log_and_record
    def run(self, duts, freq_start=4100, freq_stop=4144, freq_step=1, amp_start=0, amp_stop=0.2, amp_step=0.02,
            rise=0.01, trunc=1.2,
            t_start=0,
            t_step=0.03, t_stop=3):
        channel_from = [dut.get_c1('f01')['X'].primary_channel() for dut in duts]

        cs_pulses = [prims.LogicalPrimitive(channels=x, freq=0) for x in channel_from]

        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]

        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.set_pulse_shapes(prims.SoftSquare, amp=0, phase=0, width=0, rise=rise, trunc=trunc)

        swp_amp = sweeper(np.arange, n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
                          params=[sparam.func(cs_pulse.update_pulse_args, {}, 'amp') for cs_pulse in cs_pulses])

        swp_freq = sweeper(np.arange, n_kwargs={'start': freq_start, 'stop': freq_stop, 'step': freq_step},
                           params=[sparam.func(cs_pulse.update_freq, {}, 'freq') for cs_pulse in cs_pulses])

        swp_t = sweeper(np.arange, n_kwargs={'start': t_start / 2, 'stop': t_stop / 2, 'step': t_step / 2},
                        params=[sparam.func(cs_pulse.update_pulse_args, {}, 'width') for cs_pulse in cs_pulses])

        c1s = [dut.get_default_c1() for dut in duts]

        flip_both = prims.ParallelLPB([c1s[0]['Y'], c1s[1]['Y']])

        lpb_zz = prims.ParallelLPB(cs_pulses) + flip_both + prims.ParallelLPB(cs_pulses) + flip_both

        lpb = c1s[0]['Xp'] + lpb_zz + c1s[0]['Ym'] + prims.ParallelLPB(mprims)

        basic(lpb, sweep=swp_amp + swp_freq + swp_t, basis="<z>")

        self.result = mprims[0].result()

        # fits.Fit_1D_Freq(self.state_tomography.pauli_vector[i, :], dt=width_step)

    @register_browser_function(available_after=(run,))
    def plot(self):
        args = self.retrieve_args(self.run)

        zz_array = np.zeros_like(self.result[0, :, :])

        for freq_id in range(self.result.shape[1]):
            for amp_id in range(self.result.shape[2]):
                fit_params = fits.Fit_1D_Freq(self.result[:, freq_id, amp_id], dt=0.2)
                zz_array[freq_id, amp_id] = fit_params['Frequency'] if np.abs(fit_params['Phase']) < np.pi / 2 else - \
                    fit_params['Frequency']

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        fig = go.Figure(data=go.Heatmap(
            z=zz_array.T, x=ys, y=xs,
            colorscale='Viridis'),
        )

        fig.update_layout(
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving amplitude",
        )

        fig.show()


class ConditionalStarkTuneUpRabiYDriveSweepPhase(experiment):
    @log_and_record
    def run(self, qubits, amp, frequency=None, rise=0.05, start=0, stop=2, step=0.05, axis='Y', echo=True,
            iz_rate_cancel=0, phase_sweep_points=36):
        """
                Sweep time and find the initial guess of amplitude

                :param start: start sweeping time
                :param stop:  stop sweeping time
                :param step:  time resolution
                :return:
                """
        self.duts = qubits
        self.frequency = frequency
        self.amp = amp
        self.width = 0

        if frequency is None:
            freq_01 = qubits[1].get_c1('f01')['X'].freq
            freq_12 = qubits[1].get_c1('f12')['X'].freq

            anharmonicity = freq_01 - freq_12
            self.frequency = freq_01 - 0.3 * anharmonicity
            print(f"Choosing frequency {self.frequency}")
        else:
            self.frequency = frequency

        c1_control = self.duts[0].get_default_c1()
        c1_target = self.duts[1].get_default_c1()

        c2 = prims.build_CZ_stark_from_parameters(control_q=self.duts[0], target_q=self.duts[1],
                                                  amp_target=self.amp, amp_control=self.amp, frequency=self.frequency,
                                                  rise=rise, width=self.width)

        mprim_control = self.duts[0].get_measurement_prim_intlist(0)
        mprim_target = self.duts[1].get_measurement_prim_intlist(0)

        cs_pulse = c2['cs_pulse']

        lpb_zz = c2.get_zzp()

        lpb = lpb_zz if echo else cs_pulse

        lpb_flip_control = prims.SweepLPB([c1_control['I'], c1_control['X']])
        swp_flip = sweeper.from_sweep_lpb(lpb_flip_control)

        iz_gate = c1_target.z_omega(iz_rate_cancel)

        lpb = c1_target['Xp'] * lpb_flip_control + lpb + iz_gate + c1_target[f'{axis}p'] + mprim_target * mprim_control

        swpparams = [
            sparam.func(cs_pulse.update_pulse_args, {}, 'width'),
            sparam.func(iz_gate.set_virtual_width, {}, 'width'),
            # sparam.func(cs_pulse.update_pulse_args, {"primitive_ids": 1}, 'width')
        ]

        swp = sweeper(np.arange, n_kwargs={'start': start, 'stop': stop, 'step': step},
                      params=swpparams)

        swpparams_phase = [
            sparam.func(cs_pulse.update_pulse_args, {'primitive_ids': 1}, 'phase'),
            # sparam.func(cs_pulse.update_pulse_args, {"primitive_ids": 1}, 'width')
        ]
        swp_phase = sweeper(np.linspace, n_kwargs={'start': 0, 'stop': np.pi * 2, 'num': phase_sweep_points},
                            params=swpparams_phase)

        basic(lpb, sweep=swp + swp_flip + swp_phase, basis="<z>")

        self.result = mprim_target.result()
        self.result_control = mprim_control.result()

        self.fitting = [[
            fits.Fit_1D_Freq(self.result[j, i, :], dt=step) for j in range(self.result.shape[0])] for i in range(2)
        ]

        self.iz_rates = np.asarray([(self.fitting[0][j]['Frequency'] + self.fitting[1][j]['Frequency']) / 2 for j in
                                    range(self.result.shape[0])])
        self.zz_rates = np.asarray([(self.fitting[0][j]['Frequency'] - self.fitting[1][j]['Frequency']) / 2 for j in
                                    range(self.result.shape[0])])

        self.zz_rates_fit = fits.Fit_1D_Freq(self.zz_rates, dt=np.pi * 2 / 36)
        self.optimal_phase_diff = -self.zz_rates_fit['Phase'] + np.pi / 2

        # self.phase_contribution_from_pulse_rise_up = (self.fitting[0]['Phase'] - (self.fitting[1]['Phase'] + np.pi)) / 2

    @register_browser_function(available_after=(run,))
    def plot_2d_result(self):

        args = self.retrieve_args(self.run)

        ys = np.linspace(start=0, stop=2 * np.pi, num=args['phase_sweep_points'])
        xs = np.arange(start=args['start'], stop=args['stop'], step=args['step'])

        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, 0, :]).T, x=ys, y=xs,
            colorscale='Viridis'),
        )

        fig.update_layout(
            title='Control is at ground state',
            xaxis_title="Phase offset [rad]",
            yaxis_title="Time [us]",
        )

        fig.show()

        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, 1, :]).T, x=ys, y=xs,
            colorscale='Viridis'),
        )

        fig.update_layout(
            title='Control is at excited state',
            xaxis_title="Phase offset [rad]",
            yaxis_title="Time [us]",
        )

        fig.show()

        # fig = go.Figure(data=go.Heatmap(
        #    z=(self.result[:, 0, :] - self.result[:, 1, :]).T, x=ys, y=xs,
        #    colorscale='Viridis'),
        # )

        # fig.update_layout(
        #    xaxis_title="Phase offset [rad]",
        #    yaxis_title="Time [us]",
        # )

        # fig.show()
        # return

    @register_browser_function(available_after=(run,))
    def plot(self):

        args = self.retrieve_args(self.run)

        fig = go.Figure()

        phases = np.linspace(start=0, stop=2 * np.pi, num=args['phase_sweep_points'])

        fig.add_trace(go.Scatter(x=phases, y=self.iz_rates,
                                 mode='lines',
                                 name='IZ'))

        fig.add_trace(go.Scatter(x=phases, y=self.zz_rates,
                                 mode='lines',
                                 name='ZZ'))

        fig.update_layout(title=f'IZ and ZZ rates over phase at amp {self.amp}',
                          xaxis_title='Phase [rad]',
                          yaxis_title='Omega [MHz]', plot_bgcolor='white')

        fig.show()

        return
        args = self.retrieve_args(self.run)
        print("IZ rate", self.iz_rate)
        print("ZZ rate", self.zz_rate)
        print("Phase from edges", self.phase_contribution_from_pulse_rise_up)

        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        def plot_specific_axis(data, label, fit_params):
            plt.scatter(t, data, label=label, alpha=0.5)
            f = fit_params['Frequency']
            a = fit_params['Amplitude']
            p = fit_params['Phase'] - 2.0 * np.pi * f * args['start']
            o = fit_params['Offset']
            fit = a * np.sin(2.0 * np.pi * f * t_interpolate + p) + o
            plt.plot(t_interpolate, fit)

        plt.figure()
        plt.title(f"ZZ interaction rabi drive amp={self.amp}")
        plot_specific_axis(data=self.result[0, :], label="Ground", fit_params=self.fitting[0])
        plot_specific_axis(data=self.result[1, :], label="Excited", fit_params=self.fitting[1])
        plt.xlabel("Pulse width [us]")
        plt.ylabel("<z>")
        plt.legend()
        plt.show()


class ConditionalStarkTuneUpRabiYDriveSweepOmega(experiment):

    def update_omega(self, omega):
        cs_pulse = self.c2['cs_pulse']

        amp_control = omega * self.amp_per_omega_control
        amp_target = omega * self.amp_per_omega_target

        cs_pulse.update_pulse_args(primitive_ids=0, amp=amp_control)
        cs_pulse.update_pulse_args(primitive_ids=1, amp=amp_target)

    @log_and_record
    def run(self, qubits, omega_start=1, omega_end=20, omega_step=1, freq_offset_start=-50, freq_offset_end=-20,
            freq_step=1,
            rise=0.01,
            start=0, stop=3, step=0.03):
        """
                Sweep time and find the initial guess of amplitude

                :param start: start sweeping time
                :param stop:  stop sweeping time
                :param step:  time resolution
                :return:
                """
        self.duts = qubits
        self.width = 0

        self.amp_per_omega_control = qubits[0].get_c1('f01')['X'].get_pulse_args('amp') * \
                                     qubits[0].get_c1('f01')['X'].get_pulse_args('width') * 2

        self.amp_per_omega_target = qubits[1].get_c1('f01')['X'].get_pulse_args('amp') * \
                                    qubits[1].get_c1('f01')['X'].get_pulse_args('width') * 2

        c1_control = self.duts[0].get_default_c1()
        c1_target = self.duts[1].get_default_c1()
        freq_01 = qubits[1].get_c1('f01')['X'].freq

        frequency_start = freq_01 + freq_offset_start
        frequency_end = freq_01 + freq_offset_end

        self.c2 = prims.build_CZ_stark_from_parameters(control_q=self.duts[0], target_q=self.duts[1],
                                                       amp_target=0, amp_control=0,
                                                       frequency=self.frequency,
                                                       rise=rise, width=self.width, echo=True, phase_diff=phase_diff)

        # c2['cs_pulse'].set_pulse_shapes(shape, 0, amp=amp_control, phase=0, width=width, **shape_args)
        # c2['cs_pulse'].set_pulse_shapes(shape, 1, amp=amp_target, phase=0, width=width, **shape_args)

        mprim_control = self.duts[0].get_measurement_prim_intlist(0)
        mprim_target = self.duts[1].get_measurement_prim_intlist(0)

        cs_pulse = self.c2['cs_pulse']

        swpparams_freq = [
            sparam.func(cs_pulse.update_freq, {}, 'freq'),
        ]

        swp_freq = sweeper(np.arange,
                           n_kwargs={'start': frequency_start, 'stop': frequency_end, 'step': freq_step},
                           params=swpparams_freq)

        swpparams_amp = [
            sparam.func(self.update_omega, {}, 'omega'),
        ]

        swp_amp = sweeper(np.arange, n_kwargs={'start': omega_start, 'stop': omega_end, 'step': omega_step},
                          params=swpparams_amp)

        lpb_zz = self.c2.get_zzm_pi_over_4()

        lpb_flip_control = prims.SweepLPB([c1_control['I'], c1_control['X']])
        swp_flip = sweeper.from_sweep_lpb(lpb_flip_control)

        lpb = c1_target['Xp'] * lpb_flip_control + lpb_zz + c1_target['Yp'] + mprim_target * mprim_control

        swpparams = [
            sparam.func(cs_pulse.update_pulse_args, {}, 'width'),
            # sparam.func(cs_pulse.update_pulse_args, {"primitive_ids": 1}, 'width')
        ]

        swp = sweeper(np.arange, n_kwargs={'start': start, 'stop': stop, 'step': step},
                      params=swpparams)

        basic(lpb, sweep=swp_freq + swp_amp + swp + swp_flip, basis="<z>")

        self.result = mprim_target.result()
        self.result_control = mprim_control.result()

        # self.fitting = [
        #    fits.Fit_1D_Freq(self.result[i, :], dt=step) for i in range(2)
        # ]

    @register_browser_function(available_after=(run,))
    def plot(self):
        return
        args = self.retrieve_args(self.run)

        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        def plot_specific_axis(data, label, fit_params):
            plt.scatter(t, data, label=label, alpha=0.5)
            f = fit_params['Frequency']
            a = fit_params['Amplitude']
            p = fit_params['Phase'] - 2.0 * np.pi * f * args['start']
            o = fit_params['Offset']
            fit = a * np.sin(2.0 * np.pi * f * t_interpolate + p) + o
            plt.plot(t_interpolate, fit)

        plt.figure()
        plt.title(f"ZZ interaction rabi drive amp={self.amp}")
        plot_specific_axis(data=self.result[0, :], label="Ground", fit_params=self.fitting[0])
        plot_specific_axis(data=self.result[1, :], label="Excited", fit_params=self.fitting[1])
        plt.xlabel("Pulse width [us]")
        plt.ylabel("<z>")
        plt.legend()
        plt.show()


class ConditionalStarkTuneUp(experiment):
    @log_and_record
    def run(self, qubits, params=None, frequency=None, drive_omega=10, amp_control=None, amp_target=None,
            echo=False,
            phase_calibration=False,
            phase_calibration_amp=None,
            phase_calibration_sample_points=36,
            phase_calibration_start=0,
            phase_calibration_stop=3,
            phase_calibration_step=0.05,
            zz_calibration=True,
            zz_calibration_initial_end_time=2,
            zz_calibration_sample_points=40,
            zz_calibration_sample_period=4,
            repeated_gate_calibration=False,
            repeated_gate_calibration_ignore_zz=False,
            repeated_gate_calibration_max_gate_number=120, iz_tolerance=1e-3,
            zz_tolerance=1e-3,
            pingpong_calibration=False,
            pingpong_tolerance=1e-3,
            max_iteration=10,
            ):
        """
        :param qubits: first qubit is control, the second is target
        :param frequency: specify the frequency. Otherwise use f01_t - 0.25*alpha_t
        :return:
        """

        self.pingpong_tolerance = pingpong_tolerance
        self.max_iteration = max_iteration
        self.iz_tolerance = iz_tolerance
        self.zz_tolerance = zz_tolerance
        self.repeated_gate_calibration_ignore_zz = repeated_gate_calibration_ignore_zz

        if params is None:

            if frequency is None:
                freq_01 = qubits[1].get_c1('f01')['X'].freq
                freq_12 = qubits[1].get_c1('f12')['X'].freq

                anharmonicity = freq_01 - freq_12
                frequency = freq_01 - 0.2 * anharmonicity
                print(f"Choosing frequency {frequency}")

            if amp_control is None:
                amp_control = drive_omega * qubits[0].get_c1('f01')['X'].get_pulse_args('amp') * \
                              qubits[0].get_c1('f01')[
                                  'X'].get_pulse_args('width') * 2

                print("amp_control", amp_control)

            if amp_target is None:
                amp_target = drive_omega * qubits[1].get_c1('f01')['X'].get_pulse_args('amp') * \
                             qubits[1].get_c1('f01')[
                                 'X'].get_pulse_args('width') * 2

                print("amp_target", amp_target)

            params = {
                'width': 0,
                'amp_control': amp_control,
                'amp_target': amp_target,
                'frequency': frequency,
                'rise': 0.01,
                'iz_control': 0,
                'iz_target': 0,
                'echo': echo,
                'trunc': 1.2,
                'phase_diff': np.pi,
            }

        self.duts = qubits
        self.guessed_params = params

        if phase_calibration:
            self.calibrate_phase(points=phase_calibration_sample_points, start=phase_calibration_start,
                                 stop=phase_calibration_stop, step=phase_calibration_step)
            print(self.guessed_params)

        if zz_calibration:
            self.calibration_zz(initial_start=0, initial_stop=zz_calibration_initial_end_time,
                                sample_points=zz_calibration_sample_points,
                                sample_period=zz_calibration_sample_period, echo=echo)
            print(self.guessed_params)
        if repeated_gate_calibration:
            self.repeated_gate_calibration(max_gate_number=repeated_gate_calibration_max_gate_number)
            print(self.guessed_params)

        if pingpong_calibration:
            self.pingpong_calibration()
            print(self.guessed_params)

    def calibrate_phase(self, start=0, stop=2, step=0.05, points=36):

        self.phase_calib = ConditionalStarkTuneUpRabiYDriveSweepPhase(qubits=self.duts,
                                                                      amp=self.guessed_params['amp_target'],
                                                                      frequency=self.guessed_params['frequency'],
                                                                      rise=self.guessed_params['rise'], start=start,
                                                                      stop=stop, step=step, axis='Y', echo=False,
                                                                      iz_rate_cancel=0, phase_sweep_points=points)

        self.guessed_params['phase_diff'] = self.phase_calib.optimal_phase_diff

    def _get_max_pulse_count(self):
        pulse_total_width = (self.guessed_params['width'] + self.guessed_params['rise']) * 2

        if self.guessed_params['echo']:
            pulse_total_width += 2 * np.max([x.get_c1('f01')['X'].get_pulse_args('width') for x in self.duts])

        return int(100 / pulse_total_width / 4)

    def calibration_zz(self, initial_start=0, initial_stop=2, sample_points=100, sample_period=5, echo=True):

        iz_duts_B, zz_target = self.calibration_zz_single_qubit(flip_control_target=False,
                                                                initial_start=initial_start,
                                                                initial_stop=initial_stop,
                                                                sample_points=sample_points,
                                                                sample_period=sample_period, echo=echo)

        iz_duts_A, zz_control = self.calibration_zz_single_qubit(flip_control_target=True,
                                                                 initial_start=initial_start,
                                                                 initial_stop=initial_stop,
                                                                 sample_points=sample_points,
                                                                 sample_period=sample_period, echo=echo)

        zz = (zz_target + zz_control) / 2
        width = 1 / np.abs(zz) / 4 - self.guessed_params['rise'] * 2
        width /= 2

        # if self.guessed_params['echo']:
        #    width /= 2

        self.guessed_params['width'] = width
        self.guessed_params['iz_control'] = iz_duts_A
        self.guessed_params['iz_target'] = iz_duts_B

        print(self.guessed_params)

    def calibration_zz_single_qubit(self, initial_start=0, initial_stop=2, sample_points=100,
                                    sample_period=5, flip_control_target=False, echo=True):

        start = initial_start
        stop = initial_stop

        cancelation_iz_AB = self.guessed_params['iz_control'] if flip_control_target else self.guessed_params[
            'iz_target']

        if self.guessed_params['width'] != 0:
            period_width = self.guessed_params['width'] * 8
            stop = min(period_width * sample_period, 15)  # Set a hard limit here

        record_iz_rate = []
        record_cancel_rate = []

        if flip_control_target:
            duts = self.duts[::-1]
            amp_control = self.guessed_params['amp_target']
            amp_target = self.guessed_params['amp_control']
        else:
            duts = self.duts
            amp_control = self.guessed_params['amp_control']
            amp_target = self.guessed_params['amp_target']

        for i in range(self.max_iteration):
            print(f"Calibrating IZ shift on {duts[1].hrid} iteration {i}")
            step = (stop - start) / sample_points
            self.stark_drive_AB = ConditionalStarkTuneUpRabiXY(qubits=duts,
                                                               frequency=self.guessed_params['frequency'],
                                                               rise=self.guessed_params['rise'],
                                                               amp_control=amp_control,
                                                               amp_target=amp_target,
                                                               start=start, stop=stop, step=step, echo=echo,
                                                               iz_rate_cancel=cancelation_iz_AB
                                                               )

            record_cancel_rate.append(cancelation_iz_AB)
            record_iz_rate.append(self.stark_drive_AB.iz_rate)

            if len(record_iz_rate) > 1:
                p = np.polyfit(record_iz_rate, record_cancel_rate, deg=1)
                cancelation_iz_AB = p[-1]
            else:
                cancelation_iz_AB += self.stark_drive_AB.iz_rate

            period_width = np.abs(1 / self.stark_drive_AB.zz_rate)

            old_width = (stop - start)

            stop = min(period_width * sample_period, 15)  # Set a hard limit here

            new_scan_width = stop - start

            width_accepted = np.abs(old_width - new_scan_width) / new_scan_width < 0.2

            iz_accpted = np.abs(self.stark_drive_AB.iz_rate / self.stark_drive_AB.zz_rate) < self.iz_tolerance

            print(
                f"Calibrated IZ rate:{cancelation_iz_AB}, scan width accepted:{width_accepted}, iz accepted:{iz_accpted}")

            if iz_accpted and width_accepted:
                break
        return cancelation_iz_AB, self.stark_drive_AB.zz_rate

    def repeated_gate_calibration(self, max_gate_number=None, gate_count=40):

        if max_gate_number is None:
            max_gate_number = gate_count * 3  # self._get_max_pulse_count()

        control_params = self.repeated_gate_calibration_single_qubit(start_point=0, gate_count=gate_count,
                                                                     flip_control_target=False)
        target_params = self.repeated_gate_calibration_single_qubit(start_point=0, gate_count=gate_count,
                                                                    flip_control_target=True)

        self.guessed_params['iz_control'] = control_params['iz_control']
        self.guessed_params['iz_target'] = target_params['iz_target']

        print('max_gate_number', max_gate_number)

        for i in range(0, max_gate_number, gate_count):
            # self.guessed_params['width'] = self.guessed_params['width'] / 2

            if not self.repeated_gate_calibration_ignore_zz:
                control_params = self.repeated_gate_calibration_single_qubit(start_point=i, gate_count=gate_count,
                                                                             flip_control_target=False,
                                                                             tuneup_zz=True)
                self.guessed_params['iz_control'] = control_params['iz_control']
                self.guessed_params['width'] = control_params['width']
                target_params = self.repeated_gate_calibration_single_qubit(start_point=i, gate_count=gate_count,
                                                                            flip_control_target=True,
                                                                            tuneup_zz=True)

            self.guessed_params['width'] = (control_params['width'] + target_params['width']) / 2
            self.guessed_params['iz_control'] = control_params['iz_control']
            self.guessed_params['iz_target'] = target_params['iz_target']

    def repeated_gate_calibration_single_qubit(self, start_point, gate_count=40, flip_control_target=False,
                                               tuneup_zz=False):

        record_iz_rate = []
        record_cancel_rate = []

        record_zz_rate = []
        record_pulse_width = []

        params = copy.copy(self.guessed_params)

        if flip_control_target:
            params['amp_control'] = self.guessed_params['amp_target']
            params['amp_target'] = self.guessed_params['amp_control']
            params['iz_control'] = self.guessed_params['iz_target']
            params['iz_target'] = self.guessed_params['iz_control']

            duts = self.duts[::-1]
        else:
            duts = self.duts

        cancelation_iz_AB_rate = params['iz_target']
        pulse_zz_width = params['width']

        new_params = copy.copy(self.guessed_params)
        for i in range(self.max_iteration):

            print(f"Calibrating ZZ and IZ on {duts[1].hrid} iteration {i}")

            self.stark_drive_AB = ConditionalStarkTuneUpRepeatedGateXY(
                duts, cs_params=params, start_gate_number=start_point, gate_count=gate_count
            )

            # Calculate IZ rate

            record_cancel_rate.append(cancelation_iz_AB_rate)
            record_iz_rate.append(self.stark_drive_AB.iz_rate / pulse_zz_width)

            if len(record_iz_rate) > 1:
                p = np.polyfit(record_iz_rate[-4:], record_cancel_rate[-4:], deg=1)
                cancelation_iz_AB_rate = p[-1]
            else:
                cancelation_iz_AB_rate += self.stark_drive_AB.iz_rate / pulse_zz_width

            print(f"Calibrated IZ rate:{cancelation_iz_AB_rate}")

            # Calculate ZZ width

            if tuneup_zz:

                record_pulse_width.append(pulse_zz_width)
                record_zz_rate.append(self.stark_drive_AB.zz_rate)

                if len(record_zz_rate) > 1:
                    p = np.polyfit(record_zz_rate[-4:], record_pulse_width[-4:], deg=1)

                    if self.stark_drive_AB.zz_rate < 0:
                        rate = -0.125
                    else:
                        rate = 0.125

                    pulse_zz_width = p[1] + rate * p[0]
                else:
                    pulse_zz_width = np.abs(0.125 / self.stark_drive_AB.zz_rate) * pulse_zz_width

                if np.abs(params['width'] - pulse_zz_width) / params['width'] > 0.1:
                    pulse_zz_width = params['width'] * 1.1 if params['width'] - pulse_zz_width < 0 else params['width'] * 0.9
                    print("Ignore dramatic change")

                print(f"Calibrated ZZ rate:{self.stark_drive_AB.zz_rate}")

            params['width'] = pulse_zz_width
            params['iz_target'] = cancelation_iz_AB_rate

            tolerance_factor = (1 + gate_count / (start_point + gate_count))

            if np.abs(self.stark_drive_AB.iz_rate / self.stark_drive_AB.zz_rate) < \
                    self.iz_tolerance * tolerance_factor:
                if np.abs(np.abs(
                        self.stark_drive_AB.zz_rate) - 0.125) < self.zz_tolerance * tolerance_factor or not tuneup_zz:
                    break

            new_params = copy.copy(self.guessed_params)
            if flip_control_target:
                new_params['iz_control'] = params['iz_target']
            else:
                new_params['iz_target'] = params['iz_target']

            new_params['width'] = params['width']

            print(new_params)

        return new_params

    def pingpong_calibration(self):

        factor = 4 if self.guessed_params['echo'] else 2

        # self.pingpong_tune_up_results = []
        # self.pingpong_fit_params = []
        # self.pingpong_pulse_counts = []

        max_pulse_count = self._get_max_pulse_count()

        print('max_pulse_count', max_pulse_count)

        max_log = int(np.log2(max_pulse_count))

        for i in range(1, max_log):
            for j in range(2):
                total_count = 2 * 2 ** i
                print("Total gate count", total_count, 'iteration', j)

                step = total_count // 10
                step -= step % 2
                step = max([2, step])

                pingpong = ConditionalStarkPingPong(duts=self.duts, cs_params=self.guessed_params,
                                                    pulse_count=[x for x in range(0, total_count + 1, step)])

                error_ground = pingpong.pulse_train_ground.popts[0][0]
                error_excited = pingpong.pulse_train_excited.popts[0][0]

                error_iz = (error_ground + error_excited) / 2
                error_zz = (error_ground - error_excited) / 2

                print(f"Slope error zz {error_zz} iz {error_iz}")
                # self.pingpong_tune_up_results.append(pingpong.pulse_train.result)
                # self.pingpong_fit_params.append(pingpong.pulse_train.popts)
                # self.pingpong_pulse_counts.append(pingpong.pulse_train.pulse_count)

                correction_iz = np.arcsin(error_iz) / np.pi * factor
                correction_zz = np.arcsin(error_zz) / np.pi * factor

                # correction = self.error / np.pi * factor

                self.guessed_params['width'] = self.guessed_params['width'] * (1 - correction_zz)
                self.guessed_params['iz_control'] = self.guessed_params[
                                                        'iz_control'] - correction_iz  # * (1 - correction_iz)
                print(self.guessed_params)

                if np.abs(correction_iz * total_count) < self.pingpong_tolerance:
                    if np.abs(correction_zz * total_count) < self.pingpong_tolerance:
                        break


class ConditionalStartContinuesProcessTomography(experiment):
    @log_and_record
    def run(self, duts, frequency, dut_amps, width_start=0, width_stop=1, width_step=0.05, rise=0.01, trunc=1.2):
        channel_from = [dut.get_c1('f01')['X'].primary_channel() for dut in duts]

        cs_pulses = [prims.LogicalPrimitive(channels=x, freq=frequency) for x in channel_from]

        for cs_pulse in cs_pulses:
            cs_pulse.set_pulse_shapes(prims.SoftSquare, amp=1, phase=0, width=0, rise=rise, trunc=trunc)

        swpparams = [sparam.func(cs_pulse.update_pulse_args, {}, 'width') for cs_pulse in cs_pulses]

        swp = sweeper(np.arange, n_kwargs={'start': width_start, 'stop': width_stop, 'step': width_step},
                      params=swpparams)

        self.tomo = ProcessTomographyNQubits(duts, prims.ParallelLPB(cs_pulses), swp=swp)


class ConditionalStarkHamiltonianTomography(experiment):
    @log_and_record
    def run(self, duts, frequency, dut_amps, width_start=0, width_stop=1, width_step=0.05, rise=0.01,
            trunc=1.2, initial_lpb=None, echo=True):
        channel_from = [dut.get_c1('f01')['X'].primary_channel() for dut in duts]

        cs_pulses = [prims.LogicalPrimitive(channels=x, freq=frequency) for x in channel_from]

        c1s = [dut.get_default_c1() for dut in duts]

        flip_both = prims.ParallelLPB([c1s[0]['X'], c1s[1]['X']])

        for i, cs_pulse in enumerate(cs_pulses):
            cs_pulse.set_pulse_shapes(prims.SoftSquare, amp=dut_amps[i], phase=0, width=0, rise=rise, trunc=trunc)

        swpparams = [sparam.func(cs_pulse.update_pulse_args, {}, 'width') for cs_pulse in cs_pulses]

        if echo:
            lpb = prims.ParallelLPB(cs_pulses) + flip_both + prims.ParallelLPB(cs_pulses) + flip_both
            swp = sweeper(np.arange,
                          n_kwargs={'start': width_start / 2, 'stop': width_stop / 2, 'step': width_step / 2},
                          params=swpparams)
        else:
            lpb = prims.ParallelLPB(cs_pulses)
            swp = sweeper(np.arange,
                          n_kwargs={'start': width_start, 'stop': width_stop, 'step': width_step},
                          params=swpparams)

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        self.state_tomography = StateTomographyNQubits(duts, name='f01',
                                                       mprim_index=0,
                                                       initial_lpb=lpb,
                                                       swp=swp, extra_measurement_duts=None)

        self.pauli_vector_fitting = [
            fits.Fit_1D_Freq(self.state_tomography.pauli_vector[i, :], dt=width_step)
            for i in range(self.state_tomography.pauli_vector.shape[0])
        ]

    @register_browser_function(available_after=(run,))
    def plot(self):
        args = self.retrieve_args(self.run)

        t = np.arange(args['width_start'], args['width_stop'], args['width_step'])
        t_interpolate = np.arange(args['width_start'], args['width_stop'], args['width_step'] / 5)

        pauli_vector = self.state_tomography.pauli_vector
        qubit_number = len(args['duts'])

        labels = [""]

        for i in range(qubit_number):
            new_labels = [x + l for l in labels for x in ['I', 'X', 'Y', 'Z']]
            labels = new_labels

        def plot_specific_axis(data, label, fit_params):
            plt.scatter(t, data, label=label, alpha=0.5)
            f = fit_params['Frequency']
            a = fit_params['Amplitude']
            p = fit_params['Phase'] - 2.0 * np.pi * f * args['width_start']
            o = fit_params['Offset']
            fit = a * np.sin(2.0 * np.pi * f * t_interpolate + p) + o
            plt.plot(t_interpolate, fit)

        plt.figure()
        for i in range(pauli_vector.shape[0]):
            plot_specific_axis(pauli_vector[i, :], label=labels[i], fit_params=self.pauli_vector_fitting[i])
        plt.legend()
        plt.title("Pauli vectors")
        plt.show()

        for i in range(qubit_number):
            name_X = 'I' * i + "X" + "I" * (qubit_number - i - 1)
            name_Y = 'I' * i + "Y" + "I" * (qubit_number - i - 1)
            name_Z = 'I' * i + "Z" + "I" * (qubit_number - i - 1)
            x = pauli_vector[labels.index(name_X), :]
            y = pauli_vector[labels.index(name_Y), :]
            z = pauli_vector[labels.index(name_Z), :]

            plt.figure()
            plot_specific_axis(z, label="Z", fit_params=self.pauli_vector_fitting[labels.index(name_Z)])
            plot_specific_axis(x, label="X", fit_params=self.pauli_vector_fitting[labels.index(name_X)])
            plot_specific_axis(y, label="Y", fit_params=self.pauli_vector_fitting[labels.index(name_Y)])
            plt.legend()
            plt.show()

            plt.figure()
            vec = np.asarray([x, y, z])
            import qutip
            b = qutip.Bloch()
            b.add_vectors(vec.T)
            b.show()
        pass
