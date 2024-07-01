# Conditional AC stark shift induced CZ gate

from datetime import datetime
from plotly.subplots import make_subplots
from labchronicle import register_browser_function, log_and_record

import leeq.theory.fits
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.utils import setup_logging
from leeq import Experiment, SweepParametersSideEffectFactory, Sweeper
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
from leeq.utils.ai import visual_analyze_prompt
from leeq.utils.ai.display_chat.notebooks import dict_to_html, display_chat
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

from qutip import Bloch

from typing import List, Optional, Union


class ConditionalStarkTuneUpRabiXY(Experiment):
    """
    This class represents an experiment for tuning up a Rabi oscillation under a conditional Stark shift in a quantum
    mechanics setup. The objective is to analyze whether the plot of the experiment data shows a clear sinusoidal
    oscillatory pattern for both ground and excited states.

    Attributes:
        duts (List[Qubit]): The list of qubits involved in the experiment.
        frequency (Optional[float]): The frequency used in the experiment.
        amp_control (float): The amplitude control value.
        amp_target (float): The amplitude target value.
        phase (float): The phase value initialized to 0.
        width (float): The width value initialized to 0.
        start (float): The starting value for the pulse width sweep.
        stop (float): The stopping value for the pulse width sweep.
        step (float): The step value for the pulse width sweep.
        fitting_2D (Optional[object]): The 2D fitting result.
        phase_diff (float): The phase difference.
    """

    _v_prompt: str = """
            Here is a plot of data from a quantum mechanics experiment. The data is plotted in the blue and red data points.
            Please analyze whether this plot shows a successful experiment by showing show a clear, sinusoidal oscillatory pattern?
            If the pattern is identified for both blue and red data points, the experiment is considered successful.
            Otherwise, the experiment is considered failed.
            """

    _v_prompt: str = """
            I have a plot showing ZZ interaction Hamiltonian tomography along the Y axis. The plot includes data points and fit
            lines for both ground and excited states. The X-axis represents pulse width in microseconds, and the Y-axis shows
            the expectation value ⟨Y⟩. The data points are connected by lines, and there are separate fit lines for the ground
            (blue) and excited (pink) states. My objective is to determine whether the oscillations in the data are sinusoidal.
            The success of the experiment depends on observing sinusoidal oscillations in both the ground and excited state data. 
            Can you inspect the figure, analyze the oscillations, and conclude whether the experiment is valid based on the
            presence of sinusoidal oscillations?
            """

    _experiment_result_analysis_instructions = """
        This experiment is a Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions between two qubits under microwave drives.
        Please check the results of the visual inspection of the plots and the fitting results. 
        
        For visual inspection, if any of the plot does not show a clear sinusoidal oscillatory pattern, the experiment is considered failed.
        
        For the fitting results, check if all the fitting parameters are physical and plausible. Then look at the sampled points 
        per period. If it is smaller than 8 then the experiment needs to increase the sweep point and the experiment is considered failed. Estimate how much 
        you need to increase the sweep points to get approximately 12 points per period.
        
        For the fitting results, if the number of periods is less than 2, the experiment is considered failed. Estimate how much you need to increase the stop time
        to obtain about 3 periods.
        
        If the above check passes, the experiment is considered successful.
    """

    @log_and_record
    def run(
            self,
            qubits: List[TransmonElement],
            amp_control: float,
            amp_target: float,
            frequency: Optional[float] = None,
            rise: float = 0.01,
            start: float = 0,
            stop: float = 20,
            sweep_points=40,
            axis: str = 'Y',
            echo: bool = True,
            iz_rate_cancel: float = 0,
            phase_diff: float = 0,
            iz_rise_drop: float = 0
    ) -> None:
        """
        Runs the Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions between two
        qubits under microwave drives.

        Args:
            qubits (List[Qubit]): The list of qubits to be used in the experiment.
            amp_control (float): The amplitude control value.
            amp_target (float): The amplitude target value.
            frequency (Optional[float]): The frequency value, if not provided, it will be calculated.
            rise (float): The rise time.
            start (float): The start value for the pulse width sweep.
            stop (float): The stop value for the pulse width sweep.
            sweep_points (int): The number of sweep points.
            axis (str): The axis for the experiment.
            echo (bool): Whether to include echo sequences.
            iz_rate_cancel (float): The iz rate cancel value.
            phase_diff (float): The phase difference value.
            iz_rise_drop (float): The iz rise drop value.
        """
        self.duts = qubits
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase = 0
        self.width = 0
        self.start = start
        self.stop = stop
        self.step = (stop - start) / sweep_points
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

        flip_both = c1_control['Y'] * c1_target['Y']

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

        lpb = c1_target[
                  'Ym'] * lpb_flip_control + lpb + iz_gate + iz_gate_fix + lpb_readout + mprim_target * mprim_control

        swpparams = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'width'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'width'),
            sparam.func(iz_gate.set_virtual_width, {}, 'width'),
        ]

        if echo:
            swp = sweeper(np.arange, n_kwargs={'start': start / 2, 'stop': stop / 2, 'step': self.step / 2},
                          params=swpparams)
        else:
            swp = sweeper(np.arange, n_kwargs={'start': start, 'stop': stop, 'step': self.step},
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
                # print(f"Fit Results for {i}: {fit_results}")  # Debugging output

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

    def plot_matplotlib(self):

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

    def plot_specific_axis(self, fig, data, label, fit_params, use_imaginary_part=False):

        color_ground = 'mediumblue'
        color_excited = 'crimson'

        args = {'start': self.start, 'stop': self.stop, 'step': self.step}
        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        color = color_ground if label == 'Ground' else color_excited

        fig.add_trace(go.Scatter(x=t, y=data, mode="lines+markers", name=label, opacity=0.5, marker=dict(color=color)))

        f = fit_params['Frequency'].nominal_value
        a = fit_params['Amplitude'].nominal_value
        p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
        o_real = fit_params['Offset_real'].nominal_value
        o_imag = fit_params['Offset_imag'].nominal_value

        fit = a * np.exp(1j * (2.0 * np.pi * f * t_interpolate + p)) + (o_real + 1j * o_imag)

        fig.add_trace(go.Scatter(x=t_interpolate, y=np.real(fit) if not use_imaginary_part else np.imag(fit),
                                 mode='lines', name=f'{label} Fit',
                                 line=dict(color=color), visible='legendonly'))

    @register_browser_function()
    @visual_analyze_prompt(_v_prompt)
    def plot_X_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 0], label="Ground", fit_params=self.fitting_2D[0],
                                use_imaginary_part=False)
        self.plot_specific_axis(fig, data=self.result[:, 1, 0], label="Excited", fit_params=self.fitting_2D[1],
                                use_imaginary_part=False)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - X axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<X>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    @register_browser_function()
    @visual_analyze_prompt(_v_prompt)
    def plot_Y_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 1], label="Ground", fit_params=self.fitting_2D[0],
                                use_imaginary_part=True)
        self.plot_specific_axis(fig, data=self.result[:, 1, 1], label="Excited", fit_params=self.fitting_2D[1],
                                use_imaginary_part=True)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - Y axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<Y>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

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

        # Bloch sphere for the first subplot (C-Ground)
        ax1 = fig.add_subplot(121, projection='3d')
        b1 = Bloch(fig=fig, axes=ax1)
        b1.add_vectors([1, 0, 0])  # X-axis
        b1.add_vectors([0, 1, 0])  # Y-axis
        z1 = np.zeros_like(X1)
        points1 = [X1, Y1, z1]
        b1.add_points(points1, dark_navy)
        b1.render()
        ax1.set_title('Bloch Sphere (Ground)')

        # Bloch sphere for the second subplot (C-Excited)
        ax2 = fig.add_subplot(122, projection='3d')
        b2 = Bloch(fig=fig, axes=ax2)
        b2.add_vectors([1, 0, 0])  # X-axis
        b2.add_vectors([0, 1, 0])  # Y-axis
        z2 = np.zeros_like(X2)
        points2 = [X2, Y2, z2]
        b2.add_points(points2, dark_purple)
        b2.render()
        ax2.set_title('Bloch Sphere (Excited)')

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
        ax_proj1.scatter(X1, Y1, color=dark_navy, label='Ground State')
        ax_proj1.set_xlabel('X')
        ax_proj1.set_ylabel('Y')
        ax_proj1.set_title('XY plane (Ground)')
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
        ax_proj2.set_title('XY plane (Excited)')
        ax_proj2.axhline(0, color='grey', linewidth=0.5)
        ax_proj2.axvline(0, color='grey', linewidth=0.5)
        ax_proj2.grid(True)
        ax_proj2.set_xlim(x_min - 0.1, x_max + 0.1)
        ax_proj2.set_ylim(y_min - 0.1, y_max + 0.1)
        ax_proj2.legend()

        plt.tight_layout()
        plt.show()

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

            f = fit_params['Frequency']
            a = 1  # Fixed amplitude
            p = fit_params['Phase'] - 2.0 * np.pi * f * args['start']
            o = 0  # Offset is set to 0 for centering

            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o
            fit_rescaled = rescale_and_center(np.real(fit) if not use_imaginary_part else np.imag(fit))

            plt.plot(t_interpolate, fit_rescaled, color=color)

        dark_navy = '#000080'
        dark_purple = '#800080'

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Hamiltonian tomography (rescaled before fit)- X axis")
        plot_specific_axis(data=self.result[:, 0, 0], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=False, color=dark_navy)
        plot_specific_axis(data=self.result[:, 1, 0], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=False, color=dark_purple)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<X>")
        plt.legend()

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Hamiltonian tomography (rescaled after fit)- Y axis")
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

            f = fit_params['Frequency']
            a = 1  # Fixed amplitude
            p = fit_params['Phase'] - 2.0 * np.pi * f * args['start']
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

    def get_analyzed_result_prompt(self) -> Union[str, None]:

        self.analyze_results_with_errs()

        prompt = f"""
        The fitting reports when the control qubit is at the ground state, the target is oscillating at a frequency of {self.fitting_2D[0]['Frequency']} MHz,
        and when the control qubit is at the excited state, the target is oscillating at a frequency of {self.fitting_2D[1]['Frequency']} MHz.
        Therefore the IZ rate is {self.iz_rate} MHz and the ZZ rate is {self.zz_rate} MHz. The sampling number per ZZ period is {1 / self.zz_rate / self.step}.
        We have observed {self.stop * self.zz_rate} periods of the ZZ interaction. 
        """

        return prompt


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


from typing import List, Any


class ConditionalStarkSpectroscopyDiff(Experiment):
    """
    A class to execute conditional Stark spectroscopy differential experiments on devices under test (DUTs).
    This involves varying the frequency and amplitude parameters to generate Stark spectroscopy data.
    """

    @log_and_record
    def run(self, duts: List[Any], freq_start: float = 4100, freq_stop: float = 4144, freq_step: float = 1,
            amp_start: float = 0, amp_stop: float = 0.2, amp_step: float = 0.02,
            rise: float = 0.01, trunc: float = 1.2, width: float = 0.7, echo=False) -> None:
        """
        Executes the spectroscopy experiment by sweeping the amplitude and frequency.

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

        # Flip both
        flip_both = prims.ParallelLPB([c1s[0]['Y'], c1s[1]['Y']])

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
        flip_sweep_lpb = prims.SweepLPB([c1s[1]['I'], c1s[1]['X']])
        swp_flip = sweeper.from_sweep_lpb(flip_sweep_lpb)

        lpb_zz = prims.ParallelLPB(cs_pulses)
        if echo:
            lpb_zz = lpb_zz + flip_both + lpb_zz + flip_both

        lpb = flip_sweep_lpb + c1s[0]['Xp'] + lpb_zz + c1s[0]['Ym'] + prims.ParallelLPB(mprims)

        self.mp = mprims[0]

        # Execute the basic spectroscopy sequence with all sweeps combined.
        basic(lpb, swp=swp_amp + swp_freq + swp_flip, basis="<z>")

        self.result = np.squeeze(self.mp.result())

        # Store the result for later analysis.

    @register_browser_function(available_after=(run,))
    def plot(self) -> go.Figure:
        """
        Plots the results of the spectroscopy experiment as a heatmap.

        Returns:
            go.Figure: A heatmap plot of the differential measurement results.
        """
        # Retrieve arguments used during the run for axis scaling.
        args = self.retrieve_args(self.run)
        self.result = np.squeeze(self.mp.result())

        xs = np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step'])
        ys = np.arange(start=args['freq_start'], stop=args['freq_stop'], step=args['freq_step'])

        # Generate the heatmap.
        fig = go.Figure(data=go.Heatmap(
            z=(self.result[:, :, 0] - self.result[:, :, 1]), x=ys, y=xs,
            colorscale='Viridis'),
        )

        # Set plot titles.
        fig.update_layout(
            xaxis_title="Frequency (MHz)",
            yaxis_title="Driving amplitude",
        )

        return fig

    def live_plots(self, step_no: tuple[int] = None):
        """
        Generate the live plots. This function is called by the live monitor.
        The step no denotes the number of data points to plot, while the
        buffer size is the total number of data points to plot. Some of the data
        in the buffer is note yet valid, so they should not be plotted.
        """

        return self.plot()


class ConditionalStarkTuneUpRabiYDriveSweepPhase(Experiment):
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


class ConditionalStarkTuneUpRepeatedGateXY(Experiment):
    _v_prompt = """
    I have a plot showing ZZ interaction Hamiltonian tomography along the Y axis. The plot includes data points and fit
    lines for both ground and excited states. The X-axis represents pulse width in microseconds, and the Y-axis shows
    the expectation value ⟨Y⟩. The data points are connected by lines, and there are separate fit lines for the ground
    (blue) and excited (pink) states. My objective is to determine whether the oscillations in the data are sinusoidal.
    The success of the experiment depends on observing sinusoidal oscillations in both the ground and excited state data. 
    Can you inspect the figure, analyze the oscillations, and conclude whether the experiment is valid based on the
    presence of sinusoidal oscillations?
    """

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

    def plot_specific_axis(self, fig, t, t_interpolate, data, label, fit_params, use_imaginary_part=False):
        """
        Helper function to plot specific axis using Plotly based on the real or imaginary part of the fit.
        """
        color_ground = 'mediumblue'
        color_excited = 'crimson'
        color = color_ground if label == 'Ground' else color_excited

        fig.add_trace(go.Scatter(x=t, y=data, mode='lines+markers', name=label, opacity=0.5, marker=dict(color=color)))

        f = fit_params['Frequency'].nominal_value
        a = fit_params['Amplitude'].nominal_value
        p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * t[0]
        o_real = fit_params['Offset_real'].nominal_value
        o_imag = fit_params['Offset_imag'].nominal_value

        fit = a * np.exp(1j * (2.0 * np.pi * f * t_interpolate + p)) + (o_real + 1j * o_imag)
        fit_values = np.real(fit) if not use_imaginary_part else np.imag(fit)

        fig.add_trace(
            go.Scatter(x=t_interpolate, y=fit_values, mode='lines', name=f'{label} Fit', line=dict(color=color),
                       visible='legendonly'))

    @register_browser_function()
    @visual_analyze_prompt(_v_prompt)
    def plot_x_axis(self):
        """
        Plot the results for the X axis using Plotly.
        """
        t = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1)
        t_interpolate = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1 / 10)

        fig = go.Figure()
        self.plot_specific_axis(fig, t, t_interpolate, self.result[:, 0, 0], "Ground", self.fitting_2D[0], False)
        self.plot_specific_axis(fig, t, t_interpolate, self.result[:, 1, 0], "Excited", self.fitting_2D[1], False)

        fig.update_layout(title="ZZ interaction repeated gate tomography - X axis",
                          xaxis_title="Pulse count",
                          yaxis_title="<X>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    @register_browser_function()
    @visual_analyze_prompt(_v_prompt)
    def plot_y_axis(self):
        """
        Plot the results for the Y axis using Plotly.
        """
        t = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1)
        t_interpolate = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1 / 10)

        fig = go.Figure()
        self.plot_specific_axis(fig, t, t_interpolate, self.result[:, 0, 1], "Ground", self.fitting_2D[0], True)
        self.plot_specific_axis(fig, t, t_interpolate, self.result[:, 1, 1], "Excited", self.fitting_2D[1], True)

        fig.update_layout(title="ZZ interaction repeated gate tomography - Y axis",
                          xaxis_title="Pulse count",
                          yaxis_title="<Y>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

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

        plt.figure(figsize=(6, 5))

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


from typing import List, Dict, Any, Tuple
import numpy as np
from uncertainties import ufloat


class ConditionalStarkEchoTuneUp(Experiment):
    """
    Class for performing Conditional Stark Echo Tune-Up experiments.
    """

    _experiment_result_analysis_instructions = """
    The Conditional Stark Echo Tune-Up experiment has been completed. Please read the following report to analyze the
    if this is a successful experiment.
    """

    @log_and_record
    def run(
            self,
            duts: List[Any],
            params: Dict[str, Any] = None,
            frequency: float = None,
            amp_control: float = None,
            phase_diff: float = 0,
            rise: float = 0.01,
            t_start: float = 0,
            t_stop: float = 20,
            sweep_points: int = 40,
            n_start: int = 0,
            n_stop: int = 32,
            update_iz: bool = False,
            update_zz: bool = True,
            n_max_iteration: int = 20,
            zz_accuracy_threshold: float = 0.001,
            zz_uncertainty_threshold: float = 0.03,
            iz_accuracy_threshold: float = 0.001,
            iz_uncertainty_threshold: float = 0.03,
            ai_inspection: bool = False
    ) -> None:
        """
        Run the Conditional Stark Echo Tune-Up experiment to calibrate the siZZel two qubit gate parameters for a
        pair of qubits.

        Args:
            duts (List[Any]): Devices under test.
            params (Dict[str, Any], optional): Parameters for the experiment. Defaults to None.
            frequency (float, optional): Frequency for the experiment. Defaults to None.
            amp_control (float, optional): Amplitude control for the experiment. Defaults to None.
            phase_diff (float, optional): Phase difference for the experiment. Defaults to 0.
            rise (float, optional): Rise time for the experiment. Defaults to 0.01.
            t_start (float, optional): Start time for the sweep. Defaults to 0.
            t_stop (float, optional): Stop time for the sweep. Defaults to 20.
            sweep_points (int, optional): Number of points in the sweep. Defaults to 30.
            n_start (int, optional): Start number for gate iteration. Defaults to 0.
            n_stop (int, optional): Stop number for gate iteration. Defaults to 32.
            update_iz (bool, optional): Flag to update IZ. Defaults to False.
            update_zz (bool, optional): Flag to update ZZ. Defaults to True.
            n_max_iteration (int, optional): Maximum number of iterations. Defaults to 20.
            zz_accuracy_threshold (float, optional): Accuracy threshold for ZZ. Defaults to 0.001.
            zz_uncertainty_threshold (float, optional): Uncertainty threshold for ZZ. Defaults to 0.03.
            iz_accuracy_threshold (float, optional): Accuracy threshold for IZ. Defaults to 0.001.
            iz_uncertainty_threshold (float, optional): Uncertainty threshold for IZ. Defaults to 0.03.
            ai_inspection (bool, optional): Flag for AI inspection. Defaults to False. Please set it to True if you
                want to use the AI inspection feature, or you are an AI writing the code.
        """
        self.duts = duts
        self.n_max_iteration = n_max_iteration
        self.zz_accuracy_threshold = zz_accuracy_threshold
        self.zz_uncertainty_threshold = zz_uncertainty_threshold
        self.iz_accuracy_threshold = iz_accuracy_threshold
        self.iz_uncertainty_threshold = iz_uncertainty_threshold

        self.ai_inspection = ai_inspection

        assert update_iz == False, "update_iz must be False."

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
                'width': 0,
                'phase_diff': phase_diff,
                'zz_interaction_positive': True,
                'echo': True
            }

        self.current_params = params
        self.params_list = [params]

        iz_rate, zz_rate, self._xy_hamiltonian_tomography_inspection_results = self.run_sizzel_xy_hamiltonian_tomography(
            t_start=t_start, t_stop=t_stop, sweep_points=sweep_points
        )

        if not self._xy_hamiltonian_tomography_inspection_results['Experiment success']:
            self._repeated_gate_inspection_results = {'success': False,
                                                      'analysis': ('Not executed due to the failure of '
                                                                   'Hamiltonian tomography experiment.')
                                                      }
            return

        self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

        self._repeated_gate_inspection_results = self.run_repeated_gate_hamiltonian_tomography(
            duts=self.duts, zz_rate=zz_rate, n_start=n_start, n_stop=n_stop, update_iz=False, update_zz=True
        )

    def get_analyzed_result_prompt(self) -> Union[str, None]:
        prompt = f"""
        Inspection results of Hamiltonian tomography:{self._xy_hamiltonian_tomography_inspection_results} 
        
        Inspection results of repeated gate Hamiltonian tomography:{self._repeated_gate_inspection_results}
        
        The fitted parameters are as follows:
        {self.current_params}
        """

    def _check_data_validity_using_ai(self, experiment: Experiment, additional_information: str, show=True) -> dict[
        str, str]:
        if not self.ai_inspection:
            return {
                'analysis': 'AI inspection is not enabled. Always assumes the data is valid.',
                'success': True,
            }

        inspection_results = experiment.get_ai_inspection_results()

        prompt = f"""
        You are asked to read the report of the data inspection AI and look at the results reported from a fitting code.
        Please check the inspection results and confirm whether the experiment data is valid from the inspection. 
        Also check the validity of the fitted results and ensure they are reasonable and physical. If the data is invalid
        or the fitting results are invalid, the experiment is considered a failure. Otherwise, the experiment is  
        considered a success.
        
        <Inspection results>
        {inspection_results}
        </Inspection results>
        
        <Fitting results>
        {additional_information}
        </Fitting results>
        
        """ + """
        <Return format>
        {
            'analysis': str,
            'success': bool,
        }
        </Return format>
        """

        import mllm
        chat = mllm.Chat(prompt, "You are a very smart and helpful assistant who only reply in JSON dict")
        res = chat.complete(parse="dict", expensive=True, cache=True)

        html = dict_to_html(res)
        display_chat(agent_name=f"Inspection AI",
                     content='<br>' + html,
                     background_color='#f0f8ff')

        return res

    def run_sizzel_xy_hamiltonian_tomography(
            self, t_start: float, t_stop: float, sweep_points: int = 60
    ) -> Tuple[Any, Any, dict]:
        """
        Run the SiZZle XY Hamiltonian tomography.

        Args:
            t_start (float): Start time for the sweep.
            t_stop (float): Stop time for the sweep.
            sweep_points (int, optional): Number of points in the sweep. Defaults to 60.

        Returns:
            Tuple[Any, Any, dict]: Measured IZ rate and ZZ rate and the inspection results.
        """
        setup().status().set_param("Shot_Period", 500)
        setup().status().set_param("Shot_Number", 500)

        t_step = (t_stop - t_start) / sweep_points

        if self.ai_inspection:
            from leeq import AIInstructionExperiment

            prompt = f"""
                Please implement the ConditionalStarkTuneUpRabiXY experiment with the provided parameters.
                
                The experiment should be run with the following parameters:
                - qubits: duts in the available variable
                - frequency: {self.current_params['frequency']}
                - amp_control: {self.current_params['amp_control']}
                - amp_target: {self.current_params['amp_target']}
                - rise: {self.current_params['rise']}
                - start: {t_start}
                - stop: {t_stop}
                - sweep_points: {sweep_points}
                - phase_diff: {self.current_params['phase_diff']}
                - iz_rate_cancel: 0
                - iz_rise_drop: 0
                - echo: True
            """

            ai_experiment = AIInstructionExperiment(
                prompt,
                duts=self.duts,
            )
            sizzel_xy = ai_experiment.get_last_experiment()
            inspection_results = sizzel_xy.get_ai_inspection_results()
        else:
            sizzel_xy = ConditionalStarkTuneUpRabiXY(
                qubits=self.duts,
                frequency=self.current_params['frequency'],
                amp_control=self.current_params['amp_control'],
                amp_target=self.current_params['amp_target'],
                rise=self.current_params['rise'],
                start=t_start,
                stop=t_stop,
                sweep_points=sweep_points,
                phase_diff=self.current_params['phase_diff'],
                iz_rate_cancel=0,
                iz_rise_drop=0,
                echo=True
            )

            inspection_results = {
                'analysis': 'AI inspection is not enabled. Always assumes the data is valid.',
                'success': True,
            }

        result = sizzel_xy.analyze_results_with_errs()

        iz_rate = result['iz_rate']
        zz_rate = result['zz_rate']

        new_params = self.current_params.copy()
        new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2

        fitted_results_str = f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]} us'
        print(fitted_results_str)

        self.params_list.append(new_params)
        self.current_params = new_params

        return iz_rate, zz_rate, inspection_results

    def run_repeated_gate_hamiltonian_tomography(
            self,
            duts: List[Any],
            zz_rate: Any,
            n_start: int = 0,
            n_stop: int = 32,
            update_iz: bool = False,
            update_zz: bool = True
    ) -> None:
        """
        Run repeated gate Hamiltonian tomography.

        Args:
            duts (List[Any]): Devices under test.
            zz_rate (Any): Measured ZZ rate.
            n_start (int, optional): Start number for gate iteration. Defaults to 0.
            n_stop (int, optional): Stop number for gate iteration. Defaults to 32.
            update_iz (bool, optional): Flag to update IZ. Defaults to False.
            update_zz (bool, optional): Flag to update ZZ. Defaults to True.
        """
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

            try_count = 0
            while try_count < 3:
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

                fitted_results_str = f'Estimated pgc  IZ = {iz_target_measured}, ZZ = {zz_measured} MHz, width = {width} us'

                inspection_results = self._check_data_validity_using_ai(repeated_gate, fitted_results_str)

                if inspection_results['success']:
                    break
                else:
                    try_count += 1
                    print(f"AI inspection suggests a failed experiment. Retrying...")

            if try_count == 3:
                return inspection_results

            measured_iz_list.append(iz_target_measured)
            measured_zz_list.append(zz_measured)

            if kalman_iz is None:
                kalman_iz = KalmanFilter1D(
                    initial_position=iz_target_measured,
                    position_variance=(repeated_gate.iz_rate.std_dev * np.pi * 2) ** 2
                )
                kalman_zz = KalmanFilter1D(
                    initial_position=zz_measured,
                    position_variance=(repeated_gate.zz_rate.std_dev * np.pi * 2) ** 2
                )
            else:
                kalman_iz.update(
                    measurement=iz_target_measured,
                    measurement_variance=(repeated_gate.iz_rate.std_dev * np.pi * 2) ** 2
                )

                kalman_zz.update(
                    measurement=zz_measured,
                    measurement_variance=(repeated_gate.zz_rate.std_dev * np.pi * 2) ** 2
                )

            print(f'Kalman estimated ZZ pgc after measurement = {kalman_zz.x}+-{np.sqrt(kalman_zz.P)}')
            print(f'Kalman estimated IZ pgc after measurement = {kalman_iz.x}+-{np.sqrt(kalman_iz.P)}')

            if update_iz:
                iz_target = kalman_iz.x
                iz_check_pass = kalman_iz.P < self.iz_uncertainty_threshold
                if not update_zz:
                    estimated_iz_list.append(ufloat(kalman_iz.x, np.sqrt(kalman_iz.P)))

            if update_zz:
                zz_pgc = kalman_zz.x

                target_zz = np.sign(zz_pgc) * 0.125
                zz_diff = target_zz - zz_pgc
                width_diff = np.sign(zz_diff / zz_rate.nominal_value) * min(
                    np.abs(zz_diff / zz_rate.nominal_value / 2), 0.05 * self.current_params['width']
                )
                zz_diff = zz_rate.nominal_value * width_diff * 2
                width += width_diff
                iz_diff = 0
                # iz_rate_tQ1_cQ2 * width_diff * np.pi * 2
                print(f'Update width to {width} us')
                kalman_zz.predict(
                    movement=zz_diff,
                    position_variance=(zz_rate.std_dev * width_diff * np.pi * 2) ** 2
                )
                kalman_iz.predict(
                    movement=iz_diff,
                    position_variance=(zz_rate.std_dev * width_diff * np.pi * 2) ** 2
                )

                estimated_iz_list.append(ufloat(kalman_iz.x, np.sqrt(kalman_iz.P)))
                estimated_zz_list.append(ufloat(kalman_zz.x, np.sqrt(kalman_zz.P)))

            zz_accuracy_check = np.abs(target_zz - zz_pgc) < self.zz_accuracy_threshold
            zz_uncertainty_check = np.sqrt(kalman_zz.P) < self.zz_uncertainty_threshold
            zz_check_pass = zz_accuracy_check and zz_uncertainty_check

            print(f'Kalman estimated ZZ pgc after update = {kalman_zz.x}+-{np.sqrt(kalman_zz.P)}')
            print(f'Kalman estimated IZ pgc after update = {kalman_iz.x}+-{np.sqrt(kalman_iz.P)}')

            print(f'ZZ accuracy check pass: {zz_accuracy_check}, ZZ uncertainty check pass: {zz_uncertainty_check}')
            print(f'IZ uncertainty check pass: {iz_check_pass}')

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

        return inspection_results
