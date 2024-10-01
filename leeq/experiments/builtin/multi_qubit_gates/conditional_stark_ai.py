# Conditional AC stark shift induced CZ gate
import inspect

from mllm import Chat

from k_agents.execution.agent import execute_experiment_from_prompt
from k_agents.execution.stage_execution import get_exp_from_var_table
from k_agents.inspection.decorator import text_inspection, visual_inspection
from k_agents.utils import Singleton
from leeq.utils import setup_logging
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
from k_agents.notebook_utils import dict_to_html, display_chat
from leeq.theory import fits
from leeq.theory.fits.fit_exp import fit_2d_freq_with_cov

from leeq.theory.estimator.kalman import KalmanFilter1D
from leeq.utils.high_level_simulations.noise import apply_noise_to_data

logger = setup_logging(__name__)

import plotly.graph_objects as go
from labchronicle import log_and_record, register_browser_function
from leeq import Experiment
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.utils.compatibility import *

import matplotlib.pyplot as plt
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial
from leeq.utils.compatibility import prims
from typing import Optional, Union, Type
from typing import List, Dict, Any, Tuple
import numpy as np
from uncertainties import ufloat


def _qubit_z_expectation_value_off_resonance_drive(f_qubit, f_drive, t_start, t_stop,
                                                   t_step, drive_rate):
    # Convert frequencies from MHz to Hz
    f_qubit = f_qubit * 1e6
    f_drive = f_drive * 1e6
    Omega_R = drive_rate * 1e6

    # Time array in seconds
    t = np.arange(t_start, t_stop, t_step) * 1e-6

    # Calculate detuning
    Delta = 2 * np.pi * (f_drive - f_qubit)

    # Effective Rabi frequency
    Omega_eff = np.sqrt(Omega_R ** 2 + Delta ** 2)

    # Population in the excited state as a function of time
    P_excited = (Omega_R ** 2 / Omega_eff ** 2) * np.sin(Omega_eff * t / 2) ** 2

    # Z expectation value (difference between ground and excited state populations)
    Z_expectation = 1 - 2 * P_excited

    return Z_expectation


class ConditionalStarkShiftContinuousPhaseSweep(Experiment):
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

    #
    # _experiment_result_analysis_instructions = """
    #     This experiment is a Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions between two qubits under microwave drives.
    #     Please check the results of the visual inspection of the plots and the fitting results.
    #
    #     For visual inspection, if any of the plot does not show a clear sinusoidal oscillatory pattern, the experiment is considered failed.
    #
    #     For the fitting results, check if all the fitting parameters are physical and plausible. Then look at the sampled points
    #     per period. If it is smaller than 6 then the experiment needs to increase the sweep point and the experiment is considered failed. Estimate how much
    #     you need to increase the sweep points to get approximately 8 points per period. Note that the maximum total sweep point should be 100.
    #
    #     For the fitting results, if the number of periods is less than 2, the experiment is considered failed. Estimate how much you need to increase the stop time
    #     to obtain about 3 periods.
    #
    #     If the above check passes, the experiment is considered successful.
    # """

    @log_and_record
    def run(
            self,
            qubits: List[TransmonElement],
            amp_control: float,
            amp_target: float,
            frequency: Optional[float] = None,
            rise: float = 0.015,
            start: float = 0,
            stop: float = 15,
            sweep_points=30,
            axis: str = 'Y',
            echo: bool = True,
            iz_rate_cancel: float = 0,
            iz_rise_drop: float = 0,
            phase_sweep_points: int = 10,
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
            iz_rise_drop (float): The iz rise drop value.
            phase_sweep_points (int): The number of phase sweep points.
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
        self.phase_sweep_points = phase_sweep_points

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

        c2 = prims.build_CZ_stark_from_parameters(control_q=self.duts[0],
                                                  target_q=self.duts[1],
                                                  amp_target=self.amp_target,
                                                  amp_control=self.amp_control,
                                                  frequency=self.frequency, rise=rise,
                                                  width=self.width,
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

        swp_params = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'phase'),
        ]

        swp_phase = sweeper(np.linspace, n_kwargs={'start': 0, 'stop': np.pi * 2,
                                                   'num': self.phase_sweep_points},
                            params=swp_params)

        swpparams = [
            sparam.func(stark_drive_target_pulse.update_pulse_args, {}, 'width'),
            sparam.func(stark_drive_control_pulse.update_pulse_args, {}, 'width'),
            sparam.func(iz_gate.set_virtual_width, {}, 'width'),
        ]

        if echo:
            swp = sweeper(np.arange, n_kwargs={'start': start / 2, 'stop': stop / 2,
                                               'step': self.step / 2},
                          params=swpparams)
        else:
            swp = sweeper(np.arange,
                          n_kwargs={'start': start, 'stop': stop, 'step': self.step},
                          params=swpparams)

        basic(lpb, swp=swp_phase + swp + swp_flip + swp_readout, basis="<z>")

        self.result = np.squeeze(mprim_target.result())
        self.result_control = np.squeeze(mprim_control.result())

    def analyze_results(self):
        if self.fitting_2D is None:

            self.fitting_2D = []
            for i in range(2):
                self.real_part = self.result[:, i, 0]
                self.imag_part = self.result[:, i, 1]
                self.complex_data = self.real_part + 1j * self.imag_part

                self.fit_result = fits.fit_2d_freq(self.complex_data, dt=self.step,
                                                   use_freq_bound=False)
                self.fitting_2D.append(self.fit_result)

            self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1][
                'Frequency']) / 2
            self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1][
                'Frequency']) / 2

            self.iz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] + (
            self.fitting_2D[1]['Phase'])) / 2
            self.zz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] - (
            self.fitting_2D[1]['Phase'])) / 2

            print(f"IZ: {self.iz_rate: 0.5f} MHz, ZZ: {self.zz_rate: 0.5f} MHz")
            print(
                f"Phase IZ Contributions from Pulse Rise Drop: {self.iz_from_pulse_rise_drop: 0.5f} rad")
            print(
                f"Phase ZZ Contributions from Pulse Rise Drop: {self.zz_from_pulse_rise_drop: 0.5f} rad")

        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
            'iz_from_pulse_rise_drop': self.iz_from_pulse_rise_drop,
            'zz_from_pulse_rise_drop': self.zz_from_pulse_rise_drop
        }

    def analyze_results_with_errs(self):
        if self.fitting_2D is None:
            self.fitting_2D = []
            for i in range(2):
                self.real_part = self.result[:, i, 0]
                self.imag_part = self.result[:, i, 1]
                self.complex_data = self.real_part + 1j * self.imag_part

                fit_results = fits.fit_2d_freq_with_cov(self.complex_data, dt=self.step,
                                                        use_freq_bound=False)
                self.fitting_2D.append(fit_results)

            # Calculate iz_rate and zz_rate for Curve Fit
            self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1][
                'Frequency']) / 2
            self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1][
                'Frequency']) / 2
            self.iz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] + (
            self.fitting_2D[1]['Phase'])) / 2
            self.zz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] - (
            self.fitting_2D[1]['Phase'])) / 2

            print(f"IZ: {self.iz_rate: 0.5f} MHz: {self.zz_rate: 0.5f} MHz")
            print(
                f"Phase IZ Contributions from Pulse Rise Drop: {self.iz_from_pulse_rise_drop: 0.5f} rad")

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

            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
            o = fit_params['Offset_real'].nominal_value + 1j * fit_params[
                'Offset_imag'].nominal_value

            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o

            plt.plot(t_interpolate,
                     np.real(fit) if not use_imaginary_part else np.imag(fit))

        plt.figure(figsize=(20, 5))
        plt.title(f"ZZ interaction Hamiltonian tomography - X axis")
        plot_specific_axis(data=self.result[:, 0, 0], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=False)

        plot_specific_axis(data=self.result[:, 1, 0], label="Excited",
                           fit_params=self.fitting_2D[1],
                           use_imaginary_part=False)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<X>")
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.title(f"ZZ interaction Hamiltonian tomography - Y axis")
        plot_specific_axis(data=self.result[:, 0, 1], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=True)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited",
                           fit_params=self.fitting_2D[1],
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

        fig.add_trace(
            go.Scatter(x=t, y=data, mode="lines+markers", name=label, opacity=0.5,
                       marker=dict(color=color)))

        f = fit_params['Frequency'].nominal_value
        a = fit_params['Amplitude'].nominal_value
        p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
        o_real = fit_params['Offset_real'].nominal_value
        o_imag = fit_params['Offset_imag'].nominal_value

        fit = a * np.exp(1j * (2.0 * np.pi * f * t_interpolate + p)) + (
                    o_real + 1j * o_imag)

        fig.add_trace(go.Scatter(x=t_interpolate,
                                 y=np.real(fit) if not use_imaginary_part else np.imag(
                                     fit),
                                 mode='lines', name=f'{label} Fit',
                                 line=dict(color=color), visible='legendonly'))

    @register_browser_function()
    @visual_inspection(_v_prompt)
    def plot_X_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 0], label="Ground",
                                fit_params=self.fitting_2D[0],
                                use_imaginary_part=False)
        self.plot_specific_axis(fig, data=self.result[:, 1, 0], label="Excited",
                                fit_params=self.fitting_2D[1],
                                use_imaginary_part=False)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - X axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<X>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    @register_browser_function()
    @visual_inspection(_v_prompt)
    def plot_Y_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 1], label="Ground",
                                fit_params=self.fitting_2D[0],
                                use_imaginary_part=True)
        self.plot_specific_axis(fig, data=self.result[:, 1, 1], label="Excited",
                                fit_params=self.fitting_2D[1],
                                use_imaginary_part=True)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - Y axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<Y>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    @text_inspection
    def fitting(self) -> Union[str, None]:

        self.analyze_results_with_errs()

        prompt = f"""
        The fitting reports when the control qubit is at the ground state, the target is oscillating at a frequency of {self.fitting_2D[0]['Frequency']} MHz,
        and when the control qubit is at the excited state, the target is oscillating at a frequency of {self.fitting_2D[1]['Frequency']} MHz.
        Therefore the IZ rate is {self.iz_rate} MHz and the ZZ rate is {np.abs(self.zz_rate)} MHz. The sampling number per ZZ period is {np.abs(1 / self.zz_rate / self.step)}.
        We have observed {np.abs(self.stop * self.zz_rate)} periods of the ZZ interaction.
        Note that the sign of the ZZ rate is corresponds to the direction of the rotation and it is allowed to be negative.
        """

        return prompt


def _generate_zz_interaction_data_from_simulation(qubits,
                                                  start,
                                                  stop,
                                                  sweep_points,
                                                  zz_value,
                                                  drive_frequency,
                                                  amp_control
                                                  ):
    simulator_setup = setup().get_default_setup()
    virtual_transmon_1 = simulator_setup.get_virtual_qubit(qubits[0])
    virtual_transmon_2 = simulator_setup.get_virtual_qubit(qubits[1])

    duts = qubits

    c1_control = duts[0].get_default_c1()
    c1_target = duts[1].get_default_c1()

    # Evaluate the ZZ value

    coupling = simulator_setup.get_coupling_strength_by_qubit(virtual_transmon_1,
                                                              virtual_transmon_2)
    drive_omega = simulator_setup.get_omega_per_amp(
        channel=c1_control.channel) * amp_control

    delta = drive_frequency - virtual_transmon_1.qubit_frequency
    freq_delta = virtual_transmon_2.qubit_frequency - virtual_transmon_1.qubit_frequency

    step = (stop - start) / sweep_points
    zz_oscillation_t = np.arange(start, stop, step)

    zz_oscillation_ground_x = np.cos(2 * np.pi * zz_oscillation_t * zz_value)
    zz_oscillation_excited_x = np.cos(2 * np.pi * zz_oscillation_t * -zz_value)

    zz_oscillation_ground_y = np.sin(2 * np.pi * zz_oscillation_t * zz_value)
    zz_oscillation_excited_y = np.sin(2 * np.pi * zz_oscillation_t * -zz_value)

    # Evaluate the oscillation of the control qubit
    control_qubit_oscillation = _qubit_z_expectation_value_off_resonance_drive(
        f_qubit=virtual_transmon_1.qubit_frequency,
        f_drive=drive_frequency,
        t_start=start,
        t_stop=stop,
        t_step=(stop - start) / sweep_points,
        drive_rate=drive_omega
    )

    # Evaluate the oscillation of the target qubit
    target_qubit_oscillation = _qubit_z_expectation_value_off_resonance_drive(
        f_qubit=virtual_transmon_2.qubit_frequency,
        f_drive=drive_frequency,
        t_start=start,
        t_stop=stop,
        t_step=(stop - start) / sweep_points,
        drive_rate=drive_omega
    )

    zz_oscillation_ground_x *= np.abs(control_qubit_oscillation) * np.abs(
        target_qubit_oscillation)
    zz_oscillation_excited_x *= np.abs(control_qubit_oscillation) * np.abs(
        target_qubit_oscillation)
    zz_oscillation_ground_y *= np.abs(control_qubit_oscillation) * np.abs(
        target_qubit_oscillation)
    zz_oscillation_excited_y *= np.abs(control_qubit_oscillation) * np.abs(
        target_qubit_oscillation)

    #
    # Add noise to the data
    #

    zz_oscillation_ground_x = apply_noise_to_data(virtual_transmon_2,
                                                  zz_oscillation_ground_x)
    zz_oscillation_excited_x = apply_noise_to_data(virtual_transmon_2,
                                                   zz_oscillation_excited_x)
    zz_oscillation_ground_y = apply_noise_to_data(virtual_transmon_2,
                                                  zz_oscillation_ground_y)
    zz_oscillation_excited_y = apply_noise_to_data(virtual_transmon_2,
                                                   zz_oscillation_excited_y)

    control_qubit_oscillation_ground = apply_noise_to_data(virtual_transmon_1,
                                                           -control_qubit_oscillation)
    control_qubit_oscillation_excited = apply_noise_to_data(virtual_transmon_1,
                                                            control_qubit_oscillation)

    target_qubit_oscillation = apply_noise_to_data(virtual_transmon_2,
                                                   target_qubit_oscillation)


    result = np.array([[zz_oscillation_ground_x, zz_oscillation_excited_x],
                       [zz_oscillation_ground_y, zz_oscillation_excited_y]]).transpose(
        [2, 1, 0])

    result = np.zeros_like(result)
    # Time index, ground/excited state, X/Y axis

    result[:, 0, 0] = zz_oscillation_ground_x
    result[:, 1, 0] = zz_oscillation_excited_x
    result[:, 0, 1] = zz_oscillation_ground_y
    result[:, 1, 1] = zz_oscillation_excited_y

    result_control = np.array([
        [control_qubit_oscillation_ground,
         control_qubit_oscillation_excited],
        [control_qubit_oscillation_ground,
         control_qubit_oscillation_excited]
    ]).transpose([2, 1, 0])

    result_control = np.zeros_like(result_control)
    result_control[:, 0, 0] = control_qubit_oscillation_ground
    result_control[:, 1, 0] = control_qubit_oscillation_excited
    result_control[:, 0, 1] = control_qubit_oscillation_ground
    result_control[:, 1, 1] = control_qubit_oscillation_excited

    return result, result_control


class ConditionalStarkShiftContinuous(Experiment):
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

If the any of above check fails, the experiment is considered failed.
"""

    @log_and_record(overwrite_func_name='ConditionalStarkShiftContinuous.run')
    def run_simulated(
            self,
            duts: List[TransmonElement],
            amp_control: float,
            amp_target: float,
            frequency: Optional[float] = None,
            rise: float = 0.015,
            start: float = 0,
            stop: float = 15,
            sweep_points=30,
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

        qubits = duts
        self.duts = qubits
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase = 0
        self.width = 0
        self.start = start
        self.stop = stop
        self.step = (stop - start) / sweep_points
        self.rise = rise
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

        simulator_setup = setup().get_default_setup()
        virtual_transmon_1 = simulator_setup.get_virtual_qubit(qubits[0])
        virtual_transmon_2 = simulator_setup.get_virtual_qubit(qubits[1])

        # Evaluate the ZZ value

        from leeq.theory.sizzel_gate.sizzel_simulation import ret_zz

        coupling = simulator_setup.get_coupling_strength_by_qubit(virtual_transmon_1,
                                                                  virtual_transmon_2)
        drive_omega = simulator_setup.get_omega_per_amp(
            channel=c1_control.channel) * amp_control

        delta = self.frequency - virtual_transmon_1.qubit_frequency
        freq_delta = virtual_transmon_2.qubit_frequency - virtual_transmon_1.qubit_frequency

        zz_value = ret_zz(
            alpha1=virtual_transmon_1.anharmonicity,
            alpha2=virtual_transmon_2.anharmonicity,
            J=coupling,
            eps=drive_omega,
            delta=delta,
            freq_delta=freq_delta
        )[0]  # Delta 11 is the ZZ value

        self.result, self.result_control = (
            _generate_zz_interaction_data_from_simulation(qubits=qubits,
                                                          start=start,
                                                          stop=stop,
                                                          sweep_points=sweep_points,
                                                          zz_value=zz_value,
                                                          drive_frequency=self.frequency,
                                                          amp_control=amp_control
                                                          ))

    @log_and_record
    def run(
            self,
            duts: List[TransmonElement],
            amp_control: float,
            amp_target: float,
            frequency: Optional[float] = None,
            rise: float = 0.015,
            start: float = 0,
            stop: float = 15,
            sweep_points=30,
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
        qubits = duts
        self.duts = qubits
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase = 0
        self.width = 0
        self.start = start
        self.stop = stop
        self.step = (stop - start) / sweep_points
        self.rise = rise
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

        c2 = prims.build_CZ_stark_from_parameters(control_q=self.duts[0],
                                                  target_q=self.duts[1],
                                                  amp_target=self.amp_target,
                                                  amp_control=self.amp_control,
                                                  frequency=self.frequency, rise=rise,
                                                  width=self.width,
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
            swp = sweeper(np.arange, n_kwargs={'start': start / 2, 'stop': stop / 2,
                                               'step': self.step / 2},
                          params=swpparams)
        else:
            swp = sweeper(np.arange,
                          n_kwargs={'start': start, 'stop': stop, 'step': self.step},
                          params=swpparams)

        basic(lpb, swp=swp + swp_flip + swp_readout, basis="<z>")

        self.result = np.squeeze(mprim_target.result())
        self.result_control = np.squeeze(mprim_control.result())

    def analyze_results(self):

        if self.fitting_2D is None:

            self.fitting_2D = []
            for i in range(2):
                self.real_part = self.result[:, i, 0]
                self.imag_part = self.result[:, i, 1]
                self.complex_data = self.real_part + 1j * self.imag_part

                self.fit_result = fits.fit_2d_freq(self.complex_data, dt=self.step,
                                                   use_freq_bound=False)
                self.fitting_2D.append(self.fit_result)

            self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1][
                'Frequency']) / 2
            self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1][
                'Frequency']) / 2

            self.iz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] + (
            self.fitting_2D[1]['Phase'])) / 2
            self.zz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] - (
            self.fitting_2D[1]['Phase'])) / 2

            print(f"IZ: {self.iz_rate: 0.5f} MHz, ZZ: {self.zz_rate: 0.5f} MHz")
            print(
                f"Phase IZ Contributions from Pulse Rise Drop: {self.iz_from_pulse_rise_drop: 0.5f} rad")
            print(
                f"Phase ZZ Contributions from Pulse Rise Drop: {self.zz_from_pulse_rise_drop: 0.5f} rad")

        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
            'iz_from_pulse_rise_drop': self.iz_from_pulse_rise_drop,
            'zz_from_pulse_rise_drop': self.zz_from_pulse_rise_drop
        }

    def analyze_results_with_errs(self):

        if self.fitting_2D is None:
            self.fitting_2D = []
            for i in range(2):
                self.real_part = self.result[:, i, 0]
                self.imag_part = self.result[:, i, 1]
                self.complex_data = self.real_part + 1j * self.imag_part

                fit_results = fits.fit_2d_freq_with_cov(self.complex_data, dt=self.step,
                                                        use_freq_bound=False)
                self.fitting_2D.append(fit_results)

            # Calculate iz_rate and zz_rate for Curve Fit
            self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1][
                'Frequency']) / 2
            self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1][
                'Frequency']) / 2
            self.iz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] + (
            self.fitting_2D[1]['Phase'])) / 2
            self.zz_from_pulse_rise_drop = (self.fitting_2D[0]['Phase'] - (
            self.fitting_2D[1]['Phase'])) / 2

            print(f"IZ: {self.iz_rate: 0.5f} MHz: {self.zz_rate: 0.5f} MHz")
            print(
                f"Phase IZ Contributions from Pulse Rise Drop: {self.iz_from_pulse_rise_drop: 0.5f} rad")

        if len(self.fitting_2D) != 2:
            return {'error': 'Fitting failed'}

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

            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
            o = fit_params['Offset_real'].nominal_value + 1j * fit_params[
                'Offset_imag'].nominal_value

            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o

            plt.plot(t_interpolate,
                     np.real(fit) if not use_imaginary_part else np.imag(fit))

        plt.figure(figsize=(20, 5))
        plt.title(f"ZZ interaction Hamiltonian tomography - X axis")
        plot_specific_axis(data=self.result[:, 0, 0], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=False)

        plot_specific_axis(data=self.result[:, 1, 0], label="Excited",
                           fit_params=self.fitting_2D[1],
                           use_imaginary_part=False)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<X>")
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.title(f"ZZ interaction Hamiltonian tomography - Y axis")
        plot_specific_axis(data=self.result[:, 0, 1], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=True)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited",
                           fit_params=self.fitting_2D[1],
                           use_imaginary_part=True)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<Y>")
        plt.legend()
        plt.show()

    def plot_specific_axis(self, fig, data, label, fit_params=None,
                           use_imaginary_part=False):

        color_ground = 'mediumblue'
        color_excited = 'crimson'

        args = {'start': self.start, 'stop': self.stop, 'step': self.step}
        t = np.arange(args['start'], args['stop'], args['step'])
        t_interpolate = np.arange(args['start'], args['stop'], args['step'] / 5)

        color = color_ground if label == 'Ground' else color_excited

        fig.add_trace(
            go.Scatter(x=t, y=data, mode="lines+markers", name=label, opacity=0.5,
                       marker=dict(color=color)))

        if fit_params is not None:
            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args['start']
            o_real = fit_params['Offset_real'].nominal_value
            o_imag = fit_params['Offset_imag'].nominal_value

            fit = a * np.exp(1j * (2.0 * np.pi * f * t_interpolate + p)) + (
                        o_real + 1j * o_imag)

            fig.add_trace(go.Scatter(x=t_interpolate, y=np.real(
                fit) if not use_imaginary_part else np.imag(fit),
                                     mode='lines', name=f'{label} Fit',
                                     line=dict(color=color), visible='legendonly'))

    def get_ai_inspection_results(self):
        """
        Returns the results of the AI inspection for the experiment.
        """

        inspection_results = super().get_ai_inspection_summary()

        if self.fitting_2D is None:
            try:
                self.analyze_results_with_errs()
            except Exception as e:
                inspection_results['error'] = str(e)

        try:
            zz_rate = self.zz_rate

            inspection_results['Calibrated parameters'] = {
                'amp_control': self.amp_control,
                'amp_target': self.amp_target,
                'frequency': self.frequency,
                'rise': self.rise,
                'phase_diff': self.phase_diff,
                'width': np.abs(0.125 / zz_rate.nominal_value) / 2,
                'zz_interaction_positive': self.zz_rate > 0,
                'zz_rate': self.zz_rate.n
            }
        except:
            pass

        return inspection_results

    @register_browser_function()
    def plot_X_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 0], label="Ground",
                                fit_params=self.fitting_2D[0],
                                use_imaginary_part=False)
        self.plot_specific_axis(fig, data=self.result[:, 1, 0], label="Excited",
                                fit_params=self.fitting_2D[1],
                                use_imaginary_part=False)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - X axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<X>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    @register_browser_function()
    def plot_Y_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 1], label="Ground",
                                fit_params=self.fitting_2D[0],
                                use_imaginary_part=True)
        self.plot_specific_axis(fig, data=self.result[:, 1, 1], label="Excited",
                                fit_params=self.fitting_2D[1],
                                use_imaginary_part=True)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - Y axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<Y>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    def plot_Z_axis(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result[:, 0, 2], label="Ground",
                                use_imaginary_part=True)
        self.plot_specific_axis(fig, data=self.result[:, 1, 2], label="Excited",
                                use_imaginary_part=True)

        fig.update_layout(title="ZZ interaction Hamiltonian tomography - Z axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<Z>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    @register_browser_function()
    @visual_inspection("""
I have a plot showing ZZ interaction Hamiltonian tomography in the Fourier space. The X-axis represents 
the frequency, and the Y-axis shows the amplitude of the fourier transformed value.
My objective is to determine if the experiment is a success.
The success of the experiment depends on observing two clear peaks in the Fourier space, 
one for the ground state and one for the excited state. They should be symmetric around the center of the plot.
If the peaks are not clear, the experiment is considered failed.
If you observe more than two clear peaks, the experiment is considered failed.
Otherwise, the experiment is considered successful.
            
For example, the following Image is a successful experiment plot:
Image("ref_images/success_ConditionalStarkShiftContinuous.plot_fourier.png")
The following Image is a failure case for the experiment due to the presence of multiple peaks:
Image("ref_images/failure_ConditionalStarkShiftContinuous.plot_fourier.png")
""")
    def plot_fourier(self):
        fig = go.Figure()

        def plot_fourier_trace(fig, data, label, fit_params=None):
            color_ground = 'mediumblue'
            color_excited = 'crimson'

            args = {'start': self.start, 'stop': self.stop, 'step': self.step}
            t = np.arange(args['start'], args['stop'], args['step'])

            color = color_ground if label == 'Ground' else color_excited

            # Compute the Fourier Transform of the data

            data = data - np.mean(data)

            fourier_transform = np.fft.fft(data)
            frequencies = np.fft.fftfreq(t.size, d=args['step'])

            # Compute the amplitude of the Fourier Transform
            amplitude = np.abs(fourier_transform)

            # Sort frequencies in ascending order and reorder amplitude accordingly
            sorting_indices = np.argsort(frequencies)
            sorted_frequencies = frequencies[sorting_indices]
            sorted_amplitude = amplitude[sorting_indices]

            fig.add_trace(
                go.Scatter(x=sorted_frequencies, y=sorted_amplitude, mode="lines+markers",
                           name=label, opacity=0.5, marker=dict(color=color)))

        plot_fourier_trace(fig=fig,
                           data=self.result[:, 0, 0] + 1.j * self.result[:, 0, 1],
                           label="Ground")
        plot_fourier_trace(fig=fig,
                           data=self.result[:, 1, 0] + 1.j * self.result[:, 1, 1],
                           label="Excited")

        fig.update_layout(
            title="ZZ interaction Hamiltonian tomography - Fourier Transform",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Amplitude [a.u.]",
            plot_bgcolor='white',
            legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    @register_browser_function()
    @visual_inspection("""
I have a plot showing status of a qubit over an experiment.
My objective is to determine if the experiment is a success. The success of the experiment should see the state
of the qubits remains stable throughout the experiment.
If you observe oscillations, or the lines crosses each other, the experiment is considered failed.
Otherwise, the experiment is considered successful.
For example, the following Image is a successful experiment plot:
Image("ref_images/success_ConditionalStarkShiftContinuous.plot_control_population.png")
The following Image is a failure case for the experiment:
Image("ref_images/failure_ConditionalStarkShiftContinuous.plot_control_population.png")
    """)
    def plot_control_population(self):
        self.analyze_results_with_errs()

        fig = go.Figure()

        self.plot_specific_axis(fig, data=self.result_control[:, 0, 1], label="Ground",
                                use_imaginary_part=True)
        self.plot_specific_axis(fig, data=self.result_control[:, 1, 1], label="Excited",
                                use_imaginary_part=True)

        fig.update_layout(title="Control qubit state - Z axis",
                          xaxis_title="Pulse width [us]",
                          yaxis_title="<Z>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    @text_inspection
    def fitting(self) -> Union[str, None]:

        result = self.analyze_results_with_errs()

        error = result.get('error', None)

        if error is not None:
            return "The experiment failed due to fitting error."

        prompt_1 = f"""
The fitting reports when the control qubit is at the ground state, the target is oscillating at a frequency of {self.fitting_2D[0]['Frequency']} MHz,
and when the control qubit is at the excited state, the target is oscillating at a frequency of {self.fitting_2D[1]['Frequency']} MHz.
Therefore the IZ rate is {self.iz_rate} MHz and the ZZ rate is {self.zz_rate} MHz. The sampling number per ZZ period is {np.abs(1 / self.zz_rate / self.step)}.
We have observed {np.abs(self.stop * self.zz_rate)} periods of the ZZ interaction.
Note that the sign of the ZZ rate is corresponds to the direction of the rotation and it is allowed to be negative.
"""

        z_control_diff = self.result_control[:, 0, 1] - self.result_control[:, 1, 1]
        z_control_diff_max = np.max(np.abs(z_control_diff))
        z_control_diff_min = np.min(np.abs(z_control_diff))

        prompt_2 = f"""
The expectation value of the control qubit along the Z axis is stable through the whole experiment.
The maximum difference between the ground and excited state is {z_control_diff_max} and the minimum difference is {z_control_diff_min}.
The experiment should be considered successful if the minimum difference is greater than 50% of the maximum difference.
If the experiment is failed because of the population of the control qubit does not meet the criteria, do not retry and directly report the failure.
"""

        analyze_prompt = """
For the fitting results, if the absolute value of oscillation frequency is significantly different between the ground and excited states (more than 50%), 
the experiment is considered failed due to fitting error. 
For the fitting results, if the oscillation amplitude is less than 0.2, the experiment is considered failed due to noise data. 
The experiment is otherwise considered success.
"""

        prompt = f"""
{prompt_1}
{prompt_2}
{analyze_prompt}
<requirement>
Output your analysis of the success/failure of the experiment by a JSON with the following keys:
"fitting_analysis" (string): The analysis about whether the experiment succeeded or failed.
"success" (bool): Whether the experiment succeeded.
</requirement>              
"""
        res = Chat(prompt).complete(parse="dict", cache=True, expensive=True)
        return res


class ConditionalStarkShiftRepeatedGate(Experiment):
    _v_prompt = """
    I have a plot showing ZZ interaction Hamiltonian tomography along the Y axis. The plot includes data points and fit
    lines for both ground and excited states. The X-axis represents pulse width in microseconds, and the Y-axis shows
    the expectation value ⟨Y⟩. The data points are connected by lines, and there are separate fit lines for the ground
    (blue) and excited (pink) states. My objective is to determine whether the oscillations in the data are sinusoidal.
    The success of the experiment depends on observing sinusoidal oscillations in both the ground and excited state data. 
    The amplitude of the oscillations should be significant (more than 0.3), otherwise the experiment is invalid.
    You should observe multiple oscillation periods in the data, otherwise the experiment is invalid.  
    Can you inspect the figure, analyze the oscillations, and conclude whether the experiment is valid based on the
    presence of sinusoidal oscillations?
    """

    _experiment_result_analysis_instructions = """
This experiment is a Repeated Gate Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions between two qubits under microwave drives.
Please check the results of the visual inspection of the plots and the fitting results. 

For visual inspection, if the inspection plot_fourier indicates a failure, the experiment is considered failed and Suggested parameter updates to None.

If the experiment is failed because of the population of the control qubit doesnot meet the criteria, do not retry and directly report the failure.

If the above check passes, the experiment is considered successful.
"""

    @log_and_record(overwrite_func_name='ConditionalStarkShiftRepeatedGate.run')
    def run_simulated(self, duts, amp_control, amp_target, frequency, phase_diff=0,
                      rise=0.03,
                      echo=True, iz_control=0, iz_target=0, width=0, start_gate_number=0,
                      gate_count=40, zz_rate=None):
        """
        Runs the Repeated Gate Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions
        between two qubits under microwave drives.

        Parameters:
        duts (List[Qubit]): The list of qubits to be used in the experiment.
        amp_control (float): The amplitude applied to the control qubit (the first qubit in the list).
        amp_target (float): The amplitude applied to the target qubit (the second qubit in the list).
        frequency (float): The frequency of the stark drive pulse.
        phase_diff (float): The phase difference between the stark drive pulse send to the control and target qubits.
        rise (float): The rising and dropping edge time of the stark drive pulses.
        echo (bool): Whether to include echo sequences.
        iz_control (float): The active cancellation of the IZ rate applied to the control qubit.
        iz_target (float): The active cancellation of the IZ rate applied to the target qubit.
        width (float): The width of the stark drive pulses.
        start_gate_number (int): The number of gates to start with.
        gate_count (int): The maximum number of gates to apply in the sweep.
        zz_rate (float): The ZZ rate of the interation measured from the conditional stark shift continious experiment.
        """
        self.duts = duts
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase_diff = phase_diff
        self.width = width
        self.iz_control = iz_control
        self.iz_target = iz_target
        self.rise = rise
        self.start_gate_number = start_gate_number
        self.gate_count = gate_count

        if zz_rate is None:
            zz_rate = 0.125 / 2 / self.width

        self.zz_rate_continous = zz_rate

        c1_control = self.duts[0].get_default_c1()
        c1_target = self.duts[1].get_default_c1()

        simulator_setup = setup().get_default_setup()
        virtual_transmon_1 = simulator_setup.get_virtual_qubit(self.duts[0])
        virtual_transmon_2 = simulator_setup.get_virtual_qubit(self.duts[1])

        # Evaluate the ZZ value

        from leeq.theory.sizzel_gate.sizzel_simulation import ret_zz

        coupling = simulator_setup.get_coupling_strength_by_qubit(virtual_transmon_1,
                                                                  virtual_transmon_2)
        drive_omega = simulator_setup.get_omega_per_amp(
            channel=c1_control.channel) * amp_control

        delta = self.frequency - virtual_transmon_1.qubit_frequency
        freq_delta = virtual_transmon_2.qubit_frequency - virtual_transmon_1.qubit_frequency

        zz_value = ret_zz(
            alpha1=virtual_transmon_1.anharmonicity,
            alpha2=virtual_transmon_2.anharmonicity,
            J=coupling,
            eps=drive_omega,
            delta=delta,
            freq_delta=freq_delta
        )[0]  # Delta 11 is the ZZ value

        # assume we under shoot by 10%

        zz_value = zz_value * 0.9 * width

        self.result, self.result_control = (
            _generate_zz_interaction_data_from_simulation(qubits=self.duts,
                                                          start=start_gate_number,
                                                          stop=start_gate_number + gate_count,
                                                          sweep_points=gate_count,
                                                          zz_value=zz_value,
                                                          drive_frequency=self.frequency,
                                                          amp_control=amp_control
                                                          ))

    @log_and_record
    def run(self, duts, amp_control, amp_target, frequency, phase_diff=0, rise=0.03,
            echo=True, iz_control=0, iz_target=0, width=0, start_gate_number=0,
            gate_count=40, zz_rate=None):
        """
        Runs the Repeated Gate Conditional Stark Tune-Up Rabi XY experiment for calibrating the IZ and ZZ interactions
        between two qubits under microwave drives.

        Parameters:
        duts (List[Qubit]): The list of qubits to be used in the experiment.
        amp_control (float): The amplitude applied to the control qubit (the first qubit in the list).
        amp_target (float): The amplitude applied to the target qubit (the second qubit in the list).
        frequency (float): The frequency of the stark drive pulse.
        phase_diff (float): The phase difference between the stark drive pulse send to the control and target qubits.
        rise (float): The rising and dropping edge time of the stark drive pulses.
        echo (bool): Whether to include echo sequences.
        iz_control (float): The active cancellation of the IZ rate applied to the control qubit.
        iz_target (float): The active cancellation of the IZ rate applied to the target qubit.
        width (float): The width of the stark drive pulses.
        start_gate_number (int): The number of gates to start with.
        gate_count (int): The maximum number of gates to apply in the sweep.
        zz_rate (float): The ZZ rate of the interation measured from the conditional stark shift continious experiment.
        """
        self.duts = duts
        self.frequency = frequency
        self.amp_control = amp_control
        self.amp_target = amp_target
        self.phase_diff = phase_diff
        self.width = width
        self.iz_control = iz_control
        self.iz_target = iz_target
        self.rise = rise
        self.start_gate_number = start_gate_number
        self.gate_count = gate_count

        if zz_rate is None:
            zz_rate = 0.125 / 2 / self.width

        self.zz_rate_continous = zz_rate

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
            phase_diff=self.phase_diff,
            iz_control=self.iz_control,
            iz_target=self.iz_target,
            echo=echo,
            trunc=1.05,
            zz_interaction_positive=True
            # It doesn't matter what to use here, after one tomography we will find out.
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

    def run_repeated_gate_experiment(self, initial_lpb, initial_gate, repeated_block,
                                     final_gate, pulse_count,
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
                [int_target * ini_control] + [rep] * (n) + [
                    fin + mprim_target * mprim_control])
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

            self.fit_result = fit_2d_freq_with_cov(self.complex_data, dt=t_step,
                                                   freq_guess=0.125, use_freq_bound=True)
            self.fitting_2D.append(self.fit_result)

        self.iz_rate = (self.fitting_2D[0]['Frequency'] + self.fitting_2D[1][
            'Frequency']) / 2
        self.zz_rate = (self.fitting_2D[0]['Frequency'] - self.fitting_2D[1][
            'Frequency']) / 2

        print(
            f"IZ: {self.iz_rate: 0.5f} PGC, ZZ: {self.zz_rate: 0.5f} PGC (per gate count)")

        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
        }

    def plot_specific_axis(self, fig, t, data, label, fit_params=None, t_interpolate=None,
                           use_imaginary_part=False):
        """
        Helper function to plot specific axis using Plotly based on the real or imaginary part of the fit.
        """
        color_ground = 'mediumblue'
        color_excited = 'crimson'
        color = color_ground if label == 'Ground' else color_excited

        fig.add_trace(
            go.Scatter(x=t, y=data, mode='lines+markers', name=label, opacity=0.5,
                       marker=dict(color=color)))

        if fit_params is not None:
            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * t[0]
            o_real = fit_params['Offset_real'].nominal_value
            o_imag = fit_params['Offset_imag'].nominal_value

            fit = a * np.exp(1j * (2.0 * np.pi * f * t_interpolate + p)) + (
                        o_real + 1j * o_imag)
            fit_values = np.real(fit) if not use_imaginary_part else np.imag(fit)

            fig.add_trace(
                go.Scatter(x=t_interpolate, y=fit_values, mode='lines',
                           name=f'{label} Fit', line=dict(color=color),
                           visible='legendonly'))

    @register_browser_function()
    # @visual_analyze_prompt(_v_prompt)
    def plot_x_axis(self):
        """
        Plot the results for the X axis using Plotly.
        """
        t = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1)
        t_interpolate = np.arange(self.start_gate_number,
                                  self.start_gate_number + self.gate_count, 1 / 10)

        fig = go.Figure()
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 0, 0], label="Ground",
                                fit_params=self.fitting_2D[0], use_imaginary_part=False)
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 1, 0], label="Excited",
                                fit_params=self.fitting_2D[1], use_imaginary_part=False)

        fig.update_layout(title="ZZ interaction repeated gate tomography - X axis",
                          xaxis_title="Pulse count",
                          yaxis_title="<X>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    @register_browser_function()
    # @visual_analyze_prompt(_v_prompt)
    def plot_y_axis(self):
        """
        Plot the results for the Y axis using Plotly.
        """
        t = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1)
        t_interpolate = np.arange(self.start_gate_number,
                                  self.start_gate_number + self.gate_count, 1 / 10)

        fig = go.Figure()
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 0, 1], label="Ground",
                                fit_params=self.fitting_2D[0], use_imaginary_part=False)
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 1, 1], label="Excited",
                                fit_params=self.fitting_2D[1], use_imaginary_part=False)

        fig.update_layout(title="ZZ interaction repeated gate tomography - Y axis",
                          xaxis_title="Pulse count",
                          yaxis_title="<Y>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    # @register_browser_function()
    def plot_z_axis(self):
        """
        Plot the results for the Y axis using Plotly.
        """
        t = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1)
        t_interpolate = np.arange(self.start_gate_number,
                                  self.start_gate_number + self.gate_count, 1 / 10)

        fig = go.Figure()
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 0, 2], label="Ground",
                                use_imaginary_part=False)
        self.plot_specific_axis(fig=fig, t=t, t_interpolate=t_interpolate,
                                data=self.result[:, 1, 2], label="Excited",
                                use_imaginary_part=False)

        fig.update_layout(title="ZZ interaction repeated gate tomography - Z axis",
                          xaxis_title="Pulse count",
                          yaxis_title="<Z>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    @register_browser_function()
    @visual_inspection("""
I have a plot showing ZZ interaction Hamiltonian tomography in the Fourier space. The X-axis represents 
the frequency, and the Y-axis shows the amplitude of the fourier transformed value.
My objective is to determine if the experiment is a success.
The success of the experiment depends on observing two clear peaks in the Fourier space, 
one for the ground state and one for the excited state. They should be symmetric around the center of the plot.
If the peaks are not clear, the experiment is considered failed.
If you observe more than two clear peaks, the experiment is considered failed.
Otherwise, the experiment is considered successful.
For example, the following Image is a successful experiment plot:
Image("ref_images/success_ConditionalStarkShiftRepeatedGate.plot_fourier.png")
The following Image is a failure case for the experiment due to the presence of multiple peaks:
Image("ref_images/failure_3_ConditionalStarkShiftRepeatedGate.plot_fourier.png")
                """)
    def plot_fourier(self):
        fig = go.Figure()

        def plot_fourier_trace(fig, data, label, fit_params=None):
            color_ground = 'mediumblue'
            color_excited = 'crimson'

            t = np.arange(self.start_gate_number,
                          self.start_gate_number + self.gate_count, 1)

            color = color_ground if label == 'Ground' else color_excited

            # Compute the Fourier Transform of the data

            data = data - np.mean(data)

            fourier_transform = np.fft.fft(data)
            frequencies = np.fft.fftfreq(t.size, d=1)

            # Compute the amplitude of the Fourier Transform
            amplitude = np.abs(fourier_transform)

            # Sort frequencies in ascending order and reorder amplitude accordingly
            sorting_indices = np.argsort(frequencies)
            sorted_frequencies = frequencies[sorting_indices]
            sorted_amplitude = amplitude[sorting_indices]

            fig.add_trace(
                go.Scatter(x=sorted_frequencies, y=sorted_amplitude, mode="lines+markers",
                           name=label, opacity=0.5, marker=dict(color=color)))

        plot_fourier_trace(fig=fig,
                           data=self.result[:, 0, 0] + 1.j * self.result[:, 0, 1],
                           label="Ground")
        plot_fourier_trace(fig=fig,
                           data=self.result[:, 1, 0] + 1.j * self.result[:, 1, 1],
                           label="Excited")

        fig.update_layout(
            title="ZZ interaction repeated gate tomography - Fourier Transform",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Amplitude [a.u.]",
            plot_bgcolor='white',
            legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    @register_browser_function()
    @visual_inspection("""
I have a plot showing status of a qubit over an experiment.
My objective is to determine if the experiment is a success. The success of the experiment should see the state
of the qubits remains stable in the Y axis throughout the experiment.
If you observe oscillations in the Y axis, or the lines crosses each other, the experiment is considered failed.
Otherwise, the experiment is considered successful.
For example, the following Image is a successful experiment plot:
Image("ref_images/success_ConditionalStarkShiftRepeatedGate.plot_control_population.png")
The following Image is a failure case for the experiment due to the presence of multiple peaks:
Image("ref_images/failure_ConditionalStarkShiftRepeatedGate.plot_control_population.png")
""")
    def plot_control_population(self):

        fig = go.Figure()
        t = np.arange(self.start_gate_number, self.start_gate_number + self.gate_count, 1)

        self.plot_specific_axis(fig, t=t, data=self.result_control[:, 0, 1],
                                label="Ground",
                                use_imaginary_part=True)
        self.plot_specific_axis(fig, t=t, data=self.result_control[:, 1, 1],
                                label="Excited",
                                use_imaginary_part=True)

        fig.update_layout(title="Control qubit state - Z axis",
                          xaxis_title="Pulse count",
                          yaxis_title="<Z>",
                          plot_bgcolor='white',
                          legend=dict(x=0, y=1, traceorder='normal'))

        return fig

    def plot(self):
        """
        Plot the results.
        """
        args = self._get_run_args_dict()

        t = np.arange(args['start_gate_number'],
                      args['start_gate_number'] + args['gate_count'], 1)
        t_interpolate = np.arange(args['start_gate_number'],
                                  args['start_gate_number'] + args['gate_count'], 1 / 10)

        def plot_specific_axis(data, label, fit_params, use_imaginary_part):
            data = data.squeeze()

            plt.scatter(t, data, label=label, alpha=0.5)

            f = fit_params['Frequency'].nominal_value
            a = fit_params['Amplitude'].nominal_value
            p = fit_params['Phase'].nominal_value - 2.0 * np.pi * f * args[
                'start_gate_number']
            o = fit_params['Offset_real'].nominal_value + 1j * fit_params[
                'Offset_imag'].nominal_value

            fit = a * np.exp(1.j * (2.0 * np.pi * f * t_interpolate + p)) + o

            plt.plot(t_interpolate,
                     np.real(fit) if not use_imaginary_part else np.imag(fit))

        plt.figure(figsize=(6, 5))

        desired_num_ticks = 10  # Desired number of ticks
        step = max(1, len(t) // desired_num_ticks)
        xticks_subset = t[::step]
        plt.title(f"ZZ interaction repeated gate tomography - X axis")

        plot_specific_axis(data=self.result[:, 0, 0], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=False)
        plot_specific_axis(data=self.result[:, 1, 0], label="Excited",
                           fit_params=self.fitting_2D[1],
                           use_imaginary_part=False)

        plt.xlabel("Pulse count")
        plt.ylabel("<X>")
        plt.legend()
        plt.xticks(xticks_subset)

        plt.figure(figsize=(20, 5))
        plt.title(f"ZZ interaction repeated gate tomography - Y axis")

        plot_specific_axis(data=self.result[:, 0, 1], label="Ground",
                           fit_params=self.fitting_2D[0],
                           use_imaginary_part=True)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited",
                           fit_params=self.fitting_2D[1],
                           use_imaginary_part=True)

        plt.xlabel("Pulse count")
        plt.ylabel("<Y>")
        plt.legend()
        plt.xticks(xticks_subset)

        plt.show()

    def get_ai_inspection_results(self):
        """
        Returns the results of the AI inspection for the experiment.
        """

        inspection_results = super().get_ai_inspection_summary()
        zz_pgc = self.zz_rate
        zz_rate = self.zz_rate_continous
        zz_rate = np.sign(zz_pgc) * np.abs(zz_rate)

        target_zz = np.sign(zz_pgc) * 0.125
        zz_diff = target_zz - zz_pgc
        width_diff = np.sign(zz_diff / zz_rate) * min(
            np.abs(zz_diff / zz_rate / 2), 0.05 * self.width
        )
        width = self.width

        inspection_results['Calibrated parameters'] = {
            'amp_control': self.amp_control,
            'amp_target': self.amp_target,
            'frequency': self.frequency,
            'rise': self.rise,
            'phase_diff': self.phase_diff,
            'width': width + width_diff,
            'zz_interaction_positive': self.zz_rate > 0
        }

        return inspection_results

    @text_inspection
    def fitting(self) -> Union[str, None]:
        """
        Returns the analyzed result prompt for the experiment.
        """

        z_control_diff = self.result_control[:, 0, 1] - self.result_control[:, 1, 1]
        z_control_diff_max = np.max(np.abs(z_control_diff))
        z_control_diff_min = np.min(np.abs(z_control_diff))

        if z_control_diff_min < 0.25 * z_control_diff_max:
            extra_prompt = "The experiment failed because the population of the control qubit does not meet the criteria"
        else:
            extra_prompt = "The experiment is successful."

        prompt = f"""
        The expectation value of the control qubit along the Z axis is stable through the whole experiment.
        The maximum difference between the ground and excited state is {z_control_diff_max} and the minimum difference is {z_control_diff_min}.
        The experiment should be considered successful if the minimum difference is greater than 25% of the maximum difference.
        """ + extra_prompt

        return prompt


class ConditionalStarkEchoTuneUpAI(Experiment):
    """
    Class for performing Conditional Stark Echo Tune-Up experiments.
    """

    _experiment_result_analysis_instructions = """
The Conditional Stark Echo Tune-Up experiment has been completed. Please read the following report to analyze the
if this is a successful experiment. Make the analysis concise and clear in one short sentence describing the reason. 
"""

    def run_simulated(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @log_and_record
    def run(
            self,
            duts: List[Any],
            params: Dict[str, Any] = None,
            frequency: float = None,
            amplitude: float = None,
            phase_diff: float = 0,
            rise: float = 0.015,
            t_start: float = 0,
            t_stop: float = 15,
            sweep_points: int = 30,
            n_start: int = 0,
            n_stop: int = 32,
            update_iz: bool = False,
            update_zz: bool = True,
            n_max_iteration: int = 1,
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
            amplitude (float, optional): Amplitude control for the experiment. Defaults to None.
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

        self.ai_inspection = True

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
                'amp_control': amplitude,
                'amp_target': amplitude * area_target / area_control,
                'rise': rise,
                'width': 0,
                'phase_diff': phase_diff,
                'zz_interaction_positive': True,
                'echo': True
            }

        self.current_params = params
        self.params_list = [params]

        try:
            iz_rate, zz_rate, self._xy_hamiltonian_tomography_inspection_results = self.run_sizzel_xy_hamiltonian_tomography(
                t_start=t_start, t_stop=t_stop, sweep_points=sweep_points
            )
        except Exception as e:
            self._xy_hamiltonian_tomography_inspection_results = {'success': False,
                                                                  'analysis': f'Exception occurred: {e}'
                                                                  }
            self._repeated_gate_inspection_results = {'success': False,
                                                      'analysis': f'Exception occurred: {e}'
                                                      }
            return

        if not self._xy_hamiltonian_tomography_inspection_results['success']:
            self._repeated_gate_inspection_results = {'success': False,
                                                      'analysis': (
                                                          'Skipped due to the failure of '
                                                          'Hamiltonian tomography experiment.')
                                                      }
            return

        if self.current_params['width'] > 0.4:
            self._repeated_gate_inspection_results = {'success': True,
                                                      'analysis': (
                                                          'Skipped due the width estimation evaluated from hamiltonian tomography is too long.')
                                                      }
            return

        self.current_params['zz_interaction_positive'] = zz_rate.nominal_value > 0

        try:
            self._repeated_gate_inspection_results = self.run_repeated_gate_hamiltonian_tomography(
                zz_rate=zz_rate, n_start=n_start, n_stop=n_stop, update_iz=False,
                update_zz=True
            )
        except Exception as e:
            self._repeated_gate_inspection_results = {'success': False,
                                                      'analysis': f'Exception occurred: {e}'
                                                      }

    @text_inspection
    def fitting(self) -> Union[str, None]:

        res_dict = {
            "Inspection results from hamiltonian_tomography": self._xy_hamiltonian_tomography_inspection_results,
            "Inspection results from repeated_gate_hamiltonian_tomography": self._repeated_gate_inspection_results,
            "fitted parameters": self.current_params
        }

        return res_dict

    def _check_data_validity_using_ai(self, experiment: Experiment,
                                      additional_information: str, show=True) -> dict[
        str, str]:
        if not self.ai_inspection:
            return {
                'analysis': 'AI inspection is not enabled. Always assumes the data is valid.',
                'success': True,
            }

        inspection_results = experiment.get_ai_inspection_summary()

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
        chat = mllm.Chat(prompt,
                         "You are a very smart and helpful assistant who only reply in JSON dict. " +
                         "Keep everything in a same line in the response.")
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

            sizzel_xy = ConditionalStarkShiftContinuous(duts=self.duts, frequency=self.current_params['frequency'],
                                                            amp_control=self.current_params['amp_control'],
                                                            amp_target=self.current_params['amp_target'],
                                                            rise=self.current_params['rise'],
                                                            start=t_start, stop=t_stop, sweep_points=sweep_points,
                                                            phase_diff=self.current_params['phase_diff'], echo=True)

            inspection_results = sizzel_xy.get_ai_inspection_summary()
        else:
            sizzel_xy = ConditionalStarkShiftContinuous(
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

        try:
            iz_rate = result['iz_rate']
            zz_rate = result['zz_rate']

            new_params = self.current_params.copy()
            new_params['width'] = np.abs(0.125 / zz_rate.nominal_value) / 2

            fitted_results_str = f'Estimated IZ = {iz_rate} MHz, ZZ = {zz_rate} MHz, width = {new_params["width"]} us'
            print(fitted_results_str)

            self.params_list.append(new_params)
            self.current_params = new_params
        except Exception as e:
            iz_rate = None
            zz_rate = None
            inspection_results['success'] = False
            inspection_results['error'] = f"Failed to analyze the results. Fitting error"

        return iz_rate, zz_rate, inspection_results

    def run_repeated_gate_hamiltonian_tomography(
            self,
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
            while try_count < 1:
                if self.ai_inspection:

                    prompt = f"""
                        Please implement the ConditionalStarkShiftRepeatedGate experiment with the provided parameters.

                        The experiment should be run with the following parameters:
                        - duts: duts in the available variable
                        - frequency: {self.current_params['frequency']}
                        - amp_control: {self.current_params['amp_control']}
                        - amp_target: {self.current_params['amp_target']}
                        - rise: {self.current_params['rise']}
                        - start_gate_number: {n_start}
                        - gate_count: {n_stop}
                        - width: {width}
                        - phase_diff: {self.current_params['phase_diff']}
                        - echo: True
                    """

                    """
                    next_stage_guide = "Go to Complete if success. Otherwise Fail."
                    ai_experiment = OneInstExecutionAgent(
                        prompt,
                        duts=self.duts,
                        next_stage_guide=next_stage_guide)
                    repeated_gate = ai_experiment.get_last_experiment()
                    """

                    ai_experiment_var_table = execute_experiment_from_prompt(
                        prompt=prompt, duts=self.duts,
                    )
                    repeated_gate = get_exp_from_var_table(ai_experiment_var_table)
                else:
                    repeated_gate = ConditionalStarkShiftRepeatedGate(
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

                inspection_results = self._check_data_validity_using_ai(repeated_gate,
                                                                        fitted_results_str)

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

            print(
                f'Kalman estimated ZZ pgc after measurement = {kalman_zz.x}+-{np.sqrt(kalman_zz.P)}')
            print(
                f'Kalman estimated IZ pgc after measurement = {kalman_iz.x}+-{np.sqrt(kalman_iz.P)}')

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
                    np.abs(zz_diff / zz_rate.nominal_value / 2),
                    0.05 * self.current_params['width']
                )
                zz_diff = zz_rate.nominal_value * width_diff * 2
                width += width_diff
                iz_diff = 0
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

            print(
                f'Kalman estimated ZZ pgc after update = {kalman_zz.x}+-{np.sqrt(kalman_zz.P)}')
            print(
                f'Kalman estimated IZ pgc after update = {kalman_iz.x}+-{np.sqrt(kalman_iz.P)}')

            print(
                f'ZZ accuracy check pass: {zz_accuracy_check}, ZZ uncertainty check pass: {zz_uncertainty_check}')
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

    def get_ai_inspection_summary(self):
        inspection_results = super().get_ai_inspection_summary()
        inspection_results['Calibrated parameters'] = self.current_params

        return inspection_results


class ConditionalStarkTwoQubitGateAIParameterSearchFull(Experiment):
    # _experiment_result_analysis_instructions = """
    # The Conditional Stark Echo Tune-Up experiment has been completed. Please read the following report to analyze the
    # if this is a successful experiment. Make the analysis concise and clear in one short sentence describing the reason.
    # """

    _background_information = """
            Your objective is to find the optimal parameters for the conditional stark-shift gate that will allow you to entangle 
            two qubits. The parameters you need to find are 
            <parameters>
                'frequency': the frequency of the drive pulse ,
                'amp_control':  the amplitude of the control qubit (The first qubit),
                'rise': the rise time of the gate,
                'width': the width of the driving pulse,
                'phase_diff': the phase difference between the control and target qubits,
                'zz_interaction_positive': the sign of the ZZ interaction,
            </parameters>

            <Rules of parameter chosing>
            'frequency': It can be below, between or above the single qubit transition transition frequencies. 
                            It has to be at least 30 MHz away from both of the single qubit transition frequency.
                            It should not be lower than 60MHz below the lowest qubit frequency and not higher than 60MHz above the highest qubit frequency.
                            Round it to multiples of MHz.
            'amp_control': You should try from 1 time to 2 times of the first qubit's single qubit gate drive amplitude. 
                        The experiment may fail when chosing a amplitude too high, therefore you should start from a gentle value. The maximum value is 1. Adjust the amplitude to be multiples of 0.05.
            'rise': No more than 0.02, no less than 0.01. Usually 0.015 is the right value to choose.
            'phase_diff': keep it 0,
            'width': determined by the experiment output,
            'zz_interaction_positive': determined by the experiment output,

            Try different set of parameters, particularly varying the frequency and amplitudes, to find the ZZ rate and the pulse width.
            Try parameters of below the lowest qubit frequency, between the qubit frequencies and above the highest qubit frequency.
            The optimal parameters give the highest ZZ rate and the lowest width.
            If an experiment succeeds, you can try to improve the results by increase the amplitude or move the frequency closer to the qubits.
            If an experiment fails, you should try to move the frequency further away from the qubits or decrease the amplitude.
            To make the results comparable with the history results, do not chose a new set of parameters with both new frequency and amplitudes. 
            If the experiment failed at a certain frequency and amplitude, this is usually the frequency choice is too close to the qubit frequency or the amplitude is too high.
            </Rules of parameter chosing>
        """

    # should be around 30MHz below the lowest qubit frequency to 60 MHz below the lowest qubit frequency.

    _objective_prompt = """
        Please suggest the next experiment you would like to run to find the parameters for the conditional stark-shift gate,
        based on the previous experiment history.

        Return in a json dict with the following format:
        {
            'status': <'searching', 'error' or 'finish'>
            'analysis': <The reason for choosing this parameter set>,
            'params': {
                'frequency': float,
                'amp_control': float,
                'amp_target': float,
                'rise': float,
                'width': float,
                'phase_diff': float,
                'zz_interaction_positive': boolean
            },
        }

        If you find a set of parameters that has width less than 0.2 us immediately set status to finish. 
        If you have done 50 experiment, set status to finish.
        If you have done 20 experiment and could not find any improvements, return the set of parameters you believe to be optimal,
        please set status to 'finish'. Otherwise keeps state to 'searching' and trying new parameters. 
        If you have encountered an error, please set status to 'error'.
        """

    @log_and_record
    def run(
            self,
            duts: List[TransmonElement],
            params: Dict[str, Any] = None,
            ai_inspection: bool = False
    ) -> None:
        """
        Run the Conditional Stark Echo Tune-Up experiment to calibrate the siZZel two qubit gate parameters for a
        pair of qubits.

        Parameters:
            duts: List[TransmonElement]: Devices under test.
            params: Dict[str, Any]: Parameters for the experiment. Defaults to None.
            ai_inspection: bool: Flag for AI inspection. Defaults to False. Please set it to True if you
                want to use the AI inspection feature, or you are an AI writing the code.

        Example:
            >>> experiment_instance = ConditionalStarkTwoQubitGateAIParameterSearch(
            >>>     duts=[dut1, dut2],
            >>> )
        """
        self._experiment_history = []
        self._analyze_histroy = []
        self.duts = duts

        while self._run_next_experiment() == True:
            pass
        pass

    def _get_device_parameters_prompts(self):
        prompt = f"You have access to the following two qubits: {self.duts[0]._name} and {self.duts[1]._name}. The units for frequency and time are in MHz and microseconds The parameters for these qubits are as follows:"

        for dut in self.duts:
            prompt += f"""
            <{dut._name} Parameters>
            Single qubit gate parameters: {dut.get_c1('f01').get_parameters()}
            </{dut._name} Parameters>
            """

        return prompt

    def _experiment_history_to_prompt(self):

        if len(self._experiment_history) == 0:
            return "You have not run any experiments yet."

        prompt = f"Here is the history of the experiments you have run so far:"

        for i, exp in enumerate(self._experiment_history):
            result = exp.get_ai_inspection_summary()
            analyze_results = {k: v for k, v in result.items() if
                               k in ['success', 'Calibrated parameters',
                                     'analysis']}

            section_prompt = f"""\n\n\n
            <{i}:{exp._name}>
            {analyze_results}
            </{i}:{exp._name}>
            """

            prompt += section_prompt

        return prompt

    def _display_experiment_history(self):

        if len(self._experiment_history) == 0:
            return

        html_dict = {}

        for i, exp in enumerate(self._experiment_history):
            result = exp.get_ai_inspection_summary()
            analyze_results = {k: v for k, v in result.items() if
                               k in ['success', 'Calibrated parameters',
                                     'analysis']}
            html_dict[f"{i}:{exp._name}"] = analyze_results

        html = dict_to_html(html_dict)

        display_chat(agent_name=f"Previous experiments",
                     content='<br>' + html,
                     background_color='#f0f8ff')

    def _run_next_experiment(self):

        prompt = self._background_information + self._get_device_parameters_prompts() + self._experiment_history_to_prompt() + self._objective_prompt
        # print(prompt)

        self._display_experiment_history()

        from mllm import Chat
        chat = Chat(prompt,
                    "You are a very smart and helpful assistant who only reply in JSON dict. Keep everything in a same line in the response.")
        res = chat.complete(parse="dict", expensive=True, cache=True)
        # , model = 'claude-3-opus-20240229'

        self._analyze_histroy.append(res)

        from k_agents.notebook_utils import dict_to_html, display_chat

        html = dict_to_html(res)
        display_chat(agent_name=f"Parameter search AI",
                     content='<br>' + html,
                     background_color='#f0f8ff')

        if res['status'] in ['finish', 'error']:
            return False
        else:
            params = res['params']

            func = ConditionalStarkEchoTuneUpAI.run
            # For compatibility, select the argument that the function
            # accepts with inspect
            sig = inspect.signature(func)

            # Extract the parameter names that the function accepts
            valid_parameter_names = set(sig.parameters.keys())

            # Filter the kwargs
            filtered_kwargs = {
                k: v for k, v in res['params'].items() if k in valid_parameter_names}
            self._experiment_history.append(
                ConditionalStarkEchoTuneUpAI(duts=self.duts, ai_inspection=True,
                                             **filtered_kwargs))

        return True

    @text_inspection
    def fitting(self) -> Union[str, None]:
        return self._analyze_histroy[-1]['analysis']


class TwoQubitTuningEnv(Singleton):
    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self.amplitude_tuning_results = {}
        self.frequency_to_good_amplitude = {}


class ConditionalStarkTwoQubitGateAIParameterSearchBase(Experiment):

    @log_and_record
    def run(
            self,
            duts: List[TransmonElement],
            run_class: Type[Experiment],
            maximum_experiments: int = 20,
            filter_parameters: bool = True,
            **kwargs
    ) -> None:
        """
        The base for runing the Conditional Stark Echo Tune-Up experiment using AI to calibrate the siZZel two qubit gate parameters for a
        pair of qubits.

        Parameters:
            duts: List[TransmonElement]: Devices under test.
            run_class: Type[Experiment]: The experiment class to run.
            maximum_experiments: int: The maximum number of experiments to run. Defaults to 20.
            filter_parameters: bool: Flag to filter the parameters. Defaults to True.
            **kwargs: Dict[str, Any]: Parameters for the experiment.
        """
        self._experiment_history = []
        self._analyze_histroy = []
        self.duts = duts

        for i in range(maximum_experiments):
            if self._run_next_experiment(run_class=run_class, params=kwargs, filter_parameters=filter_parameters) in [
                'finish', 'error']:
                break

    def _get_device_parameters_prompts(self):
        prompt = f"You have access to the following two qubits: {self.duts[0]._name} and {self.duts[1]._name}. The units for frequency and time are in MHz and microseconds The parameters for these qubits are as follows:"

        for dut in self.duts:
            prompt += f"""
            <{dut._name} Parameters>
            Single qubit gate parameters: {dut.get_c1('f01').get_parameters()}
            </{dut._name} Parameters>
            """

        return prompt

    def _display_experiment_history(self):

        if len(self._experiment_history) == 0:
            return

        html_dict = {}

        for i, exp in enumerate(self._experiment_history):
            result = exp.get_ai_inspection_summary()
            analyze_results = {k: v for k, v in result.items() if
                               k in ['success', 'Calibrated parameters',
                                     'analysis']}
            html_dict[f"{i}:{exp._name}"] = analyze_results

        html = dict_to_html(html_dict)

        display_chat(agent_name=f"Previous experiments",
                     content='<br>' + html,
                     background_color='#f0f8ff')

    def _run_next_experiment(self, run_class, params, filter_parameters=True):

        prompt = self._background_information + self._get_device_parameters_prompts() + \
                 self._experiment_history_to_prompt() + self._objective_prompt
        # print(prompt)

        self._display_experiment_history()

        chat = Chat(prompt,
                    "You are a very smart and helpful assistant who only reply in JSON dict. Keep everything in a same line in the response.")
        res = chat.complete(parse="dict", expensive=True, cache=True)
        # , model = 'claude-3-opus-20240229'

        html = dict_to_html(res)
        display_chat(agent_name=f"Parameter search AI",
                     content='<br>' + html,
                     background_color='#f0f8ff')

        if res['status'] not in ['finish', 'error']:
            params_suggests = res['params']

            updated_params = params.copy()
            updated_params.update(params_suggests)

            res['params'] = updated_params

            if filter_parameters:
                func = run_class.run
                # For compatibility, select the argument that the function
                # accepts with inspect
                sig = inspect.signature(func)

                # Extract the parameter names that the function accepts
                valid_parameter_names = set(sig.parameters.keys())

                # Filter the kwargs
                filtered_kwargs = {
                    k: v for k, v in updated_params.items() if k in valid_parameter_names}
                updated_params = filtered_kwargs

            self._experiment_history.append(
                run_class(duts=self.duts, **updated_params))

        self._analyze_histroy.append(res)
        return res['status']

    #@text_inspection
    #def experiment_history(self) -> Union[str, None]:
    #   # return self._analyze_histroy[-1]['analysis']
    #    return self._experiment_history_to_prompt()

class ConditionalStarkTwoQubitGateAmplitudeAdvise(Experiment):
    _rewrite_json_requirement = True

    _experiment_result_analysis_instructions = """
    Output a JSON dict with the following keys:
    "success" (bool): true 
    "best_amplitude" (float): The best amplitude found in a successful experiment.
    "advised_amplitude" (float): The next amplitude to try.
    """

    def run_simulated(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @log_and_record
    def run(self, duts: List[TransmonElement], frequency: float):
        """
        This experiment return a suggestion for the next amplitude to try for the conditional stark-shift gate.
        The suggestion is based on the history of the experiments run so far.
        duts: List[TransmonElement]: Devices under test.
        frequency (float): Frequency for the experiment.
        """
        self.duts = duts
        self.frequency = frequency

    @text_inspection
    def next_parameter(self):
        prompt = f"""
        Your objective is to find the optimal parameters for the conditional stark-shift gate that will allow you to entangle 
        two qubits. The parameters you need to find are 
        <parameters>
        'amp_control':  the amplitude of the control qubit (The first qubit), the required amplitude accuracy is 0.01. 
        </parameters>
        
        <single qubit amplitude>
        qubit 1: {self.duts[0].get_c1('f01').get_parameters()['amp']}
        qubit 2: {self.duts[1].get_c1('f01').get_parameters()['amp']}
        </single qubit amplitude>
        
        <Rules of parameter selection>
        You should try around the amplitude of single qubits. 
        The optimal parameters give the highest ZZ rate and the lowest width.
        The experiment may fail when select a amplitude too high, therefore you should start from a gentle value.
        The maximum amplitude value is 1. The minimum amplitude value is 0.  
        If an experiment succeeds, you can try to improve the results by increase the amplitude.
        If an experiment failed, you can try to recover by reduce the amplitude.
        </Rules of parameter selection>
        
        <Experiment history>
        {self._experiment_history_to_prompt()}
        </Experiment history>
        
        <requirement>
        Suggest the next experiment to determine parameters for the conditional stark-shift gate, incorporating insights from prior experiments. Implement Binary Search methodology where applicable.
        
        Please format your response as a JSON dictionary with the following keys:
        "finished" (bool): whether the experiment is finished.
        "analysis" (str): Explanation for choosing this set of parameters.
        "current_best" (float): The highest control amplitude from a succeeded experiment. The value can be None if no experiment is successful.
        "new_amplitude_to_try" (float): The new amplitude of the control qubit to try. If the experiment is finished, set this to the optimal amplitude.
        </format>
        <requirement>
        """

        chat = Chat(prompt,
                    "You are a very smart and helpful assistant who only reply in JSON dict. Keep everything in a same line in the response.")
        res = chat.complete(parse="dict", expensive=True, cache=True)

        return res

    @text_inspection
    def best_amplitude(self):
        tuning_env = TwoQubitTuningEnv()
        if self.frequency not in tuning_env.amplitude_tuning_results:
            return {
                "best_amp": 'There is no successful experiment yet.'
            }
        results = tuning_env.amplitude_tuning_results[self.frequency]
        amps = []
        for insp in results:
            if insp['success']:
                amps.append(insp['Calibrated parameters']['amp_control'])
        # the largest amp
        if len(amps) == 0:
            return {
                "best_amp": 'There is no successful experiment yet.'
            }
        best_amp = max(amps)
        return {
            "best_amp": best_amp
        }

    def _experiment_history_to_prompt(self):
        tuning_env = TwoQubitTuningEnv()
        if self.frequency not in tuning_env.amplitude_tuning_results:
            return "You have not run any experiments yet."

        results = tuning_env.amplitude_tuning_results[self.frequency]
        prompt = f"Here is the history of the experiments you have run so far:\n"

        for i, insp in enumerate(results):
            section_prompt = f"""
            <experiment>
            amp_control: {insp['Calibrated parameters']['amp_control']}
            frequency: {insp['Calibrated parameters']['frequency']}
            success: {insp['success']}
            analysis: {insp['analysis']}
            </experiment>"""
            prompt += section_prompt

        return prompt


class ConditionalStarkTwoQubitGateAmplitudeAttempt(
    ConditionalStarkTwoQubitGateAIParameterSearchBase):

    _experiment_result_analysis_instructions = ""

    def run_simulated(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @log_and_record
    def run(
            self,
            duts: List[TransmonElement],
            amplitude: float = None,
            frequency: float = None,
            **kwargs
    ) -> None:
        """
        Run the Conditional Stark Echo Tune-Up experiment to calibrate the siZZel two qubit gate parameters for a
        pair of qubits, searching the amplitude.

        Parameters:
            duts: List[TransmonElement]: Devices under test.
            amplitude: float: Amplitude control for the experiment.
            frequency (float, optional): Frequency for the experiment.
            **kwargs: Dict[str, Any]: Parameters for the experiment.

        Example:
            >>> # Assume dut1 and dut2 are the devices under test.
            >>> experiment_instance = ConditionalStarkTwoQubitGateAmplitudeAttempt(
            >>>     duts=[dut1, dut2],
            >>> )
        """
        self.duts = duts
        self.frequency = frequency
        if amplitude is None:
            amplitude = duts[0].get_c1('f01').get_parameters()["amp"]

        exp = ConditionalStarkEchoTuneUpAI(duts=self.duts, frequency=frequency,
                                           amplitude=amplitude, **kwargs)
        inspection = exp.get_ai_inspection_summary()
        tuning_env = TwoQubitTuningEnv()
        if frequency not in tuning_env.amplitude_tuning_results:
            tuning_env.amplitude_tuning_results[frequency] = []
        tuning_env.amplitude_tuning_results[frequency].append(inspection)

        if frequency not in tuning_env.frequency_to_good_amplitude:
            tuning_env.frequency_to_good_amplitude[frequency] = {}

        if inspection['success']:
            tuning_env.frequency_to_good_amplitude[frequency] = {
                'success': True,
                'best_amplitude': amplitude
            }
        else:
            tuning_env.frequency_to_good_amplitude[frequency] = {
                'success': False,
                'best_amplitude': None,
                'analysis': inspection['analysis']
            }


class ConditionalStarkTwoQubitGateFrequencyAdvise(Experiment):
    _rewrite_json_requirement = True

    _experiment_result_analysis_instructions = """
    Output a JSON dict with the following keys:
    "success" (bool): true
    "best_frequency" (float): The best frequency found in a successful experiment.
    "advised_frequency" (float): The next frequency to try.
    """

    def run_simulated(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @log_and_record
    def run(self, duts: List[TransmonElement]):
        """
        This experiment return a suggestion for the next amplitude to try for the conditional stark-shift gate.
        The suggestion is based on the history of the experiments run so far.
        duts: List[TransmonElement]: Devices under test.
        frequency (float): Frequency for the experiment.
        """
        self.duts = duts

    @text_inspection
    def next_frequency(self):
        prompt = f"""
        Your objective is to find the optimal parameters for the conditional stark-shift gate that will allow you to entangle 
        two qubits. The parameters you need to find are 
        <parameters>
        'amp_control':  the amplitude of the control qubit (The first qubit), the required amplitude accuracy is 0.01. 
        </parameters>

        <single qubit frequency>
        qubit 1: {self.duts[0].get_c1('f01').get_parameters()['freq']}
        qubit 2: {self.duts[1].get_c1('f01').get_parameters()['freq']}
        </single qubit frequency>

        <Rules of parameter selection>
        The new frequency can be below, between or above the single qubit transition transition frequencies. 
        It has to be at least 30 MHz away from both of the single qubit transition frequency.
        It should not be lower than 60MHz below the lowest qubit frequency and not higher than 60MHz above the highest qubit frequency.
        Round it to multiples of MHz.
        Try parameters of below the lowest qubit frequency, between the qubit frequencies and above the highest qubit frequency.
        The optimal parameters give the highest ZZ rate and the lowest width.
        If an experiment succeeds, you can try to improve the results by move the frequency closer to the qubits.
        If an experiment fails, you should try to move the frequency further away from the qubits.
        If the experiment failed at a certain frequency, this is usually the frequency choice is too close to the qubit frequency.
        </Rules of parameter selection>

        <Experiment history>
        {self._experiment_history_to_prompt()}
        </Experiment history>

        <requirement>
        First consider the frequency region below the lowest qubit frequency, then between the qubit frequencies and finally above the highest qubit frequency.
        Please suggest the next experiment you would like to run to find the parameters for the conditional stark-shift gate, based on the previous experiment history.
        You should at least try each available region once.

        Please format your response as a JSON dictionary with the following keys:
        "analysis" (str): An analysis of the current situation.
        "finished" (bool): whether the experiment is finished.
        "current_best" (float): The highest control frequency from a succeeded experiment. The value can be None if no experiment is successful.
        "new_frequency_to_try" (float): The new frequency of the control qubit to try. If the experiment is finished, set this to the optimal amplitude.
        </format>
        <requirement>
        """

        chat = Chat(prompt,
                    "You are a very smart and helpful assistant who only reply in JSON dict. Keep everything in a same line in the response.")
        res = chat.complete(parse="dict", expensive=True, cache=True)

        return res

    @text_inspection
    def best_frequency(self):
        tuning_env = TwoQubitTuningEnv()
        best_freq = None
        best_amp = None
        for freq, insp in tuning_env.frequency_to_good_amplitude.items():
            if not insp['success']:
                continue
            best_freq = freq
            best_amp = insp['best_amplitude']
            break
        if best_freq is not None:
            return {
                "best_freq": best_freq,
                "best_amp": best_amp
            }
        else:
            return {
                "best_freq": "These are no successful experiments",
            }

    def _experiment_history_to_prompt(self):
        tuning_env = TwoQubitTuningEnv()
        if len(tuning_env.frequency_to_good_amplitude) == 0:
            return "You have not run any experiments yet."

        prompt = f"Here is the history of the experiments you have run so far:\n"

        for freq, insp in tuning_env.frequency_to_good_amplitude.items():
            section_prompt = f"""
<experiment>
frequency: {freq}
amp_control: {insp["best_amplitude"]}
success: {insp['success']}
analysis: {insp['analysis']}
</experiment>"""
            prompt += section_prompt

        return prompt