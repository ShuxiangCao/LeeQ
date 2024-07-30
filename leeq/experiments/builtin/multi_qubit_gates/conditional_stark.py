# Conditional AC stark shift induced CZ gate

from leeq.utils import setup_logging
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
from leeq.utils.ai import visual_analyze_prompt
from leeq.utils.ai.display_chat.notebooks import dict_to_html, display_chat
from leeq.theory import fits
from leeq.theory.fits.fit_exp import fit_2d_freq_with_cov

from leeq.theory.estimator.kalman import KalmanFilter1D

logger = setup_logging(__name__)

import plotly.graph_objects as go
from labchronicle import log_and_record, register_browser_function
from leeq import Experiment
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.utils.compatibility import *

import matplotlib.pyplot as plt
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial
from leeq.utils.compatibility import prims
from typing import List, Optional, Union
from typing import List, Dict, Any, Tuple
import numpy as np
from uncertainties import ufloat


class ConditionalStarkTuneUpRabiXYAI(Experiment):
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

    def get_analyzed_result_prompt(self) -> Union[str, None]:

        self.analyze_results_with_errs()

        prompt = f"""
        The fitting reports when the control qubit is at the ground state, the target is oscillating at a frequency of {self.fitting_2D[0]['Frequency']} MHz,
        and when the control qubit is at the excited state, the target is oscillating at a frequency of {self.fitting_2D[1]['Frequency']} MHz.
        Therefore the IZ rate is {self.iz_rate} MHz and the ZZ rate is {self.zz_rate} MHz. The sampling number per ZZ period is {1 / self.zz_rate / self.step}.
        We have observed {self.stop * self.zz_rate} periods of the ZZ interaction. 
        """

        return prompt


class ConditionalStarkTuneUpRepeatedGateXYAI(Experiment):
    _v_prompt = """
    I have a plot showing ZZ interaction Hamiltonian tomography along the Y axis. The plot includes data points and fit
    lines for both ground and excited states. The X-axis represents pulse width in microseconds, and the Y-axis shows
    the expectation value ⟨Y⟩. The data points are connected by lines, and there are separate fit lines for the ground
    (blue) and excited (pink) states. My objective is to determine whether the oscillations in the data are sinusoidal.
    The success of the experiment depends on observing sinusoidal oscillations in both the ground and excited state data. 
    The amplitude of the oscillations should be significant (more than 0.5), otherwise the experiment is invalid.
    You should observe multiple oscillation periods in the data, otherwise the experiment is invalid.  
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
            sizzel_xy = ConditionalStarkTuneUpRabiXYAI(
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
                'Experiment success': True,
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
                repeated_gate = ConditionalStarkTuneUpRepeatedGateXYAI(
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
