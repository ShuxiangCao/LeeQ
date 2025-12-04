# Conditional AC stark shift induced CZ gate
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from qutip import Bloch

from leeq import Experiment
from leeq.chronicle import log_and_record, register_browser_function
from leeq.core.primitives.logical_primitives import (
    LogicalPrimitiveBlockSerial,
    LogicalPrimitiveBlockSweep,
)
from leeq.theory import fits
from leeq.theory.estimator.kalman import KalmanFilter1D
from leeq.theory.fits import *
from leeq.utils import setup_logging
from leeq.utils.compatibility import *
from leeq.utils.compatibility import prims

logger = setup_logging(__name__)


class ConditionalStarkTuneUpRabiXY(experiment):
    EPII_INFO = {
        "name": "ConditionalStarkTuneUpRabiXY",
        "description": "Calibrates ZZ interaction using Rabi-like oscillations in XY plane",
        "purpose": "Performs Hamiltonian tomography to calibrate the ZZ interaction strength and single-qubit Z rotations in conditional Stark shift gates. Uses Rabi-like oscillations while sweeping pulse width to extract interaction parameters.",
        "attributes": {
            "duts": {
                "type": "list[TransmonElement]",
                "description": "List of two qubits [control, target]"
            },
            "frequency": {
                "type": "float",
                "description": "Stark drive frequency (MHz)"
            },
            "amp_control": {
                "type": "float",
                "description": "Stark drive amplitude on control qubit"
            },
            "amp_target": {
                "type": "float",
                "description": "Stark drive amplitude on target qubit"
            },
            "phase": {
                "type": "float",
                "description": "Phase of Stark drives"
            },
            "phase_diff": {
                "type": "float",
                "description": "Phase difference between drives"
            },
            "width": {
                "type": "float",
                "description": "Pulse width"
            },
            "start": {
                "type": "float",
                "description": "Start time (us)"
            },
            "stop": {
                "type": "float",
                "description": "Stop time (us)"
            },
            "step": {
                "type": "float",
                "description": "Time step (us)"
            },
            "rise": {
                "type": "float",
                "description": "Pulse rise time"
            },
            "trunc": {
                "type": "float",
                "description": "Pulse truncation"
            },
            "result": {
                "type": "np.ndarray[complex]",
                "description": "Tomography results for target qubit",
                "shape": "(n_time_points, 2, 2)"
            },
            "result_control": {
                "type": "np.ndarray[complex]",
                "description": "Tomography results for control qubit",
                "shape": "(n_time_points, 2, 2)"
            },
            "fitting_2D": {
                "type": "list[dict]",
                "description": "Fitted parameters for both control states"
            },
            "iz_rate": {
                "type": "unc.ufloat",
                "description": "Single-qubit Z rotation rate (MHz)"
            },
            "zz_rate": {
                "type": "unc.ufloat",
                "description": "ZZ interaction rate (MHz)"
            },
            "iz_from_pulse_rise_drop": {
                "type": "unc.ufloat",
                "description": "IZ phase from pulse rise/drop"
            },
            "zz_from_pulse_rise_drop": {
                "type": "unc.ufloat",
                "description": "ZZ phase from pulse rise/drop"
            }
        },
        "notes": [
            "Performs tomography with control qubit in both ground and excited states",
            "Fits 2D oscillations to extract IZ and ZZ rates",
            "Can use echo sequences to cancel unwanted rotations",
            "Automatically calculates Stark frequency if not provided"
        ]
    }
    @log_and_record
    def run(self, qubits, amp_control, amp_target, frequency=None, rise=0.01, trunc=1.0, start=0, stop=3, step=0.03,
            axis='Y',
            echo=False, iz_rate_cancel=0, phase_diff=0, iz_rise_drop=0):
        """
        Execute conditional Stark tune-up on hardware.

        Parameters
        ----------
        qubits : list
            List of two qubits [control, target].
        amp_control : float
            Stark drive amplitude on control qubit.
        amp_target : float
            Stark drive amplitude on target qubit.
        frequency : float, optional
            Stark drive frequency (MHz). Auto-calculated if None.
        rise : float, optional
            Pulse rise time. Default: 0.01.
        trunc : float, optional
            Pulse truncation. Default: 1.0.
        start : float, optional
            Start time (us). Default: 0.
        stop : float, optional
            Stop time (us). Default: 3.
        step : float, optional
            Time step (us). Default: 0.03.
        axis : str, optional
            Tomography axis. Default: 'Y'.
        echo : bool, optional
            Use echo sequence. Default: False.
        iz_rate_cancel : float, optional
            IZ cancellation rate. Default: 0.
        phase_diff : float, optional
            Phase difference between drives. Default: 0.
        iz_rise_drop : float, optional
            IZ compensation for rise/drop. Default: 0.

        Returns
        -------
        None
            Results stored in instance attributes.
        """
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
        self.trunc = trunc
        self.fitting_2D = None
        self.phase_diff = phase_diff

        if frequency is None:
            freq_01 = qubits[1].get_c1('f01')['X'].freq
            freq_12 = qubits[1].get_c1('f12')['X'].freq

            anharmonicity = freq_01 - freq_12
            self.frequency = freq_01 - 0.3 * anharmonicity
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
                                                  trunc=trunc, zz_interaction_positive=True)

        mprim_control = self.duts[0].get_measurement_prim_intlist(0)
        mprim_target = self.duts[1].get_measurement_prim_intlist(0)

        cs_pulse = c2.get_stark_drive_pulses()
        stark_drive_target_pulse = c2['stark_drive_target']
        stark_drive_control_pulse = c2['stark_drive_control']

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

        lpb = c1_target[
            'Ym'] * lpb_flip_control + lpb + iz_gate + iz_gate_fix + lpb_readout + mprim_target * mprim_control

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

            # print(f"IZ: {self.iz_rate: 0.5f} MHz: {self.zz_rate: 0.5f} MHz")
            # print(f"Phase IZ Contributions from Pulse Rise Drop: {self.iz_from_pulse_rise_drop: 0.5f} rad")

        return {
            'fitting_2D': self.fitting_2D,
            'iz_rate': self.iz_rate,
            'zz_rate': self.zz_rate,
            'iz_from_pulse_rise_drop': self.iz_from_pulse_rise_drop,
            'zz_from_pulse_rise_drop': self.zz_from_pulse_rise_drop
        }

    @register_browser_function(available_after=(run,))
    def plot(self):

        # self.plot_blochsphere()
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
        plt.title("ZZ interaction Hamiltonian tomography - X axis")
        plot_specific_axis(data=self.result[:, 0, 0], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=False)

        plot_specific_axis(data=self.result[:, 1, 0], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=False)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<X>")
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Hamiltonian tomography - Y axis")
        plot_specific_axis(data=self.result[:, 0, 1], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=True)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=True)

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<Y>")
        plt.legend()
        plt.show()

        # self.plot_rescaled_after_fit()

        # self.plot_blochsphere_leakage_to_control()

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
        np.arange(args['start'], args['stop'], args['step'] / 5)

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

            from scipy.fft import fft, fftfreq

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
            xf[max_amplitude_index1]
            amplitudes1[max_amplitude_index1]

            max_amplitude_index2 = np.argmax(amplitudes2)
            xf[max_amplitude_index2]
            amplitudes2[max_amplitude_index2]

            # Print the highest frequency component for both datasets

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
        np.arange(args['start'], args['stop'], args['step'] / 5)

        def plot_specific_axis(data, label, color):
            plt.scatter(t, data, label=label, alpha=0.5, color=color)
            plt.plot(t, data, color=color)

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Leakage to Control Hamiltonian tomography - target X axis")
        plot_specific_axis(data=self.result_control[:, 0, 0], label="Ground", color='#1f77b4')
        plot_specific_axis(data=self.result_control[:, 1, 0], label="Excited", color='#8B0000')

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<Z>")
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Leakage to Control Hamiltonian tomography - target Y axis")
        plot_specific_axis(data=self.result_control[:, 0, 1], label="Ground", color='#1f77b4')
        plot_specific_axis(data=self.result_control[:, 1, 1], label="Excited", color='#8B0000')

        plt.xlabel("Pulse width [us]")
        plt.ylabel("<Z>")
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
    EPII_INFO = {
        "name": "ConditionalStarkTuneUpRepeatedGateXY",
        "description": "Calibrates ZZ interaction using repeated gate sequences",
        "purpose": "Characterizes ZZ interaction by applying repeated conditional Stark gates and performing tomography. This amplifies small errors and provides more accurate calibration of interaction parameters.",
        "attributes": {
            "duts": {
                "type": "list[TransmonElement]",
                "description": "List of two qubits [control, target]"
            },
            "frequency": {
                "type": "float",
                "description": "Stark drive frequency (MHz)"
            },
            "amp_control": {
                "type": "float",
                "description": "Stark drive amplitude on control qubit"
            },
            "amp_target": {
                "type": "float",
                "description": "Stark drive amplitude on target qubit"
            },
            "phase": {
                "type": "float",
                "description": "Phase of Stark drives"
            },
            "width": {
                "type": "float",
                "description": "Gate width"
            },
            "iz_control": {
                "type": "float",
                "description": "IZ rotation on control qubit"
            },
            "iz_target": {
                "type": "float",
                "description": "IZ rotation on target qubit"
            },
            "rise": {
                "type": "float",
                "description": "Pulse rise time"
            },
            "trunc": {
                "type": "float",
                "description": "Pulse truncation"
            },
            "start_gate_number": {
                "type": "int",
                "description": "Starting number of gates"
            },
            "gate_count": {
                "type": "int",
                "description": "Number of gate counts to sweep"
            },
            "result": {
                "type": "np.ndarray[complex]",
                "description": "Tomography results for target qubit",
                "shape": "(gate_count, 2, 2)"
            },
            "result_control": {
                "type": "np.ndarray[complex]",
                "description": "Tomography results for control qubit",
                "shape": "(gate_count, 2, 2)"
            },
            "fitting_2D": {
                "type": "list[dict]",
                "description": "Fitted parameters for both control states"
            },
            "iz_rate": {
                "type": "unc.ufloat",
                "description": "Single-qubit Z rotation rate per gate"
            },
            "zz_rate": {
                "type": "unc.ufloat",
                "description": "ZZ interaction rate per gate"
            },
            "pulse_train": {
                "type": "LogicalPrimitiveBlock",
                "description": "The pulse sequence used"
            },
            "pulse_count": {
                "type": "range",
                "description": "Range of pulse counts used"
            },
            "N": {
                "type": "int",
                "description": "Number of repetitions"
            }
        },
        "notes": [
            "Sweeps number of repeated gates instead of pulse width",
            "More sensitive to small interaction rates",
            "Uses frequency bounds in fitting for better convergence",
            "Can detect and calibrate gate errors accumulated over multiple operations"
        ]
    }

    @log_and_record
    def run(self, duts, amp_control, amp_target, frequency, phase=0, rise=0.01, trunc=1.0, axis='Y',
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
        self.trunc = trunc
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
            trunc=trunc,
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

        t_start = self.start_gate_number
        t_stop = self.start_gate_number + self.gate_count
        t_step = 1

        np.arange(t_start, t_stop, t_step)

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
        args = self._get_run_args_dict()

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
        plt.title("ZZ interaction repeated gate tomography - X axis")

        plot_specific_axis(data=self.result[:, 0, 0], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=False)
        plot_specific_axis(data=self.result[:, 1, 0], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=False)

        plt.xlabel("Pulse count")
        plt.ylabel("<X>")
        plt.legend()
        plt.xticks(xticks_subset)

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction repeated gate tomography - Y axis")

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

        self.plot_leakage_to_control()

    def plot_rescaled_after_fit(self):
        args = self._get_run_args_dict()

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
        plt.title("ZZ interaction rescaled repeated gate tomography - X axis")

        plot_specific_axis(data=self.result[:, 0, 0], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=False, color=dark_navy)
        plot_specific_axis(data=self.result[:, 1, 0], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=False, color=dark_purple)

        plt.xlabel("Pulse count")
        plt.ylabel("<X>")
        plt.legend()
        plt.xticks(xticks_subset)

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction rescaled repeated gate tomography - Y axis")

        plot_specific_axis(data=self.result[:, 0, 1], label="Ground", fit_params=self.fitting_2D[0],
                           use_imaginary_part=True)
        plot_specific_axis(data=self.result[:, 1, 1], label="Excited", fit_params=self.fitting_2D[1],
                           use_imaginary_part=True)

    def plot_leakage_to_control(self):

        args = self._get_run_args_dict()

        t = np.arange(args['start_gate_number'], args['start_gate_number'] + args['gate_count'], 1)

        def plot_specific_axis(data, label, color):
            plt.scatter(t, data, label=label, alpha=0.5, color=color)
            plt.plot(t, data, color=color)

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Leakage to Control Hamiltonian tomography - target X axis")
        plot_specific_axis(data=self.result_control[:, 0, 0], label="Ground", color='#1f77b4')
        plot_specific_axis(data=self.result_control[:, 1, 0], label="Excited", color='#8B0000')

        plt.xlabel("Pulse count")
        plt.ylabel("<Z>")
        plt.legend()
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.title("ZZ interaction Leakage to Control Hamiltonian tomography - target Y axis")
        plot_specific_axis(data=self.result_control[:, 0, 1], label="Ground", color='#1f77b4')
        plot_specific_axis(data=self.result_control[:, 1, 1], label="Excited", color='#8B0000')

        plt.xlabel("Pulse count")
        plt.ylabel("<Z>")
        plt.legend()
        plt.show()


class ConditionalStarkEchoTuneUp(Experiment):
    EPII_INFO = {
        "name": "ConditionalStarkEchoTuneUp",
        "description": "Iterative calibration of conditional Stark gates using echo sequences",
        "purpose": "Performs iterative calibration of conditional Stark shift gates using echo sequences to cancel unwanted single-qubit rotations. Uses Kalman filtering for optimal parameter estimation across multiple iterations.",
        "attributes": {
            "duts": {
                "type": "list[TransmonElement]",
                "description": "List of two qubits [control, target]"
            },
            "n_max_iteration": {
                "type": "int",
                "description": "Maximum number of iterations"
            },
            "params": {
                "type": "dict",
                "description": "Initial gate parameters"
            },
            "frequency": {
                "type": "float",
                "description": "Stark drive frequency (MHz)"
            },
            "amp_control": {
                "type": "float",
                "description": "Control qubit amplitude"
            },
            "phase_diff": {
                "type": "float",
                "description": "Phase difference between drives"
            },
            "rise": {
                "type": "float",
                "description": "Pulse rise time"
            },
            "trunc": {
                "type": "float",
                "description": "Pulse truncation"
            },
            "iz_control_history": {
                "type": "list[float]",
                "description": "History of control IZ rates"
            },
            "iz_target_history": {
                "type": "list[float]",
                "description": "History of target IZ rates"
            },
            "zz_history": {
                "type": "list[float]",
                "description": "History of ZZ interaction rates"
            },
            "width_history": {
                "type": "list[float]",
                "description": "History of gate widths"
            },
            "kalman_iz_control": {
                "type": "KalmanFilter1D",
                "description": "Kalman filter for control IZ"
            },
            "kalman_iz_target": {
                "type": "KalmanFilter1D",
                "description": "Kalman filter for target IZ"
            },
            "kalman_zz": {
                "type": "KalmanFilter1D",
                "description": "Kalman filter for ZZ rate"
            },
            "best_params": {
                "type": "dict",
                "description": "Optimized gate parameters"
            }
        },
        "notes": [
            "Uses iterative refinement with Kalman filtering",
            "Echo sequences cancel single-qubit phase accumulation",
            "Automatically determines optimal gate width for target ZZ angle",
            "Can update either IZ or ZZ parameters selectively"
        ]
    }

    @log_and_record
    def run(self, duts, params=None, frequency=None, amp_control=None, phase_diff=0, rise=0.01, trunc=1.0,
            t_start=0, t_stop=20, sweep_points=40,
            n_start=0, n_stop=21, update_iz=False, update_zz=True, n_max_iteration=20
            ):
        """
        Execute the experiment on hardware.

        Parameters
        ----------
        duts : list[TransmonElement]
            List of two qubits [control, target].
        params : dict, optional
            Initial gate parameters. Default: None (auto-calculated)
        frequency : float, optional
            Stark drive frequency (MHz). Default: None
        amp_control : float, optional
            Control qubit amplitude. Default: None
        phase_diff : float, optional
            Phase difference between drives. Default: 0
        rise : float, optional
            Pulse rise time. Default: 0.01
        trunc : float, optional
            Pulse truncation. Default: 1.0
        t_start : float, optional
            Start time for sweep. Default: 0
        t_stop : float, optional
            Stop time for sweep. Default: 20
        sweep_points : int, optional
            Number of sweep points. Default: 40
        n_start : int, optional
            Start echo count. Default: 0
        n_stop : int, optional
            Stop echo count. Default: 21
        update_iz : bool, optional
            Update single-qubit Z rates. Default: False
        update_zz : bool, optional
            Update ZZ interaction rate. Default: True
        n_max_iteration : int, optional
            Maximum iterations. Default: 20

        Returns
        -------
        None
            Results are stored in instance attributes.
        """
        self.duts = duts
        self.n_max_iteration = n_max_iteration

        assert not update_iz

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

        # Creating a dataframe
        df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])

        # Formatting the float values to three decimal places
        df['Value'] = df['Value'].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

        # Display the dataframe
        display(df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
            'selector': 'th',
            'props': [('text-align', 'center')]
        }]))

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
            trunc=self.current_params['trunc'],
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

        for _i in range(self.n_max_iteration):
            repeated_gate = ConditionalStarkTuneUpRepeatedGateXY(
                duts=self.duts,
                iz_control=0,
                iz_target=iz_target,
                frequency=self.current_params['frequency'],
                amp_control=self.current_params['amp_control'],
                amp_target=self.current_params['amp_target'],
                rise=self.current_params['rise'],
                trunc=self.current_params['trunc'],
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
                kalman_zz.predict(movement=zz_diff,
                                  position_variance=(zz_rate.std_dev * width_diff * np.pi * 2) ** 2)
                kalman_iz.predict(movement=iz_diff,
                                  position_variance=(zz_rate.std_dev * width_diff * np.pi * 2) ** 2)

                estimated_iz_list.append(unc.ufloat(kalman_iz.x, np.sqrt(kalman_iz.P)))
                estimated_zz_list.append(unc.ufloat(kalman_zz.x, np.sqrt(kalman_zz.P)))

            zz_accuracy_check = np.abs(target_zz - zz_pgc) < 1e-3
            zz_uncertainty_check = np.sqrt(kalman_zz.P) < 1e-3
            zz_check_pass = zz_accuracy_check and zz_uncertainty_check



            if (iz_check_pass or not update_iz) and (zz_check_pass or not update_zz):
                break

        new_params = self.current_params.copy()

        new_params['iz_target'] = iz_target
        new_params['width'] = width


        self.params_list.append(new_params)
        self.current_params = new_params

        self.estimated_iz_list = estimated_iz_list
        self.estimated_zz_list = estimated_zz_list
