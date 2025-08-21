import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
from k_agents.inspection.decorator import text_inspection, visual_inspection
from scipy import optimize as so

from leeq import Experiment, ExperimentManager, Sweeper, setup
from leeq.chronicle import log_and_record, register_browser_function
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import SweepParametersSideEffectFactory
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.utils import setup_logging

logger = setup_logging(__name__)

__all__ = [
    'ResonatorSweepTransmissionWithExtraInitialLPB',
    'ResonatorSweepTransmissionWithExtraInitialLPB',
    'ResonatorSweepAmpFreqWithExtraInitialLPB',
    'ResonatorSweepTransmissionXiComparison'
]


class ResonatorSweepTransmissionWithExtraInitialLPB(Experiment):
    """
    Class representing a resonator sweep transmission experiment with extra initial LPB.
    Inherits from a generic "experiment" class.
    """

#     _experiment_result_analysis_instructions = """
# Inspect the plot to detect the resonator's presence. If present:
# 1. Consider the resonator linewidth (typically sub-MHz to a few MHz).
# 2. If the step size is much larger than the linewidth:
#    a. Focus on the expected resonator region.
#    b. Reduce the step size for better accuracy.
# 3. If linewidth < 0.1 MHz, it's likely not a resonator; move on.
# The experiment is considered successful if a resonator is detected. Otherwise, it is considered unsuccessful and suggest
# a new sweeping range and step size.
#     """
#
    @log_and_record
    def run(self,
            dut_qubit: TransmonElement,
            start: float = 8000,
            stop: float = 9000,
            step: float = 5.0,
            num_avs: int = 5000,
            rep_rate: float = 0.0,
            mp_width: float = 8,
            initial_lpb=None,
            amp: float = 0.02) -> None:
        """
        Run the resonator sweep transmission experiment. Usually used to find the resonator.

        The initial lpb is for exciting the qubit to a different state.

        Parameters:
            dut_qubit: The device under test (DUT) qubit.
            start (float): Start frequency for the sweep. Unit: MHz Default is 8000.
            stop (float): Stop frequency for the sweep. Unit: MHz Default is 9000.
            step (float): Frequency step for the sweep. Unit: MHz Default is 5.0.
            num_avs (int): Number of averages. Default is 1000.
            rep_rate (float): Repetition rate. Default is 0.0.
            mp_width (float): Measurement pulse width. Unit us.:w If None, uses rep_rate. Default is None.
            initial_lpb: Initial linear phase behavior (LPB). Default is None.
            amp (float): Amplitude. Default is 0.02.
        """
        # Sweep the frequency
        mp = dut_qubit.get_default_measurement_prim_intlist().clone()

        # Update pulse width
        mp.update_pulse_args(
            width=rep_rate) if mp_width is None else mp.update_pulse_args(
            width=mp_width)
        if amp is not None:
            mp.update_pulse_args(amp=amp)

        # Clear the transform function to get the raw data
        mp.set_transform_function(None)

        # Save the mp for live plots
        self.mp = mp

        lpb = initial_lpb + mp if initial_lpb is not None else mp

        # Define sweeper
        swp = Sweeper(
            np.arange,
            n_kwargs={
                "start": start,
                "stop": stop,
                "step": step},
            params=[
                SweepParametersSideEffectFactory.func(
                    mp.update_freq,
                    {},
                    "freq",
                    name='frequency')],
        )

        # Perform the experiment
        with ExperimentManager().status().with_parameters(
                shot_number=num_avs,
                shot_period=rep_rate,
                acquisition_type='IQ_average'
        ):
            ExperimentManager().run(lpb, swp)

        result = np.squeeze(mp.result())
        self.data = result

        # Save results
        self.result = {
            "Magnitude": np.absolute(result),
            "Phase": np.angle(result),
        }

    @log_and_record(overwrite_func_name='ResonatorSweepTransmissionWithExtraInitialLPB.run')
    def run_simulated(self,
                      dut_qubit: TransmonElement,
                      start: float = 8000,
                      stop: float = 9000,
                      step: float = 5.0,
                      num_avs: int = 1000,
                      rep_rate: float = 0.0,
                      mp_width: float = None,
                      initial_lpb=None,
                      amp: float = 0.02) -> None:
        """
        Run the resonator sweep transmission experiment.

        The initial lpb is for exciting the qubit to a different state.

        Parameters:
            dut_qubit: The device under test (DUT) qubit.
            start (float): Start frequency for the sweep. Default is 8000.
            stop (float): Stop frequency for the sweep. Default is 9000.
            step (float): Frequency step for the sweep. Default is 5.0.
            num_avs (int): Number of averages. Default is 1000.
            rep_rate (float): Repetition rate. Default is 10.0.
            mp_width (float): Measurement pulse width. If None, uses rep_rate. Default is None.
            initial_lpb: Initial linear phase behavior (LPB). Default is None.
            amp (float): Amplitude. Default is 1.0.
        """

        if initial_lpb is not None:
            logger.warning("initial_lpb is ignored in the simulated mode.")

        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()
        virtual_transmon = simulator_setup.get_virtual_qubit(dut_qubit)

        mprim = dut_qubit.get_default_measurement_prim_intlist()

        omega_per_amp = simulator_setup.get_omega_per_amp(mprim.channel)  # MHz
        effective_amp = amp * omega_per_amp

        f = np.arange(start, stop, step)

        response = virtual_transmon.get_resonator_response(
            f, effective_amp, baseline=0.001 * effective_amp
        )[0, :]

        noise_scale = 1 / num_avs

        noise = (np.random.normal(0, noise_scale, response.shape)
                 + 1j * np.random.normal(0, noise_scale, response.shape))

        slope = np.random.normal(-0.1, 0.01)

        phase_slope = np.exp(1j * 2 * np.pi * slope * (f - start))

        response = response * phase_slope + noise

        self.data = response

        # Save results
        self.result = {
            "Magnitude": np.absolute(response),
            "Phase": np.angle(response),
        }

    def live_plots(self, step_no: tuple[int] = None):
        """
        Generate the live plots. This function is called by the live monitor.
        The step no denotes the number of data points to plot, while the
        buffer size is the total number of data points to plot. Some of the data
        in the buffer is note yet valid, so they should not be plotted.
        """

        from plotly.subplots import make_subplots

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        traces = self._get_basic_plot_traces(step_no)

        fig.add_trace(traces['Magnitude'], row=1, col=1)
        fig.add_trace(traces['Phase'], row=2, col=1)
        fig.add_trace(traces['Phase Gradient'], row=3, col=1)

        fig.update_layout(
            title="Resonator spectroscopy live plot",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Magnitude",
            plot_bgcolor="white",
        )

        return fig

    def _get_basic_plot_traces(self, step_no: tuple[int] = None):
        """
        Generate the basic plots, of mangitude and phase and phase gradient.

        Parameters:
            step_no (tuple[int]): Optional. The step number. When not specified, all data will be plotted.

        Returns:
            (Any): The figure.
        """

        # Get the sweep parameters
        args = self._get_run_args_dict()

        # Get the data
        result = self.data
        f = np.arange(args["start"], args["stop"], args["step"])

        if step_no is not None:
            # For the sweep defined above, the step_no is a tuple of one
            # element, the current frequency steps
            valid_data_n = step_no[0]
            result = result[: valid_data_n]
            f = f[:valid_data_n]

        unwrapped_phase = np.unwrap(np.angle(result))

        data = {
            "Magnitude": (f, np.absolute(result)),
            "Phase": (f, unwrapped_phase),
            "Phase Gradient": ((f[:-1] + f[1:]) / 2, np.gradient(unwrapped_phase))
        }

        traces = dict([
            (name, go.Scatter(
                x=data[name][0],
                y=data[name][1],
                mode="lines",
                name=name))
            for name in data
        ])

        return traces

    @staticmethod
    def root_lorentzian(
            f: float,
            f0: float,
            Q: float,
            amp: float,
            baseline: float) -> float:
        """
        Calculate the root of the Lorentzian function.

        Parameters:
        f (float): The frequency at which the Lorentzian is evaluated.
        f0 (float): The resonant frequency (i.e., the peak position).
        Q (float): The quality factor which determines the width of the peak.
        amp (float): Amplitude of the Lorentzian peak.
        baseline (float): Baseline offset of the Lorentzian.

        Returns:
        float: The absolute value of the root Lorentzian function at frequency f.

        Note:
        The Lorentzian function is given by:
            L(f) = amp / [1 + 2jQ(f - f0)/f0] + baseline
        Where:
            - j is the imaginary unit.
            - The root Lorentzian is obtained by taking the absolute value.
        """

        # Compute the Lorentzian function
        lorentzian = np.abs(amp / (1 + (2j * Q * (f - f0) / f0))) + baseline

        return lorentzian

    def _fit_phase_gradient(self):
        """
        Fit the phase gradient to a Lorentzian function.

        Returns:
            z, f0, Q, amp, baseline, direction (tuple): The phase gradient, the resonant frequency, the quality factor,
            the amplitude, the baseline, and the direction of the Lorentzian peak.
        """
        args = self._get_run_args_dict()
        f = np.arange(args["start"], args["stop"], args["step"])
        phase_trace = self.result["Phase"]
        phase_unwrapped = np.unwrap(phase_trace)

        def leastsq(x, f, z):
            """
            Least square function for fitting the phase gradient to a Lorentzian function.

            Parameters:
               x (list): List of parameters to fit.
               f (float): The frequency at which the Lorentzian is evaluated.
               z (float): The phase gradient.

            Returns:
               float: The sum of the square of the difference between the phase gradient and the Lorentzian function.
            """
            f0, Q, amp, baseline = x
            fit = self.root_lorentzian(f, f0, Q, amp, baseline)
            return np.sum((fit - z) ** 2)

        # Find the gradient per step
        z = (phase_unwrapped[1:] - phase_unwrapped[:-1]) / args["step"]

        z_balanced = z - z.mean()
        # Find the direction of lorentzian peak
        direction = 1 if np.abs(
            np.max(z_balanced)) > np.abs(
            np.min(z_balanced)) else -1

        # Find the frequency for the gradient
        f = (f[:-1] + f[1:]) / 2

        # Find the initial guess for the parameters
        f0, amp, baseline = (
            f[np.argmax((z) * direction)],  # Peak with max
            max(z) - min(z),
            # Amplitude by finding the difference between max and min
            # Find the baseline by finding the effective min. If the lorentzian
            # is negative,
            min(z * direction),
            # the baseline is the max. If the lorentzian is positive, the
            # baseline is the min.
        )

        # Find the initial guess for Q
        half_cut = z * direction - baseline - amp / 2

        # find another derivative to estimate the half line width
        f_diff = (f[:-1] + f[1:]) / 2
        turn_point = np.argwhere(half_cut[:-1] * half_cut[1:] < 0)

        kappa_guess = f_diff[turn_point[1]] - f_diff[turn_point[0]]

        Q_guess = f0 / kappa_guess
        if isinstance(Q_guess, np.ndarray):
            Q_guess = Q_guess[0]
        if isinstance(Q_guess, list):
            Q_guess = Q_guess[0]

        # Finally, we can fit the data
        result = so.minimize(
            leastsq,
            np.array([f0, Q_guess, amp, baseline * direction], dtype=object),
            args=(f, z * direction),
            tol=1.0e-20,
        )  # method='Nelder-Mead',
        f0, Q, amp, baseline = result.x
        baseline = baseline * direction

        return z, f0, Q, amp, baseline, direction

    @register_browser_function(available_after=(run,))
    @visual_inspection("""
    Analyze a new resonator spectroscopy magnitude plot to determine if it shows evidence of a resonator. Focus on:
    1. Sharp dips or peaks at specific frequencies
    2. Signal stability
    3. Noise levels
    4. Behavior around suspected resonant frequencies
    Provide a detailed analysis of the magnitude and frequency data. Identifying a resonator indicates a successful experiment.
    """)
    def plot_magnitude(self):
        args = self._get_run_args_dict()
        f = np.arange(args["start"], args["stop"], args["step"])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=f,
                y=self.result["Magnitude"],
                mode="lines",
                name="Magnitude"))

        fig.update_layout(
            title="Resonator spectroscopy magnitude",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Magnitude",
            plot_bgcolor="white",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_phase(self):

        fig = go.Figure()

        traces = self._get_basic_plot_traces()

        del traces['Magnitude']

        fig.add_traces(list(traces.values()))

        fig.update_layout(
            title="Resonator spectroscopy phase plot",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Magnitude",
            plot_bgcolor="white",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_phase_gradient_fit(self):

        fit_succeed = False

        try:
            z, f0, Q, amp, baseline, direction = self._fit_phase_gradient()
            fit_succeed = True
        except Exception as e:
            logger.error(f"Error fitting phase gradient: {e}")
            args = self._get_run_args_dict()
            f = np.arange(args["start"], args["stop"], args["step"])
            phase_trace = self.result["Phase"]
            phase_unwrapped = np.unwrap(phase_trace)
            z = (phase_unwrapped[1:] - phase_unwrapped[:-1]) / args["step"]

        args = self._get_run_args_dict()
        f = np.arange(args["start"], args["stop"], args["step"])
        f_interpolate = np.arange(
            args["start"],
            args["stop"],
            args["step"] / 5)

        fig = go.Figure()

        if fit_succeed:
            fig.add_trace(
                go.Scatter(
                    x=f_interpolate,
                    y=self.root_lorentzian(
                        f_interpolate,
                        f0,
                        Q,
                        amp,
                        baseline)
                    * direction,
                    mode="lines",
                    name="Lorentzian fit",
                ))

        fig.add_trace(
            go.Scatter(
                x=f,
                y=z,
                mode="markers",
                name="Phase gradient"))

        fig.update_layout(
            title="Resonator spectroscopy phase gradient fitting",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Phase",
            plot_bgcolor="white",
        )

        if fit_succeed:
            print(
                "Phase gradient fit f0:%s, Q:%s, amp:%s, base:%s kappa:%f"
                % (f0, Q, amp, baseline, f0 / Q)
            )

        return fig

    @text_inspection
    def fitting(self) -> str:
        """
        Get the analyzed result prompt.

        Returns:
            str: The analyzed result prompt.
        """

        try:
            z, f0, Q, amp, baseline, direction = self._fit_phase_gradient()
            fit_succeed = True
        except Exception as e:
            return f"The experiment has an error fitting phase gradient, implying the experiment is failed."

        return ("The fitting suggest that the resonant frequency is at %f MHz, "
                "with a quality factor of %f (resonator linewidth kappa of %f MHz), an amplitude of %f, and a baseline of %f.") % (
            f0, Q, f0 / Q, amp, baseline)


class ResonatorSweepAmpFreqWithExtraInitialLPB(Experiment):
    @log_and_record
    def run(self,
            dut_qubit: TransmonElement,
            start: float = 8000,
            stop: float = 9000,
            step: float = 5.,
            num_avs: int = 200,
            rep_rate: float = 0.,
            mp_width: Optional[float] = 8,
            initial_lpb: LogicalPrimitiveBlock = None,
            amp_start: float = 0,
            amp_stop: float = 1,
            amp_step: float = 0.05) -> None:
        """
        Run an experiment by sweeping the frequency and amplitude of a qubit. Do not use this
        if you are not sweeping the amplitude.

        Parameters:
        - dut_qubit: The qubit under test.
        - start (float): The starting frequency for the sweep. In MHz
        - stop (float): The stopping frequency for the sweep. In MHz
        - step (float): The step size between frequencies in the sweep. In MHz
        - num_avs (int): The number of averages to take.
        - rep_rate (float): Repetition rate of the experiment.
        - mp_width (Optional[float]): Measurement primitive width. If None, `rep_rate` is used.
        - initial_lpb: Initial LPB to be added to the delay and measurement primitive.
        - update (bool): Flag to decide whether to update parameters or not.
        - amp_start (float): The starting amplitude for the amplitude sweep.
        - amp_stop (float): The stopping amplitude for the amplitude sweep.
        - amp_step (float): The step size between amplitudes in the amplitude sweep.

        Returns:
        None
        """
        # Get the original measurement primitive.
        mprim_index = '0'
        mp = dut_qubit.get_measurement_prim_intlist(mprim_index).clone()
        original_freq = mp.freq

        # Update the pulse arguments with either the provided mp_width or
        # rep_rate if mp_width is None.
        mp.update_pulse_args(
            width=mp_width if mp_width is not None else rep_rate)

        # Remove any previous transform functions.
        mp.set_transform_function(None)

        self.mp = mp

        lpb = mp

        # If initial_lpb is provided, concatenate it with delay and mp.
        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        # Define the frequency sweeper using np.arange and updating mp's
        # frequency.
        swp_freq = Sweeper(
            np.arange,
            n_kwargs={
                "start": start,
                "stop": stop,
                "step": step},
            params=[
                SweepParametersSideEffectFactory.func(
                    mp.update_freq,
                    {},
                    "freq")],
        )

        # Define the amplitude sweeper using np.arange and updating mp's
        # amplitude.
        swp_amp = Sweeper(
            np.arange,
            n_kwargs={
                'start': amp_start,
                'stop': amp_stop,
                'step': amp_step},
            params=[
                SweepParametersSideEffectFactory.func(
                    mp.update_pulse_args,
                    {},
                    'amp')])

        # Perform the experiment with specified setup parameters.
        with ExperimentManager().status().with_parameters(
                shot_number=num_avs,
                shot_period=rep_rate,
                acquisition_type='IQ_average'
        ):
            ExperimentManager().run(lpb, swp_freq + swp_amp)

        # Save the result in trace attribute, transposing for further analysis.
        self.trace = np.squeeze(mp.result()).transpose()

    def _plot_data(self, x, y, z, title):
        """
        Plot the magnitude of the resonator response.

        Parameters:
            x (np.ndarray): The x-axis data.
            y (np.ndarray): The y-axis data.
            z (np.ndarray): The z-axis data.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale='Viridis')
        )

        fig.update_layout(
            title=title,
            xaxis_title="Frequency [MHz]",
            yaxis_title="Driving Amplitude [a.u.]",
        )

        return fig

    @register_browser_function(available_after=(run,))
    def plot_magnitude(self):
        """
        Plot the magnitude of the resonator response.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        args = self._get_run_args_dict()
        trace = np.squeeze(self.mp.result()).transpose()

        return self._plot_data(
            x=np.arange(
                start=args['start'],
                stop=args['stop'],
                step=args['step']),
            y=np.arange(
                start=args['amp_start'],
                stop=args['amp_stop'],
                step=args['amp_step']),
            z=np.abs(trace),
            title="Resonator response magnitude")

    @register_browser_function(available_after=(run,))
    def plot_phase(self):
        """
        Plot the phase of the resonator response.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """

        args = self._get_run_args_dict()
        trace = np.squeeze(self.mp.result()).transpose()

        return self._plot_data(
            x=np.arange(
                start=args['start'],
                stop=args['stop'],
                step=args['step']),
            y=np.arange(
                start=args['amp_start'],
                stop=args['amp_stop'],
                step=args['amp_step']),
            z=np.unwrap(
                np.angle(trace)),
            title="Resonator response phase")

    @register_browser_function(available_after=(run,))
    def plot_phase_gradient(self):
        """
        Plot the phase gradient of the resonator response.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        args = self._get_run_args_dict()
        trace = np.squeeze(self.mp.result()).transpose()

        return self._plot_data(
            x=np.arange(
                start=args['start'],
                stop=args['stop'],
                step=args['step']),
            y=np.arange(
                start=args['amp_start'],
                stop=args['amp_stop'],
                step=args['amp_step']),
            z=np.gradient(
                np.unwrap(
                    np.angle(trace)),
                axis=1),
            title="Resonator response phase gradient")

    @register_browser_function(available_after=(run,))
    def plot_mag_logscale(self):
        """
        Plot the magnitude of the resonator response in log scale.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        args = self._get_run_args_dict()
        trace = np.squeeze(self.mp.result()).transpose()
        return self._plot_data(
            x=np.arange(
                start=args['start'], stop=args['stop'], step=args['step']), y=np.arange(
                start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step']), z=np.log(
                np.abs(trace)), title="Resonator response magnitude (log scale)")

    def live_plots(self, step_no: tuple[int] = None):
        """
        Generate the live plots. This function is called by the live monitor.
        The step no denotes the number of data points to plot, while the
        buffer size is the total number of data points to plot. Some of the data
        in the buffer is note yet valid, so they should not be plotted.
        """

        return self.plot_phase_gradient()


class ResonatorSweepTransmissionXiComparison(Experiment):
    """
    Class for comparing resonator sweep transmission with extra initial logical primitive block (LPB).
    It includes methods to run the experiment, and to plot magnitude and phase using both
    matplotlib and plotly.
    """

    @log_and_record
    def run(self,
            dut_qubit: Any,
            lpb_scan: Union[List, Tuple, Dict],
            start: float = 8000,
            stop: float = 9000,
            step: float = 5.,
            num_avs: int = 5000,
            rep_rate: float = 0.,
            mp_width: Optional[float] = 8,
            amp: Optional[float] = None) -> None:
        """
        Runs the resonator sweep transmission experiment and compare the response from the different state of the qubit.
        Not used for resonator discovery.

        Parameters:
            dut_qubit: The device under test (DUT) qubit.
            lpb_scan: The LPB to be scanned.
            start (float): Start frequency for the sweep. Default is 8000.
            stop (float): Stop frequency for the sweep. Default is 9000.
            step (float): Frequency step for the sweep. Default is 5.0.
            res_power (float): Power of the resonator. Default is 15.0.
            num_avs (int): Number of averages. Default is 1000.
            rep_rate (float): Repetition rate. Default is 10.0.
            mp_width (float): Measurement pulse width. If None, uses rep_rate. Default is None.
            amp (float): Amplitude. Default is None.
        """
        if isinstance(lpb_scan, (tuple, list)):
            lpb_scan = dict(enumerate(lpb_scan))

        self.result_dict = {
            key: ResonatorSweepTransmissionWithExtraInitialLPB(
                dut_qubit=dut_qubit, start=start, stop=stop, step=step,
                num_avs=num_avs, rep_rate=rep_rate,
                mp_width=mp_width, initial_lpb=lpb, amp=amp
            ) for key, lpb in lpb_scan.items()
        }

    @register_browser_function(available_after=(run,))
    def plot_magnitude_plotly(self) -> None:
        """
        Plots the magnitude of the resonator spectroscopy using Plotly.
        """
        args = self._get_run_args_dict()
        f = np.arange(args['start'], args['stop'], args['step'])

        fig = go.Figure()

        for key, sweep in self.result_dict.items():
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=sweep.result['Magnitude'],
                    mode='lines',
                    name=key))

        fig.update_layout(
            title='Resonator spectroscopy magnitude',
            xaxis_title='Frequency [MHz]',
            yaxis_title='Magnitude',
            plot_bgcolor='white')

        fig.show()

    @register_browser_function(available_after=(run,))
    def plot_phase_plotly(self) -> None:
        """
        Plots the phase of the resonator spectroscopy using Plotly.
        """
        args = self._get_run_args_dict()
        f = np.arange(args['start'], args['stop'], args['step'])

        fig = go.Figure()

        for key, sweep in self.result_dict.items():
            phase_trace = sweep.result['Phase']
            phase_trace_mod = sweep.UnwrapPhase(phase_trace)
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=phase_trace_mod,
                    mode='lines',
                    name=key))

        fig.update_layout(
            title='Resonator spectroscopy phase',
            xaxis_title='Frequency [MHz]',
            yaxis_title='Phase',
            plot_bgcolor='white')

        fig.show()

    @register_browser_function(available_after=(run,))
    def plot_phase_diff_fit_plotly(self) -> None:
        """
        Plots the differentiated phase and its Lorentzian fit using Plotly.
        """
        args = self._get_run_args_dict()
        f = np.arange(args['start'], args['stop'], args['step'])
        f_interpolate = np.arange(
            args['start'],
            args['stop'],
            args['step'] / 5)

        fig = go.Figure()

        for key, sweep in self.result_dict.items():
            z, f0, Q, amp, baseline, direction = sweep.fit_phase_diff()
            lorentzian_fit = sweep.root_lorentzian(
                f_interpolate, f0, Q, amp, baseline) * direction

            fig.add_trace(
                go.Scatter(
                    x=f_interpolate,
                    y=lorentzian_fit,
                    mode='lines',
                    name=f'{key} Lorentzian fit'))
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=z,
                    mode='markers',
                    name=f'{key} Phase derivative'))

            print(
                f'Phase diff fit {key} f0:{f0}, Q:{Q}, amp:{amp}, base:{baseline}, kappa:{f0 / Q}')

        fig.update_layout(
            title='Resonator spectroscopy phase fitting',
            xaxis_title='Frequency [MHz]',
            yaxis_title='Phase',
            plot_bgcolor='white')

        fig.show()


# Assuming other necessary modules are imported elsewhere in the project.

class MeasurementScanParams(Experiment):
    """
    Class for managing and executing measurement scan parameters
    in an experimental setup.
    """

    @log_and_record
    def run(self, dut, sweep_lpb_list, mprim_index: int,
            amp_scan: dict = None, freq_scan: dict = None,
            accumulate_snr_for_all_distinguishable_state: bool = True,
            disable_sub_plot: bool = True):
        """
        Execute the measurement scan with given parameters.

        Args:
            dut: Device under test.
            sweep_lpb_list: List of sweep parameters.
            mprim_index (int): Measurement primitive index.
            amp_scan (dict, optional): Parameters for amplitude scan.
            freq_scan (dict, optional): Parameters for frequency scan.
            accumulate_snr_for_all_distinguishable_state (bool, optional):
                Flag to accumulate SNR for all distinguishable states.
            disable_sub_plot (bool, optional): Flag to disable subplot.
        """
        # Initialize lists to store scan results
        self.snrs = []
        self.scanned_freqs = []
        self.scanned_amps = []
        self.measurement_scan_result = []

        # Get measurement primitives
        mprim = dut.get_measurement_prim_intlist(mprim_index)

        # Set scanned frequencies and amplitudes
        self.scanned_freqs = [
            mprim.freq] if freq_scan is None else np.arange(
            **freq_scan)
        self.scanned_amps = [
            mprim.primary_kwargs()['amp']] if amp_scan is None else np.arange(
            **amp_scan)

        # Check for plot settings in Jupyter
        plot_result_in_jupyter = setup().status().get_param("Plot_Result_In_Jupyter")
        if disable_sub_plot:
            setup().status().set_param("Plot_Result_In_Jupyter", False)

        from leeq.experiments.builtin import MeasurementCalibrationMultilevelGMM

        # Perform measurement scan
        for freq in self.scanned_freqs:
            for amp in self.scanned_amps:
                result = MeasurementCalibrationMultilevelGMM(
                    dut=dut,
                    sweep_lpb_list=sweep_lpb_list,
                    mprim_index=mprim_index,
                    freq=freq,
                    amp=amp)

                snr = 1 / np.sum([1 / (x + 1e-6) for x in result.snr.values()]) \
                    if accumulate_snr_for_all_distinguishable_state else result.SNR[(mprim_index, mprim_index + 1)]

                self.snrs.append(snr)
                self.measurement_scan_result.append(result)

        # Restore plot settings
        setup().status().set_param("Plot_Result_In_Jupyter", plot_result_in_jupyter)
        self.snrs = np.asarray(self.snrs).reshape(
            [len(self.scanned_freqs), len(self.scanned_amps)])

    # Additional methods follow the same pattern of revision.
    # ...
    @register_browser_function(available_after=(run,))
    def plot_snr_vs_freq(self):
        """
        Plots Signal-to-Noise Ratio (SNR) versus frequency.
        """
        if len(self.scanned_freqs) == 1:
            return
        if len(self.scanned_amps) > 1:
            return
        plt.figure()
        plt.title("SNR vs Frequency")
        plt.xlabel('Resonator Frequency')
        plt.ylabel('SNR')
        plt.plot(self.scanned_freqs, np.asarray(self.snrs).flatten())
        plt.grid()
        plt.show()

    @register_browser_function(available_after=(run,))
    def plot_snr_vs_amp(self):
        """
        Plots Signal-to-Noise Ratio (SNR) versus amplitude.
        """
        if len(self.scanned_amps) == 1:
            return
        if len(self.scanned_freqs) > 1:
            return

        plt.figure()
        plt.title("SNR vs Amplitude")
        plt.xlabel('Driving Amplitude')
        plt.ylabel('SNR')
        plt.plot(self.scanned_amps, np.asarray(self.snrs).flatten())
        plt.grid()
        plt.show()

    @register_browser_function(available_after=(run,))
    def plot_snr_vs_amp_freq(self, fig_size=(10, 10)):
        """
        Plots Signal-to-Noise Ratio (SNR) versus both amplitude and frequency.

        Args:
            fig_size (tuple, optional): Figure size. Defaults to (10, 10).
        """

        if len(self.scanned_freqs) == 1 or len(self.scanned_amps) == 1:
            return

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title("SNR vs Amplitude / Frequency")
        cax = ax.imshow(
            np.asarray(
                self.snrs),
            aspect='auto',
            interpolation='nearest')

        # Adding text annotation inside the cells
        for i in range(len(self.scanned_freqs)):
            for j in range(len(self.scanned_amps)):
                text = ax.text(j, i, f"{self.snrs[i, j]:.2f}",
                               ha="center", va="center", color="w")

        # set ticks
        ax.set_xticks(ticks=np.arange(len(self.scanned_amps)),
                      labels=[f"{x:.2f}" for x in self.scanned_amps])
        ax.set_yticks(ticks=np.arange(len(self.scanned_freqs)),
                      labels=[f"{x:.2f}" for x in self.scanned_freqs])

        plt.xlabel('Amplitude [a.u.]')
        plt.ylabel('Frequency [MHz]')

        # Creating color bar
        fig.colorbar(cax, ax=ax)
        plt.show()

    def dump_data(self):
        """
        Dumps the scan data to a pickle file.
        """
        path = 'dump.pickle'
        data = {
            "freqs": self.scanned_freqs,
            "amps": self.scanned_amps,
            "shot_data": [x.result for x in self.measurement_scan_result],
            'clfs': [x.clf for x in self.measurement_scan_result]
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Dumped data to {path}")
