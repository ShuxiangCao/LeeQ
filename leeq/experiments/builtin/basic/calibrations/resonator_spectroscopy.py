import numpy as np
from typing import Optional
from scipy import optimize as so
import plotly.graph_objects as go
import plotly
from labchronicle import log_and_record, register_browser_function
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.core.primitives.built_in.common import *
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import SweepParametersSideEffectFactory
from leeq import Experiment, Sweeper, ExperimentManager


class ResonatorSweepTransmissionWithExtraInitialLPB(Experiment):
    """
    Class representing a resonator sweep transmission experiment with extra initial LPB.
    Inherits from a generic "experiment" class.
    """

    @log_and_record
    def run(self,
            dut_qubit: TransmonElement,
            start: float = 8000,
            stop: float = 9000,
            step: float = 5.0,
            num_avs: int = 1000,
            rep_rate: float = 10.0,
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
            update (bool): Whether to update. Default is True.
            amp (float): Amplitude. Default is 1.0.
        """
        # Sweep the frequency
        mp = dut_qubit.get_default_measurement_prim_intlist().clone()

        # Update pulse width
        mp.update_pulse_args(width=rep_rate) if mp_width is None else mp.update_pulse_args(width=mp_width)
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
            n_kwargs={"start": start, "stop": stop, "step": step},
            params=[SweepParametersSideEffectFactory.func(mp.update_freq, {}, "freq", name='frequency')],
        )

        # Perform the experiment
        with ExperimentManager().status().with_parameters(
                shot_number=num_avs,
                shot_period=rep_rate,
                acquisition_type='IQ_average'
        ):
            ExperimentManager().run(lpb, swp)

        result = np.squeeze(mp.result())

        # Save results
        self.result = {
            "Magnitude": np.absolute(result),
            "Phase": np.angle(result),
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
        args = self.retrieve_args(self.run)

        # Get the data
        result = np.squeeze(self.mp.result())
        f = np.arange(args["start"], args["stop"], args["step"])

        if step_no is not None:
            # For the sweep defined above, the step_no is a tuple of one element, the current frequency steps
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
    def root_lorentzian(f: float, f0: float, Q: float, amp: float, baseline: float) -> float:
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
        lorentzian = amp / (1 + (2j * Q * (f - f0) / f0)) + baseline

        # Return the absolute value of the Lorentzian
        return abs(lorentzian)

    def _fit_phase_gradient(self):
        """
        Fit the phase gradient to a Lorentzian function.

        Returns:
            z, f0, Q, amp, baseline, direction (tuple): The phase gradient, the resonant frequency, the quality factor,
            the amplitude, the baseline, and the direction of the Lorentzian peak.
        """
        args = self.retrieve_args(self.run)
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

        # Find the direction of lorentzian peak
        direction = 1 if np.abs(np.max(z)) > np.abs(np.min(z)) else -1

        # Find the frequency for the gradient
        f = (f[:-1] + f[1:]) / 2

        # Find the initial guess for the parameters
        f0, amp, baseline = (
            f[np.argmax(z * direction)],  # Peak with max
            max(z) - min(z),  # Amplitude by finding the difference between max and min
            min(z * direction),  # Find the baseline by finding the effective min. If the lorentzian is negative,
            # the baseline is the max. If the lorentzian is positive, the baseline is the min.
        )

        # Find the initial guess for Q
        half_cut = z * direction - baseline - amp / 2

        # find another derivative to estimate the half line width
        f_diff = (f[:-1] + f[1:]) / 2
        turn_point = np.argwhere(half_cut[:-1] * half_cut[1:] < 0)

        kappa_guess = f_diff[turn_point[1]] - f_diff[turn_point[0]]

        Q_guess = f0 / kappa_guess

        # Finally, we can fit the data
        result = so.minimize(
            leastsq,
            np.array([f0, Q_guess, amp, baseline], dtype=object),
            args=(f, z * direction),
            tol=1.0e-20,
        )  # method='Nelder-Mead',
        f0, Q, amp, baseline = result.x

        return z, f0, Q, amp, baseline, direction

    @register_browser_function(available_after=(run,))
    def plot_magnitude(self):
        args = self.retrieve_args(self.run)
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
        z, f0, Q, amp, baseline, direction = self._fit_phase_gradient()

        args = self.retrieve_args(self.run)
        f = np.arange(args["start"], args["stop"], args["step"])
        f_interpolate = np.arange(
            args["start"],
            args["stop"],
            args["step"] / 5)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=f_interpolate,
                y=self.root_lorentzian(
                    f_interpolate,
                    f0,
                    Q,
                    amp,
                    baseline) *
                  direction,
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

        print(
            "Phase gradient fit f0:%s, Q:%s, amp:%s, base:%s kappa:%f"
            % (f0, Q, amp, baseline, f0 / Q)
        )

        return fig


class ResonatorSweepAmpFreqWithExtraInitialLPB(Experiment):
    @log_and_record
    def run(self,
            dut_qubit: TransmonElement,
            start: float = 8000,
            stop: float = 9000,
            step: float = 5.,
            num_avs: int = 200,
            rep_rate: float = 10.,
            mp_width: Optional[float] = 8,
            initial_lpb: LogicalPrimitiveBlock = None,
            amp_start: float = 0,
            amp_stop: float = 1,
            amp_step: float = 0.05) -> None:
        """
        Run an experiment by sweeping the frequency and amplitude of a qubit.

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
        mprim_index = 0
        mp = dut_qubit.get_measurement_prim_intlist(mprim_index).clone()
        original_freq = mp.freq

        # Update the pulse arguments with either the provided mp_width or rep_rate if mp_width is None.
        mp.update_pulse_args(width=mp_width if mp_width is not None else rep_rate)

        # Remove any previous transform functions.
        mp.set_transform_function(None)

        self.mp = mp

        lpb = mp

        # If initial_lpb is provided, concatenate it with delay and mp.
        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        # Define the frequency sweeper using np.arange and updating mp's frequency.
        swp_freq = Sweeper(
            np.arange,
            n_kwargs={"start": start, "stop": stop, "step": step},
            params=[SweepParametersSideEffectFactory.func(mp.update_freq, {}, "freq")],
        )

        # Define the amplitude sweeper using np.arange and updating mp's amplitude.
        swp_amp = Sweeper(
            np.arange,
            n_kwargs={'start': amp_start, 'stop': amp_stop, 'step': amp_step},
            params=[SweepParametersSideEffectFactory.func(mp.update_pulse_args, {}, 'amp')]
        )

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
        args = self.retrieve_args(self.run)
        trace = np.squeeze(self.mp.result()).transpose()

        return self._plot_data(
            x=np.arange(start=args['start'], stop=args['stop'], step=args['step']),
            y=np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step']),
            z=np.abs(trace),
            title="Resonator response magnitude"
        )

    @register_browser_function(available_after=(run,))
    def plot_phase(self):
        """
        Plot the phase of the resonator response.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """

        args = self.retrieve_args(self.run)
        trace = np.squeeze(self.mp.result()).transpose()

        return self._plot_data(
            x=np.arange(start=args['start'], stop=args['stop'], step=args['step']),
            y=np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step']),
            z=np.unwrap(np.angle(trace)),
            title="Resonator response phase"
        )

    @register_browser_function(available_after=(run,))
    def plot_phase_gradient(self):
        """
        Plot the phase gradient of the resonator response.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        args = self.retrieve_args(self.run)
        trace = np.squeeze(self.mp.result()).transpose()

        return self._plot_data(
            x=np.arange(start=args['start'], stop=args['stop'], step=args['step']),
            y=np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step']),
            z=np.gradient(np.unwrap(np.angle(trace)), axis=1),
            title="Resonator response phase gradient"
        )

    @register_browser_function(available_after=(run,))
    def plot_mag_logscale(self):
        """
        Plot the magnitude of the resonator response in log scale.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        args = self.retrieve_args(self.run)
        trace = np.squeeze(self.mp.result()).transpose()
        return self._plot_data(
            x=np.arange(start=args['start'], stop=args['stop'], step=args['step']),
            y=np.arange(start=args['amp_start'], stop=args['amp_stop'], step=args['amp_step']),
            z=np.log(np.abs(trace)),
            title="Resonator response magnitude (log scale)"
        )

    def live_plots(self, step_no: tuple[int] = None):
        """
        Generate the live plots. This function is called by the live monitor.
        The step no denotes the number of data points to plot, while the
        buffer size is the total number of data points to plot. Some of the data
        in the buffer is note yet valid, so they should not be plotted.
        """

        return self.plot_phase_gradient()
