import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from k_agents.inspection.decorator import text_inspection, visual_inspection
from scipy import optimize as so

from leeq import Experiment, ExperimentManager, Sweeper, setup
from leeq.chronicle import log_and_record, register_browser_function
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import SweepParametersSideEffectFactory
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.theory.simulation.numpy.dispersive_readout.multi_qubit_simulator import MultiQubitDispersiveReadoutSimulator
from leeq.utils import setup_logging

logger = setup_logging(__name__)

__all__ = [
    'ResonatorSweepTransmissionWithExtraInitialLPB',
    'ResonatorSweepTransmissionWithExtraInitialLPB',
    'ResonatorSweepAmpFreqWithExtraInitialLPB',
    'ResonatorSweepTransmissionXiComparison',
    'ResonatorPowerSweepSpectroscopy',
    'ResonatorBistabilityCharacterization',
    'ResonatorThreeRegimeCharacterization'
]


class ResonatorSweepTransmissionWithExtraInitialLPB(Experiment):
    EPII_INFO = {
        "name": "ResonatorSweepTransmissionWithExtraInitialLPB",
        "description": "Resonator frequency sweep with optional initial state preparation",
        "purpose": "Performs a frequency sweep to find and characterize the resonator response. Supports optional initial logical primitive block (LPB) for state preparation, allowing measurement of the resonator in different qubit states. Used for finding resonator frequency and characterizing dispersive shifts.",
        "attributes": {
            "mp": {
                "type": "MeasurementPrimitive",
                "description": "The measurement primitive used for readout"
            },
            "data": {
                "type": "np.ndarray[complex]",
                "description": "Raw complex IQ response data",
                "shape": "(n_frequency_points,)"
            },
            "result": {
                "type": "dict",
                "description": "Processed results containing magnitude and phase",
                "keys": {
                    "Magnitude": "np.ndarray[float] - Magnitude of resonator response",
                    "Phase": "np.ndarray[float] - Phase of resonator response"
                }
            },
            "use_kerr_nonlinearity": {
                "type": "bool",
                "description": "Whether Kerr nonlinearity is enabled (simulation only)"
            },
            "drive_power": {
                "type": "float",
                "description": "Drive power for Kerr simulation (simulation only)"
            }
        },
        "notes": [
            "Initial LPB allows measuring dispersive shift by preparing different qubit states",
            "Supports Kerr nonlinearity simulation for high-power regime",
            "Provides phase gradient fitting for resonator characterization",
            "Multi-qubit simulation uses channel-based readout"
        ]
    }

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
        Execute the experiment on hardware.

        Parameters
        ----------
        dut_qubit : TransmonElement
            The device under test (qubit object).
        start : float, optional
            Start frequency for the sweep (MHz). Default: 8000.0
        stop : float, optional
            Stop frequency for the sweep (MHz). Default: 9000.0
        step : float, optional
            Frequency increment (MHz). Default: 5.0
        num_avs : int, optional
            Number of averages. Default: 5000
        rep_rate : float, optional
            Repetition rate. Default: 0.0
        mp_width : float, optional
            Measurement pulse width (μs). If None, uses rep_rate. Default: 8.0
        initial_lpb : LogicalPrimitiveBlock, optional
            Initial LPB for state preparation. Default: None
        amp : float, optional
            Drive amplitude. Default: 0.02

        Returns
        -------
        None
            Results are stored in instance attributes.
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
                      amp: float = 0.02,
                      use_kerr_nonlinearity: bool = False,
                      power: float = None) -> None:
        """
        Execute the experiment in simulation mode.

        Parameters
        ----------
        dut_qubit : TransmonElement
            The device under test (qubit object).
        start : float, optional
            Start frequency for the sweep (MHz). Default: 8000.0
        stop : float, optional
            Stop frequency for the sweep (MHz). Default: 9000.0
        step : float, optional
            Frequency increment (MHz). Default: 5.0
        num_avs : int, optional
            Number of averages. Default: 1000
        rep_rate : float, optional
            Repetition rate. Default: 0.0
        mp_width : float, optional
            Measurement pulse width (μs). If None, uses rep_rate. Default: None
        initial_lpb : LogicalPrimitiveBlock, optional
            Initial LPB for state preparation. Must be None for multi-qubit simulation. Default: None
        amp : float, optional
            Drive amplitude. Default: 0.02
        use_kerr_nonlinearity : bool, optional
            Enable Kerr nonlinearity effects for high-power regime. Default: False
        power : float, optional
            Explicit drive power for Kerr simulation. If None, calculated as amp². Default: None

        Returns
        -------
        None
            Results are stored in instance attributes.

        Raises
        ------
        ValueError
            If initial_lpb is not None (not supported in multi-qubit mode).

        Notes
        -----
        - The simulation assumes ground state initialization for all qubits
        - Channel mapping is automatically determined from the dut_qubit configuration
        - Results maintain identical format to hardware experiments
        - Kerr effects are only applied if use_kerr_nonlinearity=True
        - Noise is simulated based on num_avs parameter (noise_std = 1/√num_avs)
        """

        if initial_lpb is not None:
            raise ValueError("initial_lpb not supported in high-level simulation mode. "
                           "Multi-qubit dispersive readout simulation requires ground state initialization.")

        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()

        # Extract parameters and build channel mapping for multi-qubit simulation
        params, channel_map, string_to_int_channel_map = self._extract_params(simulator_setup, dut_qubit)

        # Create multi-qubit simulator
        sim = MultiQubitDispersiveReadoutSimulator(**params)

        # Determine which channel we're measuring
        mprim = dut_qubit.get_default_measurement_prim_intlist()
        measurement_channel_str = mprim.channel
        # Channel type handling: support both string and integer channel types
        # This logic addresses compatibility issues where measurement_channel_str can be:
        # - String format: 'readout_0', 'readout_1', etc. (traditional format)
        # - Integer format: 0, 1, 2, etc. (new format used in some test setups)
        # The channel mapping built in _extract_params() stores both formats as keys
        if isinstance(measurement_channel_str, int):
            # For integer channels, lookup directly in the mapping
            # The mapping contains both integer and string keys for compatibility
            measurement_channel = string_to_int_channel_map.get(
                measurement_channel_str,
                # Fallback: if integer channel not in map, use channel 0 (the first channel)
                # This ensures robustness when channel mapping is incomplete
                0
            )
        else:
            # For string channels, use the mapping first, then try fallback parsing
            measurement_channel = string_to_int_channel_map.get(
                measurement_channel_str,
                # Fallback: try to extract number from string like 'readout_0' -> 0
                # This handles cases where the channel mapping doesn't include all string formats
                int(measurement_channel_str.split('_')[-1]) if '_' in str(measurement_channel_str) else 0
            )

        # Create ground state for all qubits
        ground_state = (0,) * params['n_qubits']

        f = np.arange(start, stop, step)
        responses = []

        # Store Kerr parameters if requested
        if use_kerr_nonlinearity:
            self.use_kerr_nonlinearity = True
            self.drive_power = power if power is not None else (amp ** 2)
        else:
            self.use_kerr_nonlinearity = False

        # Run frequency sweep with channel-based readout
        for freq in f:
            # Use channel-based readout with proper multiplexing
            channel_traces = sim.simulate_channel_readout(
                joint_state=ground_state,
                probe_frequencies=[freq] * params['n_resonators'],  # Same freq for all resonators
                channel_map=channel_map,
                noise_std=1/np.sqrt(num_avs)
            )

            # Extract response for our measurement channel
            trace = channel_traces[measurement_channel]

            # Integrate trace to match VirtualTransmon output format
            integrated_response = np.mean(trace)
            responses.append(integrated_response)

        response = np.array(responses)

        # Add phase slope (same as original implementation)
        slope = np.random.normal(-0.1, 0.01)
        phase_slope = np.exp(1j * 2 * np.pi * slope * (f - start))
        response = response * phase_slope

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

        traces = {name: go.Scatter(
                x=data[name][0],
                y=data[name][1],
                mode="lines",
                name=name)
            for name in data}

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
            pass

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
        except Exception:
            return "The experiment has an error fitting phase gradient, implying the experiment is failed."

        return ("The fitting suggest that the resonant frequency is at %f MHz, "
                "with a quality factor of %f (resonator linewidth kappa of %f MHz), an amplitude of %f, and a baseline of %f.") % (
            f0, Q, f0 / Q, amp, baseline)

    def detect_bistability_features(self):
        """
        Detect bistability features in the resonator response.

        Returns:
            dict: Analysis of potential bistability features including
                  S-curve characteristics and jump points.
        """
        if not hasattr(self, 'use_kerr_nonlinearity') or not self.use_kerr_nonlinearity:
            return {"bistability_detected": False, "reason": "Kerr nonlinearity not enabled"}

        magnitude = self.result["Magnitude"]
        phase = self.result["Phase"]

        # Look for S-curve characteristics
        magnitude_gradient = np.gradient(magnitude)
        phase_gradient = np.gradient(np.unwrap(phase))

        # Detect steep transitions (potential jump points)
        steep_transitions = np.where(np.abs(magnitude_gradient) > 3 * np.std(magnitude_gradient))[0]

        # Look for nonlinear phase response
        phase_curvature = np.gradient(phase_gradient)
        high_curvature_points = np.where(np.abs(phase_curvature) > 3 * np.std(phase_curvature))[0]

        analysis = {
            "bistability_detected": len(steep_transitions) > 0 or len(high_curvature_points) > 2,
            "steep_transitions": len(steep_transitions),
            "transition_indices": steep_transitions.tolist(),
            "high_curvature_points": len(high_curvature_points),
            "max_magnitude_gradient": np.max(np.abs(magnitude_gradient)),
            "max_phase_curvature": np.max(np.abs(phase_curvature))
        }

        if hasattr(self, 'drive_power'):
            analysis["drive_power"] = self.drive_power

        return analysis

    @register_browser_function(available_after=(run_simulated,))
    @visual_inspection("""
    Analyze for Kerr nonlinearity and bistability features if enabled:
    1. Look for S-curve response in magnitude plot
    2. Identify any jump discontinuities or steep transitions
    3. Check for distorted lineshapes compared to simple Lorentzian
    4. Assess nonlinear phase response
    5. Look for power-dependent frequency shifts
    Provide analysis of any nonlinear or bistability features observed.
    """)
    def plot_magnitude_with_kerr_analysis(self):
        """Plot magnitude with Kerr nonlinearity analysis if applicable."""
        args = self._get_run_args_dict()
        f = np.arange(args["start"], args["stop"], args["step"])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=f,
                y=self.result["Magnitude"],
                mode="lines",
                name="Magnitude",
                line={'width': 2}
            )
        )

        # Add bistability analysis if Kerr is enabled
        if hasattr(self, 'use_kerr_nonlinearity') and self.use_kerr_nonlinearity:
            bistability_analysis = self.detect_bistability_features()

            # Mark steep transitions
            if bistability_analysis["steep_transitions"] > 0:
                for idx in bistability_analysis["transition_indices"]:
                    if idx < len(f):
                        fig.add_vline(
                            x=f[idx],
                            line={'color': "red", 'dash': "dash", 'width': 1},
                            annotation_text="Transition"
                        )

            # Update title with analysis
            title = "Resonator Spectroscopy (Kerr-enabled)"
            if bistability_analysis["bistability_detected"]:
                title += " - Bistability Features Detected"
            if hasattr(self, 'drive_power'):
                title += f" at P={self.drive_power:.3f}"
        else:
            title = "Resonator spectroscopy magnitude"

        fig.update_layout(
            title=title,
            xaxis_title="Frequency [MHz]",
            yaxis_title="Magnitude",
            plot_bgcolor="white",
        )

        return fig

    def _extract_params(self, setup: HighLevelSimulationSetup, dut_qubit: TransmonElement) -> Tuple[Dict, Dict, Dict]:
        """
        Extract parameters from HighLevelSimulationSetup and build coupling matrix and channel map.

        This method extracts all necessary parameters for multi-qubit dispersive readout simulation,
        including qubit frequencies, anharmonicities, resonator parameters, and constructs the
        coupling matrix from dispersive shifts and qubit-qubit couplings.

        Parameters:
            setup: The high-level simulation setup containing virtual qubits
            dut_qubit: The transmon element being measured (used for channel mapping)

        Returns:
            Tuple containing:
                - params_dict: Dictionary with simulator parameters (frequencies, couplings, etc.)
                - channel_map: Dictionary mapping integer channel IDs to resonator indices
                - string_to_int_channel_map: Dictionary mapping string channel names to integer IDs
        """
        virtual_qubits = setup._virtual_qubits

        # Extract basic parameters from virtual qubits
        qubit_frequencies = [vq.qubit_frequency for vq in virtual_qubits.values()]
        resonator_frequencies = [vq.readout_frequency for vq in virtual_qubits.values()]
        anharmonicities = [getattr(vq, 'anharmonicity', -200.0) for vq in virtual_qubits.values()]
        resonator_kappas = [getattr(vq, 'readout_linewidth', 1.0) for vq in virtual_qubits.values()]

        # Build coupling matrix from dispersive shifts and qubit-qubit couplings
        coupling_matrix = {}
        qubit_list = list(virtual_qubits.values())

        # Add qubit-resonator couplings from dispersive shift
        for i, vq in enumerate(qubit_list):
            # Extract dispersive shift (chi)
            chi = getattr(vq, 'readout_dipsersive_shift', 1.0)  # Note: typo in attribute name preserved
            delta = vq.readout_frequency - vq.qubit_frequency

            # Calculate coupling strength from dispersive shift: g = sqrt(|chi * delta|)
            g = (abs(chi * delta)) ** 0.5
            coupling_matrix[(f"Q{i}", f"R{i}")] = g

        # Add qubit-qubit couplings if they exist
        for i, vq1 in enumerate(qubit_list):
            for j, vq2 in enumerate(qubit_list):
                if i < j:  # Avoid duplicate entries
                    try:
                        # Try to get coupling strength between qubits
                        J = setup.get_coupling_strength_by_qubit(vq1, vq2)
                        if J != 0:
                            coupling_matrix[(f"Q{i}", f"Q{j}")] = J
                    except (AttributeError, KeyError):
                        # No coupling defined between these qubits
                        pass

        # Build channel map - maps measurement channels to lists of resonator indices
        # The simulator expects integer channel IDs, so we create a mapping
        channel_map = {}
        string_to_int_channel_map = {}
        channels = sorted(virtual_qubits.keys())
        for i, channel_id in enumerate(channels):
            # Simple 1:1 mapping - each channel reads one resonator
            channel_map[i] = [i]  # Integer channel ID maps to resonator index
            # Channel mapping compatibility: store both string and original key formats
            # This dual mapping approach ensures compatibility with different test scenarios:
            # - string_to_int_channel_map[str(channel_id)] = i  # Handles string lookups
            # - string_to_int_channel_map[channel_id] = i       # Handles original type lookups
            # This prevents KeyError exceptions when tests use mixed channel ID types
            string_to_int_channel_map[str(channel_id)] = i  # Convert to string for consistent keys
            string_to_int_channel_map[channel_id] = i  # Also store original key

        # Assemble parameters dictionary for MultiQubitDispersiveReadoutSimulator
        params_dict = {
            'qubit_frequencies': qubit_frequencies,
            'qubit_anharmonicities': anharmonicities,
            'resonator_frequencies': resonator_frequencies,
            'resonator_kappas': resonator_kappas,
            'coupling_matrix': coupling_matrix,
            'n_qubits': len(virtual_qubits),
            'n_resonators': len(virtual_qubits)  # Assuming 1:1 mapping
        }

        return params_dict, channel_map, string_to_int_channel_map


class ResonatorSweepAmpFreqWithExtraInitialLPB(Experiment):
    EPII_INFO = {
        "name": "ResonatorSweepAmpFreqWithExtraInitialLPB",
        "description": "2D resonator sweep of frequency and amplitude with optional state preparation",
        "purpose": "Performs a 2D sweep of both frequency and amplitude to characterize resonator response across different drive powers. Supports optional initial LPB for measuring dispersive shifts in different qubit states. Used for power-dependent characterization and nonlinearity studies.",
        "attributes": {
            "mp": {
                "type": "MeasurementPrimitive",
                "description": "The measurement primitive used for readout"
            },
            "trace": {
                "type": "np.ndarray[complex]",
                "description": "2D array of complex IQ response data",
                "shape": "(n_amplitude_points, n_frequency_points)"
            }
        },
        "notes": [
            "2D sweep creates amplitude x frequency grid",
            "Initial LPB allows measuring in different qubit states",
            "Useful for identifying power-dependent effects",
            "No run_simulated method currently implemented"
        ]
    }

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
        Execute the experiment on hardware.

        Parameters
        ----------
        dut_qubit : TransmonElement
            The device under test (qubit object).
        start : float, optional
            Start frequency for the sweep (MHz). Default: 8000.0
        stop : float, optional
            Stop frequency for the sweep (MHz). Default: 9000.0
        step : float, optional
            Frequency increment (MHz). Default: 5.0
        num_avs : int, optional
            Number of averages. Default: 200
        rep_rate : float, optional
            Repetition rate. Default: 0.0
        mp_width : Optional[float], optional
            Measurement primitive width (μs). If None, uses rep_rate. Default: 8.0
        initial_lpb : LogicalPrimitiveBlock, optional
            Initial LPB for state preparation. Default: None
        amp_start : float, optional
            Start amplitude for the sweep. Default: 0.0
        amp_stop : float, optional
            Stop amplitude for the sweep. Default: 1.0
        amp_step : float, optional
            Amplitude increment. Default: 0.05

        Returns
        -------
        None
            Results are stored in instance attributes.
        """
        # Get the original measurement primitive.
        mprim_index = '0'
        mp = dut_qubit.get_measurement_prim_intlist(mprim_index).clone()

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
    EPII_INFO = {
        "name": "ResonatorSweepTransmissionXiComparison",
        "description": "Compare resonator response across different qubit states",
        "purpose": "Performs resonator frequency sweeps with different initial qubit states to measure and compare dispersive shifts. This allows extraction of chi (dispersive shift) by comparing resonator response when qubit is in |0> vs |1> or higher states.",
        "attributes": {
            "result_dict": {
                "type": "dict",
                "description": "Dictionary of ResonatorSweepTransmissionWithExtraInitialLPB experiments",
                "keys": "State labels mapping to experiment instances"
            }
        },
        "notes": [
            "Each LPB in lpb_scan prepares a different qubit state",
            "Useful for measuring dispersive shift chi",
            "Can compare ground, excited, and higher level states",
            "No run_simulated method currently implemented"
        ]
    }

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
        Execute the experiment on hardware.

        Parameters
        ----------
        dut_qubit : Any
            The device under test (qubit object).
        lpb_scan : Union[List, Tuple, Dict]
            LPBs for different state preparations to compare.
        start : float, optional
            Start frequency for the sweep (MHz). Default: 8000.0
        stop : float, optional
            Stop frequency for the sweep (MHz). Default: 9000.0
        step : float, optional
            Frequency increment (MHz). Default: 5.0
        num_avs : int, optional
            Number of averages. Default: 5000
        rep_rate : float, optional
            Repetition rate. Default: 0.0
        mp_width : Optional[float], optional
            Measurement pulse width (μs). Default: 8.0
        amp : Optional[float], optional
            Drive amplitude. Default: None

        Returns
        -------
        None
            Results are stored in instance attributes.
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


        fig.update_layout(
            title='Resonator spectroscopy phase fitting',
            xaxis_title='Frequency [MHz]',
            yaxis_title='Phase',
            plot_bgcolor='white')

        fig.show()


# New Kerr-enabled experiments for high-power regime characterization

class ResonatorPowerSweepSpectroscopy(Experiment):
    EPII_INFO = {
        "name": "ResonatorPowerSweepSpectroscopy",
        "description": "Power sweep spectroscopy to observe bistability and S-curves",
        "purpose": "Sweeps power at a fixed frequency to observe the characteristic S-curve response and bistability phenomena in the high-power regime. Used to characterize nonlinear resonator behavior and find critical powers for bistability.",
        "attributes": {
            "data": {
                "type": "np.ndarray[complex]",
                "description": "Complex IQ response data at each power point",
                "shape": "(n_power_points,)"
            },
            "result": {
                "type": "dict",
                "description": "Processed results containing magnitude and phase",
                "keys": {
                    "Magnitude": "np.ndarray[float] - Response magnitude at each power",
                    "Phase": "np.ndarray[float] - Response phase at each power"
                }
            },
            "freq": {
                "type": "float",
                "description": "Fixed frequency for the power sweep (MHz)"
            },
            "powers": {
                "type": "np.ndarray[float]",
                "description": "Array of drive powers used in sweep",
                "shape": "(n_power_points,)"
            },
            "sweep_direction": {
                "type": "str",
                "description": "Direction of power sweep ('up' or 'down')"
            }
        },
        "notes": [
            "Demonstrates S-curve response in bistable regime",
            "Sweep direction affects hysteresis behavior",
            "Critical power marks onset of bistability",
            "Uses Kerr nonlinearity simulation for realistic modeling"
        ]
    }

    """
    Power sweep spectroscopy experiment to observe bistability and S-curves.

    This experiment sweeps power at a fixed frequency to observe the
    characteristic S-curve response and bistability in the high-power regime.
    """

    @log_and_record
    def run(self,
            dut_qubit: TransmonElement,
            freq: float = 7000,
            power_start: float = 0.01,
            power_stop: float = 1.0,
            power_step: float = 0.01,
            num_avs: int = 1000,
            rep_rate: float = 0.0,
            mp_width: float = 8,
            initial_lpb=None,
            sweep_direction: str = 'up') -> None:
        """
        Run power sweep spectroscopy at fixed frequency.

        Parameters:
            dut_qubit: The device under test (DUT) qubit.
            freq (float): Fixed probe frequency in MHz. Default is 7000.
            power_start (float): Start power for sweep. Default is 0.01.
            power_stop (float): Stop power for sweep. Default is 1.0.
            power_step (float): Power step size. Default is 0.01.
            num_avs (int): Number of averages. Default is 1000.
            rep_rate (float): Repetition rate. Default is 0.0.
            mp_width (float): Measurement pulse width. Default is 8.
            initial_lpb: Initial logical primitive block. Default is None.
            sweep_direction (str): Sweep direction 'up' or 'down'. Default is 'up'.
        """
        # This would be implemented similar to frequency sweep but sweeping power
        # For now, we provide the simulated version
        logger.warning("Hardware version not implemented. Use run_simulated instead.")

    @log_and_record(overwrite_func_name='ResonatorPowerSweepSpectroscopy.run')
    def run_simulated(self,
                      dut_qubit: TransmonElement,
                      freq: float = 7000,
                      power_start: float = 0.01,
                      power_stop: float = 1.0,
                      power_step: float = 0.01,
                      num_avs: int = 1000,
                      rep_rate: float = 0.0,
                      mp_width: float = None,
                      initial_lpb=None,
                      sweep_direction: str = 'up') -> None:
        """
        Run simulated power sweep spectroscopy at fixed frequency.

        This method uses the Kerr-enabled simulator to demonstrate S-curve
        response and bistability phenomena.
        """
        if initial_lpb is not None:
            logger.warning("initial_lpb is ignored in the simulated mode.")

        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()
        virtual_transmon = simulator_setup.get_virtual_qubit(dut_qubit)

        # Enable Kerr nonlinearity for this experiment
        if hasattr(virtual_transmon, 'use_kerr_nonlinearity'):
            virtual_transmon.use_kerr_nonlinearity = True

        powers = np.arange(power_start, power_stop, power_step)

        # Store sweep parameters
        self.freq = freq
        self.powers = powers if sweep_direction == 'up' else powers[::-1]
        self.sweep_direction = sweep_direction

        # Simulate power sweep with hysteresis
        response_list = []
        for power in self.powers:
            try:
                # Use bistability-aware simulation if available
                if hasattr(virtual_transmon, 'simulate_power_sweep_with_hysteresis'):
                    response = virtual_transmon._simulate_trace_with_bistability(
                        0, freq, power, noise_std=1/np.sqrt(num_avs)
                    )
                else:
                    # Fallback to regular simulation
                    response = virtual_transmon.get_resonator_response(
                        np.array([freq]), power=power
                    )[0, 0]
                response_list.append(response)
            except Exception as e:
                logger.warning(f"Error simulating power {power}: {e}")
                response_list.append(0)

        self.data = np.array(response_list)

        # Save results
        self.result = {
            "Magnitude": np.absolute(self.data),
            "Phase": np.angle(self.data),
        }

    @register_browser_function(available_after=(run_simulated,))
    @visual_inspection("""
    Analyze the power sweep plot to identify bistability features:
    1. S-curve response (characteristic S-shaped curve)
    2. Hysteresis loops between up and down sweeps
    3. Jump points where the system switches branches
    4. Critical power where bistability begins
    Provide detailed analysis of any bistability or nonlinear effects observed.
    """)
    def plot_s_curve(self):
        """Plot S-curve response showing magnitude vs power."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.powers,
                y=self.result["Magnitude"],
                mode="lines+markers",
                name=f"Magnitude ({self.sweep_direction} sweep)",
                line={'width': 2}
            )
        )

        fig.update_layout(
            title=f"Power Sweep S-Curve at {self.freq} MHz",
            xaxis_title="Drive Power [a.u.]",
            yaxis_title="Response Magnitude",
            plot_bgcolor="white",
        )

        return fig

    @register_browser_function(available_after=(run_simulated,))
    def plot_phase_vs_power(self):
        """Plot phase response vs power."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.powers,
                y=np.unwrap(self.result["Phase"]),
                mode="lines+markers",
                name=f"Phase ({self.sweep_direction} sweep)",
                line={'width': 2}
            )
        )

        fig.update_layout(
            title=f"Phase vs Power at {self.freq} MHz",
            xaxis_title="Drive Power [a.u.]",
            yaxis_title="Phase [rad]",
            plot_bgcolor="white",
        )

        return fig


class ResonatorBistabilityCharacterization(Experiment):
    EPII_INFO = {
        "name": "ResonatorBistabilityCharacterization",
        "description": "Characterize bistability by measuring hysteresis loops",
        "purpose": "Performs both forward and backward power sweeps to map out the hysteresis loop and identify critical powers. Used to fully characterize bistable behavior including jump points and hysteresis width.",
        "attributes": {
            "forward_result": {
                "type": "dict",
                "description": "Results from forward (increasing power) sweep",
                "keys": {
                    "Magnitude": "np.ndarray[float] - Forward sweep magnitude",
                    "Phase": "np.ndarray[float] - Forward sweep phase"
                }
            },
            "backward_result": {
                "type": "dict",
                "description": "Results from backward (decreasing power) sweep",
                "keys": {
                    "Magnitude": "np.ndarray[float] - Backward sweep magnitude",
                    "Phase": "np.ndarray[float] - Backward sweep phase"
                }
            },
            "powers_forward": {
                "type": "np.ndarray[float]",
                "description": "Power array for forward sweep",
                "shape": "(n_power_points,)"
            },
            "powers_backward": {
                "type": "np.ndarray[float]",
                "description": "Power array for backward sweep",
                "shape": "(n_power_points,)"
            },
            "freq": {
                "type": "float",
                "description": "Fixed frequency for power sweeps (MHz)"
            },
            "forward_jump_power": {
                "type": "float or None",
                "description": "Critical power for forward jump (low to high branch)"
            },
            "backward_jump_power": {
                "type": "float or None",
                "description": "Critical power for backward jump (high to low branch)"
            },
            "hysteresis_width": {
                "type": "float or None",
                "description": "Width of hysteresis loop (difference in critical powers)"
            }
        },
        "notes": [
            "Hysteresis loop reveals bistable dynamics",
            "Jump points indicate critical powers for branch switching",
            "Hysteresis width quantifies bistability strength",
            "Requires Kerr nonlinearity simulation for accurate results"
        ]
    }

    """
    Characterize bistability by measuring hysteresis loops.

    This experiment performs both forward and backward power sweeps
    to map out the hysteresis loop and identify critical powers.
    """

    def run(self,
             dut_qubit: TransmonElement,
             freq: float = 7000,
             power_start: float = 0.01,
             power_stop: float = 1.0,
             power_step: float = 0.005,
             num_avs: int = 1000) -> None:
        """
        Execute the experiment on hardware.

        Parameters
        ----------
        dut_qubit : TransmonElement
            The device under test (qubit object).
        freq : float, optional
            Fixed probe frequency (MHz). Default: 7000.0
        power_start : float, optional
            Start power for sweep. Default: 0.01
        power_stop : float, optional
            Stop power for sweep. Default: 1.0
        power_step : float, optional
            Power step size. Default: 0.005
        num_avs : int, optional
            Number of averages. Default: 1000

        Returns
        -------
        None
            Results are stored in instance attributes.
        """
        logger.warning("Hardware version not implemented. Use run_simulated instead.")

    @log_and_record
    def run_simulated(self,
                      dut_qubit: TransmonElement,
                      freq: float = 7000,
                      power_start: float = 0.01,
                      power_stop: float = 1.0,
                      power_step: float = 0.005,
                      num_avs: int = 1000) -> None:
        """
        Execute the experiment in simulation mode.

        Parameters
        ----------
        dut_qubit : TransmonElement
            The device under test (qubit object).
        freq : float, optional
            Fixed probe frequency (MHz). Default: 7000.0
        power_start : float, optional
            Start power for sweep. Default: 0.01
        power_stop : float, optional
            Stop power for sweep. Default: 1.0
        power_step : float, optional
            Power step size. Default: 0.005
        num_avs : int, optional
            Number of averages. Default: 1000

        Returns
        -------
        None
            Results are stored in instance attributes.
        """
        # Run forward sweep (increasing power)
        forward_exp = ResonatorPowerSweepSpectroscopy()
        forward_exp.run_simulated(
            dut_qubit, freq, power_start, power_stop, power_step,
            num_avs, sweep_direction='up'
        )

        # Run backward sweep (decreasing power)
        backward_exp = ResonatorPowerSweepSpectroscopy()
        backward_exp.run_simulated(
            dut_qubit, freq, power_stop, power_start, -power_step,
            num_avs, sweep_direction='down'
        )

        # Store results
        self.forward_result = forward_exp.result
        self.backward_result = backward_exp.result
        self.powers_forward = forward_exp.powers
        self.powers_backward = backward_exp.powers
        self.freq = freq

        # Analyze hysteresis
        self._analyze_hysteresis()

    def _analyze_hysteresis(self):
        """Analyze hysteresis loop and find critical powers."""
        try:
            # Find jump points (large magnitude changes)
            forward_mag = self.forward_result["Magnitude"]
            backward_mag = self.backward_result["Magnitude"]

            # Find forward jump (low to high branch)
            forward_diff = np.diff(forward_mag)
            forward_jump_idx = np.argmax(forward_diff)
            self.forward_jump_power = self.powers_forward[forward_jump_idx]

            # Find backward jump (high to low branch)
            backward_diff = np.diff(backward_mag)
            backward_jump_idx = np.argmin(backward_diff)  # Looking for negative jump
            self.backward_jump_power = self.powers_backward[backward_jump_idx]

            # Calculate hysteresis width
            self.hysteresis_width = abs(self.forward_jump_power - self.backward_jump_power)

            logger.info(f"Forward jump power: {self.forward_jump_power:.3f}")
            logger.info(f"Backward jump power: {self.backward_jump_power:.3f}")
            logger.info(f"Hysteresis width: {self.hysteresis_width:.3f}")

        except Exception as e:
            logger.warning(f"Could not analyze hysteresis: {e}")
            self.forward_jump_power = None
            self.backward_jump_power = None
            self.hysteresis_width = None

    @register_browser_function(available_after=(run_simulated,))
    @visual_inspection("""
    Analyze the hysteresis plot to characterize bistability:
    1. Identify the forward and backward sweep traces
    2. Look for the hysteresis loop area where traces separate
    3. Find jump points where system switches branches
    4. Measure hysteresis width (difference in critical powers)
    5. Assess the stability of upper and lower branches
    Provide quantitative analysis of bistability characteristics.
    """)
    def plot_hysteresis_loop(self):
        """Plot complete hysteresis loop."""
        fig = go.Figure()

        # Forward sweep
        fig.add_trace(
            go.Scatter(
                x=self.powers_forward,
                y=self.forward_result["Magnitude"],
                mode="lines+markers",
                name="Forward sweep (↑ power)",
                line={'color': "blue", 'width': 2},
                marker={'size': 4}
            )
        )

        # Backward sweep
        fig.add_trace(
            go.Scatter(
                x=self.powers_backward,
                y=self.backward_result["Magnitude"],
                mode="lines+markers",
                name="Backward sweep (↓ power)",
                line={'color': "red", 'width': 2},
                marker={'size': 4}
            )
        )

        # Mark jump points if found
        if hasattr(self, 'forward_jump_power') and self.forward_jump_power:
            fig.add_vline(
                x=self.forward_jump_power,
                line={'color': "blue", 'dash': "dash"},
                annotation_text=f"Forward jump: {self.forward_jump_power:.3f}"
            )

        if hasattr(self, 'backward_jump_power') and self.backward_jump_power:
            fig.add_vline(
                x=self.backward_jump_power,
                line={'color': "red", 'dash': "dash"},
                annotation_text=f"Backward jump: {self.backward_jump_power:.3f}"
            )

        fig.update_layout(
            title=f"Bistability Hysteresis Loop at {self.freq} MHz",
            xaxis_title="Drive Power [a.u.]",
            yaxis_title="Response Magnitude",
            plot_bgcolor="white",
            legend={'yanchor': "top", 'y': 0.99, 'xanchor': "left", 'x': 0.01}
        )

        return fig


class ResonatorThreeRegimeCharacterization(Experiment):
    EPII_INFO = {
        "name": "ResonatorThreeRegimeCharacterization",
        "description": "Comprehensive characterization of all three power regimes",
        "purpose": "Demonstrates linear, bistable, and high-power regimes by performing frequency sweeps at different power levels. Used to understand the full nonlinear behavior of the resonator across different driving strengths.",
        "attributes": {
            "regime_results": {
                "type": "dict",
                "description": "Results for each power regime",
                "keys": {
                    "linear": "dict - Results from linear regime sweep",
                    "bistable": "dict - Results from bistable regime sweep",
                    "high_power": "dict - Results from high-power regime sweep"
                }
            },
            "powers": {
                "type": "dict",
                "description": "Power values used for each regime",
                "keys": {
                    "linear": "float - Power for linear regime",
                    "bistable": "float - Power for bistable regime",
                    "high_power": "float - Power for high-power regime"
                }
            },
            "frequencies": {
                "type": "np.ndarray[float]",
                "description": "Frequency array for sweeps (MHz)",
                "shape": "(n_frequency_points,)"
            },
            "regime_analysis": {
                "type": "dict",
                "description": "Analyzed characteristics for each regime",
                "keys": {
                    "linear": "dict - Linear regime characteristics",
                    "bistable": "dict - Bistable regime characteristics",
                    "high_power": "dict - High-power regime characteristics"
                }
            }
        },
        "notes": [
            "Auto-selects powers based on critical power if enabled",
            "Linear regime shows simple Lorentzian response",
            "Bistable regime exhibits S-curve and hysteresis",
            "High-power regime shows shifted and distorted lineshapes",
            "Comprehensive view of nonlinear resonator physics"
        ]
    }

    """
    Comprehensive characterization of all three power regimes.

    This experiment demonstrates linear, bistable, and high-power regimes
    by performing frequency sweeps at different power levels.
    """

    def run(self,
             dut_qubit: TransmonElement,
             start: float = 6500,
             stop: float = 7500,
             step: float = 2.0,
             num_avs: int = 1000,
             auto_power_selection: bool = True,
             linear_power: float = 0.05,
             bistable_power: float = 0.3,
             high_power: float = 2.0) -> None:
        """
        Execute the experiment on hardware.

        Parameters
        ----------
        dut_qubit : TransmonElement
            The device under test (qubit object).
        start : float, optional
            Start frequency for sweeps (MHz). Default: 6500.0
        stop : float, optional
            Stop frequency for sweeps (MHz). Default: 7500.0
        step : float, optional
            Frequency step size (MHz). Default: 2.0
        num_avs : int, optional
            Number of averages. Default: 1000
        auto_power_selection : bool, optional
            Auto-select powers based on critical power. Default: True
        linear_power : float, optional
            Power for linear regime (used if auto=False). Default: 0.05
        bistable_power : float, optional
            Power for bistable regime (used if auto=False). Default: 0.3
        high_power : float, optional
            Power for high-power regime (used if auto=False). Default: 2.0

        Returns
        -------
        None
            Results are stored in instance attributes.
        """
        logger.warning("Hardware version not implemented. Use run_simulated instead.")

    @log_and_record
    def run_simulated(self,
                      dut_qubit: TransmonElement,
                      start: float = 6500,
                      stop: float = 7500,
                      step: float = 2.0,
                      num_avs: int = 1000,
                      auto_power_selection: bool = True,
                      linear_power: float = 0.05,
                      bistable_power: float = 0.3,
                      high_power: float = 2.0) -> None:
        """
        Execute the experiment in simulation mode.

        Parameters
        ----------
        dut_qubit : TransmonElement
            The device under test (qubit object).
        start : float, optional
            Start frequency for sweeps (MHz). Default: 6500.0
        stop : float, optional
            Stop frequency for sweeps (MHz). Default: 7500.0
        step : float, optional
            Frequency step size (MHz). Default: 2.0
        num_avs : int, optional
            Number of averages. Default: 1000
        auto_power_selection : bool, optional
            Auto-select powers based on critical power. Default: True
        linear_power : float, optional
            Power for linear regime (used if auto=False). Default: 0.05
        bistable_power : float, optional
            Power for bistable regime (used if auto=False). Default: 0.3
        high_power : float, optional
            Power for high-power regime (used if auto=False). Default: 2.0

        Returns
        -------
        None
            Results are stored in instance attributes.
        """
        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()
        virtual_transmon = simulator_setup.get_virtual_qubit(dut_qubit)

        # Enable Kerr nonlinearity
        if hasattr(virtual_transmon, 'use_kerr_nonlinearity'):
            virtual_transmon.use_kerr_nonlinearity = True

        # Auto-select powers if requested
        if auto_power_selection and hasattr(virtual_transmon, 'kerr_calculator'):
            try:
                # Estimate critical power
                kappa = getattr(virtual_transmon, 'kappa', 1.0) * 2 * np.pi * 1e6  # Convert to Hz
                kerr_coeff = getattr(virtual_transmon, 'kerr_coefficient', -0.01)
                P_c = virtual_transmon.kerr_calculator.find_bifurcation_power(kerr_coeff, kappa)

                # Select powers relative to critical power
                powers = {
                    'linear': 0.3 * P_c,
                    'bistable': 1.2 * P_c,
                    'high_power': 10 * P_c
                }
                logger.info(f"Auto-selected powers: {powers}")
            except Exception as e:
                logger.warning(f"Could not auto-select powers: {e}. Using defaults.")
                powers = {
                    'linear': linear_power,
                    'bistable': bistable_power,
                    'high_power': high_power
                }
        else:
            powers = {
                'linear': linear_power,
                'bistable': bistable_power,
                'high_power': high_power
            }

        # Run frequency sweeps at different powers
        self.regime_results = {}
        self.powers = powers
        f = np.arange(start, stop, step)

        for regime, power in powers.items():
            logger.info(f"Running {regime} regime at power {power}")

            # Run standard frequency sweep with Kerr enabled
            exp = ResonatorSweepTransmissionWithExtraInitialLPB()
            exp.run_simulated(
                dut_qubit, start, stop, step, num_avs,
                use_kerr_nonlinearity=True, power=power
            )

            self.regime_results[regime] = exp.result

        self.frequencies = f

        # Analyze regime characteristics
        self._analyze_regime_characteristics()

    def _analyze_regime_characteristics(self):
        """Analyze characteristics of each regime."""
        self.regime_analysis = {}

        for regime, result in self.regime_results.items():
            magnitude = result["Magnitude"]
            phase = result["Phase"]

            # Find resonance peak/dip
            if magnitude.max() / magnitude.min() > 2:
                # Strong feature - likely peak
                resonance_idx = np.argmax(magnitude)
                feature_type = "peak"
            else:
                # Weak feature - likely dip
                resonance_idx = np.argmin(magnitude)
                feature_type = "dip"

            resonance_freq = self.frequencies[resonance_idx]
            contrast = (magnitude.max() - magnitude.min()) / magnitude.mean()

            # Phase slope analysis
            phase_unwrapped = np.unwrap(phase)
            phase_gradient = np.gradient(phase_unwrapped)
            max_phase_slope = np.max(np.abs(phase_gradient))

            self.regime_analysis[regime] = {
                'resonance_freq': resonance_freq,
                'feature_type': feature_type,
                'contrast': contrast,
                'max_phase_slope': max_phase_slope,
                'power': self.powers[regime]
            }

            logger.info(f"{regime.capitalize()} regime: freq={resonance_freq:.1f} MHz, "
                       f"contrast={contrast:.3f}, phase_slope={max_phase_slope:.3f}")

    @register_browser_function(available_after=(run_simulated,))
    @visual_inspection("""
    Compare the three power regimes to understand nonlinear evolution:
    1. Linear regime: Single Lorentzian peak at original frequency
    2. Bistable regime: Distorted lineshape, possible S-curve effects
    3. High-power regime: Shifted peak, different amplitude
    4. Look for frequency shifts between regimes
    5. Assess how lineshape changes with power
    Provide detailed comparison of regime characteristics.
    """)
    def plot_three_regimes_comparison(self):
        """Plot frequency response for all three regimes."""
        fig = go.Figure()

        colors = {'linear': 'blue', 'bistable': 'orange', 'high_power': 'red'}

        for regime, result in self.regime_results.items():
            fig.add_trace(
                go.Scatter(
                    x=self.frequencies,
                    y=result["Magnitude"],
                    mode="lines",
                    name=f"{regime.replace('_', ' ').title()} (P={self.powers[regime]:.3f})",
                    line={'color': colors.get(regime, 'black'), 'width': 2}
                )
            )

        fig.update_layout(
            title="Three Power Regimes Comparison",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Response Magnitude",
            plot_bgcolor="white",
            legend={'yanchor': "top", 'y': 0.99, 'xanchor': "right", 'x': 0.99}
        )

        return fig

    @register_browser_function(available_after=(run_simulated,))
    def plot_regime_phase_comparison(self):
        """Plot phase response for all three regimes."""
        fig = go.Figure()

        colors = {'linear': 'blue', 'bistable': 'orange', 'high_power': 'red'}

        for regime, result in self.regime_results.items():
            fig.add_trace(
                go.Scatter(
                    x=self.frequencies,
                    y=np.unwrap(result["Phase"]),
                    mode="lines",
                    name=f"{regime.replace('_', ' ').title()}",
                    line={'color': colors.get(regime, 'black'), 'width': 2}
                )
            )

        fig.update_layout(
            title="Phase Response Comparison Across Regimes",
            xaxis_title="Frequency [MHz]",
            yaxis_title="Phase [rad]",
            plot_bgcolor="white",
        )

        return fig

    @text_inspection
    def regime_analysis_summary(self) -> str:
        """Provide quantitative analysis of regime characteristics."""
        if not hasattr(self, 'regime_analysis'):
            return "Regime analysis not available. Run the experiment first."

        summary = "Three-Regime Characterization Summary:\n\n"

        for regime, analysis in self.regime_analysis.items():
            summary += f"{regime.replace('_', ' ').title()} Regime:\n"
            summary += f"  - Power: {analysis['power']:.3f}\n"
            summary += f"  - Resonance frequency: {analysis['resonance_freq']:.1f} MHz\n"
            summary += f"  - Feature type: {analysis['feature_type']}\n"
            summary += f"  - Contrast: {analysis['contrast']:.3f}\n"
            summary += f"  - Max phase slope: {analysis['max_phase_slope']:.3f} rad/MHz\n\n"

        # Calculate frequency shifts
        try:
            linear_freq = self.regime_analysis['linear']['resonance_freq']
            bistable_freq = self.regime_analysis['bistable']['resonance_freq']
            high_power_freq = self.regime_analysis['high_power']['resonance_freq']

            summary += "Frequency Shifts:\n"
            summary += f"  - Bistable vs Linear: {bistable_freq - linear_freq:.2f} MHz\n"
            summary += f"  - High-power vs Linear: {high_power_freq - linear_freq:.2f} MHz\n"
            summary += f"  - High-power vs Bistable: {high_power_freq - bistable_freq:.2f} MHz\n"

        except KeyError:
            summary += "Could not calculate frequency shifts.\n"

        return summary


# Assuming other necessary modules are imported elsewhere in the project.

class MeasurementScanParams(Experiment):
    EPII_INFO = {
        "name": "MeasurementScanParams",
        "description": "Scan measurement parameters to optimize SNR",
        "purpose": "Scans measurement frequency and amplitude parameters to find optimal signal-to-noise ratio (SNR) for state discrimination. Used to optimize readout parameters for improved qubit state measurement fidelity.",
        "attributes": {
            "snrs": {
                "type": "np.ndarray[float]",
                "description": "Signal-to-noise ratios for each scan point",
                "shape": "(n_freqs, n_amps)"
            },
            "scanned_freqs": {
                "type": "list[float]",
                "description": "List of scanned frequencies (MHz)"
            },
            "scanned_amps": {
                "type": "list[float]",
                "description": "List of scanned amplitudes"
            },
            "measurement_scan_result": {
                "type": "list",
                "description": "List of MeasurementCalibrationMultilevelGMM results for each scan point"
            }
        },
        "notes": [
            "Optimizes readout SNR for better state discrimination",
            "Can scan frequency and/or amplitude parameters",
            "Accumulates SNR across all distinguishable states if enabled",
            "Results help identify optimal measurement settings"
        ]
    }

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
        Execute the experiment on hardware.

        Parameters
        ----------
        dut : Any
            Device under test.
        sweep_lpb_list : list
            List of sweep parameters.
        mprim_index : int
            Measurement primitive index.
        amp_scan : dict, optional
            Parameters for amplitude scan. Default: None
        freq_scan : dict, optional
            Parameters for frequency scan. Default: None
        accumulate_snr_for_all_distinguishable_state : bool, optional
            Flag to accumulate SNR for all distinguishable states. Default: True
        disable_sub_plot : bool, optional
            Flag to disable subplot. Default: True

        Returns
        -------
        None
            Results are stored in instance attributes.
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
                ax.text(j, i, f"{self.snrs[i, j]:.2f}",
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

