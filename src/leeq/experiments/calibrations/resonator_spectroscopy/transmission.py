from .common import *

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
