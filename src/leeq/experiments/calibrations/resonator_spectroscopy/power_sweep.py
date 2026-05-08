from .common import *

class ResonatorPowerSweepSpectroscopy(Experiment):
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
