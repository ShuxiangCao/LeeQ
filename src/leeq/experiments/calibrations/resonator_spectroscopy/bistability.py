from .common import *
from .power_sweep import ResonatorPowerSweepSpectroscopy

class ResonatorBistabilityCharacterization(Experiment):
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
