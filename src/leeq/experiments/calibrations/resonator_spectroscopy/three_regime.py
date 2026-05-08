from .common import *
from .transmission import ResonatorSweepTransmissionWithExtraInitialLPB

class ResonatorThreeRegimeCharacterization(Experiment):
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
