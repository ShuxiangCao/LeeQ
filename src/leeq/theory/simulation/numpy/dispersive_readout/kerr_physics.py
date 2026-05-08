"""
Kerr bistability physics calculations for high-power resonator response simulation.

This module implements the physics of Kerr nonlinearity in superconducting resonators,
including bistability effects and S-curve response in the high-power regime.
"""

import numpy as np
from scipy.optimize import fsolve
from typing import List, Tuple, Union, Optional


class KerrBistabilityCalculator:
    """
    Calculator for Kerr nonlinearity effects with proper bistability modeling.

    This class handles the physics of driven Kerr resonators including:
    - Kerr coefficient calculation from transmon parameters
    - Steady-state cubic equation solutions (up to 3 branches)
    - Stability analysis to identify stable/unstable branches
    - Bifurcation power calculation
    - Power regime identification (linear/bistable/high-power)
    """

    def __init__(self):
        """Initialize the Kerr bistability calculator."""
        self._solution_tolerance = 1e-10
        self._stability_epsilon = 1e-6

    def calculate_kerr_coefficient(self, f_r: float, f_q: float, anharmonicity: float, g: float, num_levels: int = 4) -> float:
        """
        Calculate Kerr coefficient from transmon parameters.

        For a transmon-resonator system in the dispersive regime:
        K ≈ -α*(g/Δ)²/2

        where α is the anharmonicity, g is the coupling strength,
        and Δ = f_q - f_r is the detuning.

        Args:
            f_r: Resonator frequency (Hz)
            f_q: Qubit frequency (Hz)
            anharmonicity: Transmon anharmonicity (Hz)
            g: Coupling strength (Hz)
            num_levels: Number of transmon levels (not used in this approximation)

        Returns:
            Kerr coefficient K (Hz)
        """
        detuning = f_q - f_r

        if abs(detuning) < abs(g):
            raise ValueError(f"Not in dispersive regime: |Δ| = {abs(detuning):.3e} Hz < g = {abs(g):.3e} Hz. Need |Δ| >> g.")

        # Kerr coefficient: K ≈ -α*(g/Δ)²/2
        kerr_coeff = -anharmonicity * (g / detuning) ** 2 / 2

        return kerr_coeff

    def find_steady_state_solutions(
        self, omega_drive: float, omega_r: float, kappa: float, kerr_coeff: float, drive_amplitude: float
    ) -> List[complex]:
        """
        Solve steady-state cubic equation for all branches.

        Solves: a*[-i(ω_d - ω_r - K|a|²) + κ/2] = F

        This cubic equation in |a|² can have up to 3 solutions:
        - Lower stable branch (always exists)
        - Middle unstable branch (exists in bistable regime)
        - Upper stable branch (exists above bifurcation)

        Args:
            omega_drive: Drive frequency (Hz)
            omega_r: Resonator frequency (Hz)
            kappa: Resonator decay rate (Hz)
            kerr_coeff: Kerr coefficient K (Hz)
            drive_amplitude: Drive field amplitude (√Hz)

        Returns:
            List of complex amplitudes (up to 3 solutions)
        """
        solutions = []

        # Try multiple initial guesses to find all branches
        # Scale guesses based on drive amplitude and system parameters
        guess_scale = abs(drive_amplitude) / kappa if kappa > 0 else 1.0
        initial_guesses = [
            0.1 * guess_scale,  # Small amplitude (lower branch)
            guess_scale,  # Medium amplitude
            5.0 * guess_scale,  # Large amplitude (upper branch)
        ]

        for guess in initial_guesses:
            try:
                # Solve complex equation by splitting into real/imaginary parts
                def residual(x):
                    a_real, a_imag = x
                    a = complex(a_real, a_imag)
                    return self._steady_state_residual(a, omega_drive, omega_r, kappa, kerr_coeff, drive_amplitude)

                sol = fsolve(residual, [guess, 0.0], xtol=1e-12)
                # fsolve returns a numpy array - ensure we have at least 2 elements
                if len(sol) >= 2:
                    amplitude = complex(sol[0], sol[1])
                else:
                    amplitude = complex(sol[0], 0.0)

                # Verify this is actually a solution
                residual_check = self._steady_state_residual(
                    amplitude, omega_drive, omega_r, kappa, kerr_coeff, drive_amplitude
                )
                if np.sqrt(residual_check[0] ** 2 + residual_check[1] ** 2) < self._solution_tolerance:
                    solutions.append(amplitude)

            except Exception:
                # fsolve failed for this initial guess, continue with others
                continue

        # Filter out duplicate solutions
        unique_solutions = self._filter_unique_solutions(solutions)

        return unique_solutions

    def check_solution_stability(
        self, amplitude: complex, omega_drive: float, omega_r: float, kappa: float, kerr_coeff: float
    ) -> bool:
        """
        Check if solution is stable using linear stability analysis.

        The stability is determined by the slope of the response curve.
        The middle branch with negative slope d|a|²/dF² < 0 is unstable.

        Args:
            amplitude: Complex amplitude to check
            omega_drive: Drive frequency (Hz)
            omega_r: Resonator frequency (Hz)
            kappa: Resonator decay rate (Hz)
            kerr_coeff: Kerr coefficient K (Hz)

        Returns:
            True if solution is stable, False if unstable
        """
        n_photons = abs(amplitude) ** 2

        # Calculate response slope using numerical derivative
        delta_n = self._stability_epsilon * max(n_photons, 1.0)

        # Test two nearby points
        n1 = n_photons - delta_n
        n2 = n_photons + delta_n

        # Calculate effective drive needed for each photon number
        F1_squared = self._drive_squared_for_photon_number(n1, omega_drive, omega_r, kappa, kerr_coeff)
        F2_squared = self._drive_squared_for_photon_number(n2, omega_drive, omega_r, kappa, kerr_coeff)

        if F1_squared < 0 or F2_squared < 0:
            # Unphysical solution
            return False

        # Stability requires positive slope dn/dF²
        slope = (n2 - n1) / (F2_squared - F1_squared)

        return slope > 0

    def find_bifurcation_power(self, kerr_coeff: float, kappa: float) -> float:
        """
        Calculate critical power for bifurcation.

        The critical power is: P_c = κ^(3/2) / (2√(3|K|))

        Args:
            kerr_coeff: Kerr coefficient K (Hz)
            kappa: Resonator decay rate (Hz)

        Returns:
            Critical power P_c (W or photon number units)
        """
        if kerr_coeff == 0:
            return float("inf")

        P_c = kappa ** (3 / 2) / (2 * np.sqrt(3 * abs(kerr_coeff)))

        return P_c

    def identify_power_regime(self, power: float, kerr_coeff: float, kappa: float) -> str:
        """
        Identify which power regime the system is in.

        Three regimes:
        - Linear: P < 0.5*P_c (single stable solution, minimal shift)
        - Bistable: 0.5*P_c < P < 3*P_c (S-curve response with hysteresis)
        - High-power stable: P > 3*P_c (single stable solution at shifted frequency)

        Args:
            power: Drive power
            kerr_coeff: Kerr coefficient K (Hz)
            kappa: Resonator decay rate (Hz)

        Returns:
            Regime string: 'linear', 'bistable', or 'high_power_stable'
        """
        P_c = self.find_bifurcation_power(kerr_coeff, kappa)

        if power < 0.5 * P_c:
            return "linear"
        elif power < 3.0 * P_c:
            return "bistable"
        else:
            return "high_power_stable"

    def _steady_state_residual(
        self, amplitude: complex, omega_drive: float, omega_r: float, kappa: float, kerr_coeff: float, drive_amplitude: float
    ) -> Tuple[float, float]:
        """
        Calculate residual of steady-state equation.

        Steady-state equation: a*[-i(ω_d - ω_r - K|a|²) + κ/2] = F

        Returns real and imaginary parts of residual.
        """
        n_photons = abs(amplitude) ** 2
        delta_eff = omega_drive - omega_r - kerr_coeff * n_photons

        # Left side of equation
        lhs = amplitude * (-1j * delta_eff + kappa / 2)

        # Right side (drive amplitude)
        rhs = drive_amplitude

        residual = lhs - rhs

        return (residual.real, residual.imag)

    def _filter_unique_solutions(self, solutions: List[complex]) -> List[complex]:
        """
        Filter out duplicate solutions within numerical tolerance.
        """
        if not solutions:
            return []

        unique_solutions = []

        for sol in solutions:
            is_unique = True
            for unique_sol in unique_solutions:
                if abs(sol - unique_sol) < self._solution_tolerance:
                    is_unique = False
                    break
            if is_unique:
                unique_solutions.append(sol)

        # Sort by amplitude magnitude for consistent ordering
        unique_solutions.sort(key=lambda x: abs(x))

        return unique_solutions

    def _drive_squared_for_photon_number(
        self, n_photons: float, omega_drive: float, omega_r: float, kappa: float, kerr_coeff: float
    ) -> float:
        """
        Calculate |F|² required for a given photon number.

        From steady-state equation: |F|² = |a|²*[(ω_d - ω_r - K|a|²)² + (κ/2)²]
        """
        if n_photons < 0:
            return -1  # Unphysical

        delta_eff = omega_drive - omega_r - kerr_coeff * n_photons
        F_squared = n_photons * (delta_eff**2 + (kappa / 2) ** 2)

        return F_squared
