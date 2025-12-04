"""
Classical Duffing oscillator model for resonator dynamics.

This module implements the classical Duffing oscillator as an alternative
to the quantum Kerr model for modeling nonlinear resonator behavior.
The Duffing equation provides a classical framework for understanding
bistability and hysteresis in driven nonlinear oscillators.
"""

import numpy as np
from scipy.optimize import fsolve, brentq
from typing import List, Tuple, Optional, Dict, Union
import warnings


class DuffingOscillatorModel:
    """
    Classical Duffing oscillator for resonator dynamics.

    The Duffing equation is: ẍ + δẋ + αx + βx³ = γcos(ωt)

    In steady state, this reduces to finding amplitude A such that:
    A * [α + (3/4)βA² + i*δω] = γ

    where ω is the drive frequency and δ is the damping coefficient.

    This provides a classical alternative to quantum Kerr models for
    comparison and validation of nonlinear resonator physics.
    """

    def __init__(self):
        """Initialize the Duffing oscillator model."""
        self._solution_tolerance = 1e-10
        self._stability_epsilon = 1e-6

    def steady_state_amplitude(
        self, freq: float, power: float, alpha: float, beta: float, kappa: float, omega_0: float = 0.0
    ) -> complex:
        """
        Solve for steady-state amplitude using harmonic balance.

        For the Duffing oscillator ẍ + δẋ + αx + βx³ = γcos(ωt),
        the steady-state amplitude A satisfies:
        A * [(α - ω²) + i*δω + (3/4)β|A|²] = γ

        Args:
            freq: Drive frequency (Hz)
            power: Drive power (proportional to γ²)
            alpha: Linear restoring force coefficient (related to ω₀²)
            beta: Nonlinear coefficient (Duffing nonlinearity)
            kappa: Damping coefficient δ (Hz)
            omega_0: Natural frequency (Hz)

        Returns:
            Complex steady-state amplitude
        """
        # Convert power to drive amplitude (assuming γ ∝ √power)
        gamma = np.sqrt(power)

        # Natural frequency squared
        omega_0_squared = alpha if omega_0 == 0.0 else omega_0**2

        # Try multiple initial guesses to find all solutions
        guess_scale = gamma / kappa if kappa > 0 else 1.0
        initial_guesses = [0.1 * guess_scale, guess_scale, 5.0 * guess_scale]

        best_solution = 0j
        min_residual = float("inf")

        for guess in initial_guesses:
            try:

                def residual(x):
                    A_real, A_imag = x
                    A = complex(A_real, A_imag)
                    return self._duffing_residual(A, freq, gamma, omega_0_squared, beta, kappa)

                sol = fsolve(residual, [guess, 0.0], xtol=1e-12)
                amplitude = complex(sol[0], sol[1])

                # Check if this is a good solution
                res_check = self._duffing_residual(amplitude, freq, gamma, omega_0_squared, beta, kappa)
                residual_magnitude = np.sqrt(res_check[0] ** 2 + res_check[1] ** 2)

                if residual_magnitude < min_residual:
                    min_residual = residual_magnitude
                    best_solution = amplitude

            except Exception:
                continue

        return best_solution

    def detect_bistability(self, freq: float, power: float, params: Dict[str, float]) -> bool:
        """
        Check if system parameters lead to bistable region.

        Bistability occurs when the nonlinearity is strong enough
        compared to damping and when the system is driven sufficiently hard.

        The criterion is approximately: |β| * A³ > δω for some amplitude A

        Args:
            freq: Drive frequency (Hz)
            power: Drive power
            params: Dictionary with 'alpha', 'beta', 'kappa' keys

        Returns:
            True if system is in bistable region, False otherwise
        """
        alpha = params.get("alpha", 0.0)
        beta = params.get("beta", 0.0)
        kappa = params.get("kappa", 1.0)

        if beta == 0:
            return False

        # Calculate steady-state amplitude for this power
        amplitude = self.steady_state_amplitude(freq, power, alpha, beta, kappa)
        abs(amplitude)

        # Find all possible solutions by trying multiple approaches
        all_solutions = self._find_all_duffing_solutions(freq, power, alpha, beta, kappa)

        # Bistability requires at least 2 stable solutions
        stable_count = sum(1 for sol in all_solutions if self._check_duffing_stability(sol, freq, alpha, beta, kappa))

        return stable_count >= 2

    def frequency_response(
        self, freq_array: np.ndarray, power: float, params: Dict[str, float], sweep_direction: str = "up"
    ) -> Tuple[np.ndarray, Dict[str, Optional[float]]]:
        """
        Generate frequency response with hysteresis.

        This method sweeps frequency and tracks which solution branch
        the system follows, accounting for hysteresis and jump phenomena.

        Args:
            freq_array: Array of frequencies to sweep (Hz)
            power: Constant drive power
            params: Parameter dictionary with 'alpha', 'beta', 'kappa'
            sweep_direction: 'up' for increasing frequency, 'down' for decreasing

        Returns:
            Tuple of (complex_amplitudes, jump_info) where:
            - complex_amplitudes: Array of complex amplitudes
            - jump_info: Dict with jump frequency information
        """
        if sweep_direction not in ["up", "down"]:
            raise ValueError("sweep_direction must be 'up' or 'down'")

        alpha = params.get("alpha", 0.0)
        beta = params.get("beta", 0.0)
        kappa = params.get("kappa", 1.0)

        # Order frequencies according to sweep direction
        freq_sequence = freq_array if sweep_direction == "up" else freq_array[::-1]

        amplitudes = []
        current_branch = "lower" if sweep_direction == "up" else "upper"
        jump_info = {"jump_up_freq": None, "jump_down_freq": None}

        for i, freq in enumerate(freq_sequence):
            # Find all solutions at this frequency
            all_solutions = self._find_all_duffing_solutions(freq, power, alpha, beta, kappa)

            # Filter stable solutions
            stable_solutions = [sol for sol in all_solutions if self._check_duffing_stability(sol, freq, alpha, beta, kappa)]

            if len(stable_solutions) == 0:
                amplitudes.append(0j)
            elif len(stable_solutions) == 1:
                amplitudes.append(stable_solutions[0])
            else:
                # Multiple stable solutions - use branch tracking
                stable_solutions.sort(key=lambda x: abs(x))

                if current_branch == "lower":
                    # Check if we should jump to upper branch
                    if self._should_jump_up_duffing(stable_solutions, freq, alpha, beta):
                        current_branch = "upper"
                        if sweep_direction == "up":
                            jump_info["jump_up_freq"] = freq
                        amplitudes.append(stable_solutions[-1])  # Upper branch
                    else:
                        amplitudes.append(stable_solutions[0])  # Lower branch

                else:  # current_branch == 'upper'
                    # Check if we should jump to lower branch
                    if self._should_jump_down_duffing(stable_solutions, freq, alpha, beta):
                        current_branch = "lower"
                        if sweep_direction == "down":
                            jump_info["jump_down_freq"] = freq
                        amplitudes.append(stable_solutions[0])  # Lower branch
                    else:
                        amplitudes.append(stable_solutions[-1])  # Upper branch

        # Reverse if we swept down to maintain original frequency order
        if sweep_direction == "down":
            amplitudes = amplitudes[::-1]

        return np.array(amplitudes), jump_info

    def _duffing_residual(
        self, amplitude: complex, freq: float, gamma: float, omega_0_squared: float, beta: float, kappa: float
    ) -> Tuple[float, float]:
        """
        Calculate residual of Duffing steady-state equation.

        For steady state: A * [(ω₀² - ω²) + i*κω + (3/4)β|A|²] = γ

        Args:
            amplitude: Complex amplitude to test
            freq: Drive frequency
            gamma: Drive amplitude
            omega_0_squared: Natural frequency squared
            beta: Nonlinear coefficient
            kappa: Damping coefficient

        Returns:
            Tuple of (real_residual, imag_residual)
        """
        omega = 2 * np.pi * freq  # Convert to angular frequency
        A_magnitude_squared = abs(amplitude) ** 2

        # Linear terms
        linear_term = omega_0_squared - omega**2

        # Damping term (imaginary)
        damping_term = 1j * kappa * omega

        # Nonlinear term (3/4 factor from harmonic balance)
        nonlinear_term = (3.0 / 4.0) * beta * A_magnitude_squared

        # Total response
        response_factor = linear_term + damping_term + nonlinear_term

        # Steady-state equation: A * response_factor = γ
        lhs = amplitude * response_factor
        residual = lhs - gamma

        return (residual.real, residual.imag)

    def _find_all_duffing_solutions(self, freq: float, power: float, alpha: float, beta: float, kappa: float) -> List[complex]:
        """
        Find all possible Duffing solutions at a given frequency and power.

        Uses multiple initial guesses to find different solution branches.
        """
        gamma = np.sqrt(power)
        omega_0_squared = alpha

        solutions = []

        # Try many initial guesses to find all branches
        guess_scale = gamma / kappa if kappa > 0 else 1.0
        guess_amplitudes = [
            0.01 * guess_scale,
            0.1 * guess_scale,
            0.5 * guess_scale,
            1.0 * guess_scale,
            2.0 * guess_scale,
            5.0 * guess_scale,
            10.0 * guess_scale,
        ]

        for guess in guess_amplitudes:
            try:

                def residual(x):
                    A_real, A_imag = x
                    A = complex(A_real, A_imag)
                    return self._duffing_residual(A, freq, gamma, omega_0_squared, beta, kappa)

                sol = fsolve(residual, [guess, 0.0], xtol=1e-12)
                amplitude = complex(sol[0], sol[1])

                # Verify this is actually a solution
                res_check = self._duffing_residual(amplitude, freq, gamma, omega_0_squared, beta, kappa)
                if np.sqrt(res_check[0] ** 2 + res_check[1] ** 2) < self._solution_tolerance:
                    solutions.append(amplitude)

            except Exception:
                continue

        # Remove duplicates
        unique_solutions = []
        for sol in solutions:
            is_unique = True
            for unique_sol in unique_solutions:
                if abs(sol - unique_sol) < self._solution_tolerance:
                    is_unique = False
                    break
            if is_unique:
                unique_solutions.append(sol)

        # Sort by amplitude magnitude
        unique_solutions.sort(key=lambda x: abs(x))

        return unique_solutions

    def _check_duffing_stability(self, amplitude: complex, freq: float, alpha: float, beta: float, kappa: float) -> bool:
        """
        Check stability of Duffing solution using linear stability analysis.

        Stability requires positive slope d|A|²/d(drive power).
        """
        A_magnitude = abs(amplitude)

        if A_magnitude == 0:
            return True  # Zero solution is always stable in this context

        # Calculate derivative of response with respect to amplitude
        2 * np.pi * freq
        omega_0_squared = alpha

        # The effective spring constant including nonlinearity
        omega_0_squared + (3.0 / 4.0) * beta * A_magnitude**2

        # Stability condition based on response curve slope
        # For Duffing oscillator, the middle branch is unstable

        # Calculate the discriminant of the cubic equation
        # This determines if we're on the stable or unstable branch

        # Simplified stability check: if β > 0 (hardening), upper branch stable
        # if β < 0 (softening), lower branch more stable

        if beta > 0:
            # Hardening spring - higher amplitudes generally more stable
            return True
        elif beta < 0:
            # Softening spring - need more careful analysis
            # The stability depends on the specific parameters
            delta_A = self._stability_epsilon * max(A_magnitude, 1.0)

            # Test stability by perturbing amplitude slightly
            A1 = A_magnitude - delta_A
            A2 = A_magnitude + delta_A

            if A1 <= 0:
                return True  # Near zero amplitude

            # Calculate effective frequency shift for each amplitude
            shift1 = (3.0 / 4.0) * beta * A1**2
            shift2 = (3.0 / 4.0) * beta * A2**2

            # Stable if slope is positive
            slope = (A2 - A1) / (shift2 - shift1) if shift2 != shift1 else 1
            return slope > 0
        else:
            # Linear oscillator (β = 0)
            return True

    def _should_jump_up_duffing(self, stable_solutions: List[complex], freq: float, alpha: float, beta: float) -> bool:
        """
        Determine if system should jump to upper branch in Duffing oscillator.

        This is a simplified model - in practice depends on noise and
        specific stability conditions.
        """
        if len(stable_solutions) < 2:
            return False

        # For hardening springs (β > 0), more likely to jump up
        # For softening springs (β < 0), less likely

        # Simple threshold based on frequency detuning
        omega = 2 * np.pi * freq
        omega_0 = np.sqrt(alpha) if alpha > 0 else 0

        # Jump more likely when frequency is above resonance for hardening spring
        detuning = omega - 2 * np.pi * omega_0

        if beta > 0:  # Hardening
            return detuning > 0  # Jump when above resonance
        else:  # Softening
            return False  # More conservative jumping

    def _should_jump_down_duffing(self, stable_solutions: List[complex], freq: float, alpha: float, beta: float) -> bool:
        """
        Determine if system should jump to lower branch in Duffing oscillator.
        """
        if len(stable_solutions) < 2:
            return False

        # For softening springs (β < 0), more likely to jump down
        omega = 2 * np.pi * freq
        omega_0 = np.sqrt(alpha) if alpha > 0 else 0
        detuning = omega - 2 * np.pi * omega_0

        if beta < 0:  # Softening
            return detuning < 0  # Jump when below resonance
        else:  # Hardening
            return False  # More conservative jumping
