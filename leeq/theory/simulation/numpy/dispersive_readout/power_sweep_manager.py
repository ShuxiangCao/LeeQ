"""
Power sweep manager for handling hysteresis and branch switching in Kerr resonators.

This module manages power sweeps with proper hysteresis tracking, branch selection,
and jump point identification for bistable resonator systems.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from .kerr_physics import KerrBistabilityCalculator


class PowerSweepManager:
    """
    Manager for power sweeps with hysteresis tracking and branch selection.

    This class handles the S-curve response and ensures proper branch switching
    behavior for Kerr resonators in the bistable regime. It tracks the current
    branch state and manages transitions between upper and lower branches.
    """

    def __init__(self, kerr_calculator: Optional[KerrBistabilityCalculator] = None):
        """
        Initialize the power sweep manager.

        Args:
            kerr_calculator: KerrBistabilityCalculator instance for physics calculations
        """
        self.kerr_calculator = kerr_calculator or KerrBistabilityCalculator()
        self.current_branch = "lower"  # Track which branch we're currently on
        self._jump_up_power = None  # Power where system jumps to upper branch
        self._jump_down_power = None  # Power where system jumps to lower branch
        self._branch_tolerance = 1e-6  # Tolerance for branch identification

    def sweep_with_direction(
        self, omega_drive: float, omega_r: float, kappa: float, kerr_coeff: float, powers: np.ndarray, direction: str = "up"
    ) -> Tuple[List[complex], Dict[str, float]]:
        """
        Perform power sweep with proper hysteresis behavior.

        Args:
            omega_drive: Drive frequency (Hz)
            omega_r: Resonator frequency (Hz)
            kappa: Resonator decay rate (Hz)
            kerr_coeff: Kerr coefficient K (Hz)
            powers: Array of drive powers to sweep
            direction: 'up' for increasing power, 'down' for decreasing power

        Returns:
            Tuple of (amplitudes, jump_points) where:
            - amplitudes: List of complex amplitudes for each power
            - jump_points: Dict with 'jump_up' and 'jump_down' powers
        """
        if direction not in ["up", "down"]:
            raise ValueError("Direction must be 'up' or 'down'")

        # Initialize branch state based on sweep direction
        self.current_branch = "lower" if direction == "up" else "upper"

        # Order powers according to sweep direction
        power_sequence = powers if direction == "up" else powers[::-1]
        amplitudes = []
        jump_points = {"jump_up": None, "jump_down": None}

        for i, power in enumerate(power_sequence):
            # Get all steady-state solutions
            solutions = self.kerr_calculator.find_steady_state_solutions(
                omega_drive, omega_r, kappa, kerr_coeff, np.sqrt(power)
            )

            # Identify power regime
            regime = self.identify_power_regime(power, kerr_coeff, kappa)

            if regime == "bistable":
                # In bistable regime, use branch selection logic
                amplitude, branch_changed = self.branch_selection_logic(
                    solutions, power, omega_drive, omega_r, kappa, kerr_coeff, direction
                )

                # Record jump points if branch changed
                if branch_changed:
                    if direction == "up" and self.current_branch == "upper":
                        jump_points["jump_up"] = power
                        self._jump_up_power = power
                    elif direction == "down" and self.current_branch == "lower":
                        jump_points["jump_down"] = power
                        self._jump_down_power = power

            else:
                # In linear or high-power regime, take the single stable solution
                stable_solutions = [
                    sol
                    for sol in solutions
                    if self.kerr_calculator.check_solution_stability(sol, omega_drive, omega_r, kappa, kerr_coeff)
                ]
                amplitude = stable_solutions[0] if stable_solutions else 0j

            amplitudes.append(amplitude)

        # Reverse amplitudes if we swept down to maintain original power order
        if direction == "down":
            amplitudes = amplitudes[::-1]

        return amplitudes, jump_points

    def branch_selection_logic(
        self,
        solutions: List[complex],
        power: float,
        omega_drive: float,
        omega_r: float,
        kappa: float,
        kerr_coeff: float,
        direction: str,
    ) -> Tuple[complex, bool]:
        """
        Select appropriate branch based on hysteresis and current state.

        Args:
            solutions: All steady-state solutions found
            power: Current drive power
            omega_drive: Drive frequency (Hz)
            omega_r: Resonator frequency (Hz)
            kappa: Resonator decay rate (Hz)
            kerr_coeff: Kerr coefficient K (Hz)
            direction: Sweep direction ('up' or 'down')

        Returns:
            Tuple of (selected_amplitude, branch_changed)
        """
        # Filter stable solutions
        stable_solutions = []
        for sol in solutions:
            if self.kerr_calculator.check_solution_stability(sol, omega_drive, omega_r, kappa, kerr_coeff):
                stable_solutions.append(sol)

        if len(stable_solutions) == 0:
            return 0j, False
        elif len(stable_solutions) == 1:
            return stable_solutions[0], False
        elif len(stable_solutions) == 2:
            # Bistable regime with upper and lower branches
            # Sort by amplitude magnitude
            stable_solutions.sort(key=lambda x: abs(x))
            lower_branch = stable_solutions[0]  # Smaller amplitude
            upper_branch = stable_solutions[1]  # Larger amplitude

            branch_changed = False

            if self.current_branch == "lower":
                # Currently on lower branch
                if direction == "up":
                    # Check if we should jump to upper branch
                    if self._should_jump_to_upper(power, kerr_coeff, kappa):
                        self.current_branch = "upper"
                        branch_changed = True
                        return upper_branch, branch_changed
                    else:
                        return lower_branch, branch_changed
                else:  # direction == 'down'
                    # Sweeping down on lower branch - stay on lower
                    return lower_branch, branch_changed

            else:  # self.current_branch == 'upper'
                # Currently on upper branch
                if direction == "down":
                    # Check if we should jump to lower branch
                    if self._should_jump_to_lower(power, kerr_coeff, kappa):
                        self.current_branch = "lower"
                        branch_changed = True
                        return lower_branch, branch_changed
                    else:
                        return upper_branch, branch_changed
                else:  # direction == 'up'
                    # Sweeping up on upper branch - stay on upper
                    return upper_branch, branch_changed

        # More than 2 stable solutions - shouldn't happen in Kerr resonator
        # Return solution closest to current branch
        if self.current_branch == "lower":
            return min(stable_solutions, key=lambda x: abs(x)), False
        else:
            return max(stable_solutions, key=lambda x: abs(x)), False

    def identify_power_regime(self, power: float, kerr_coeff: float, kappa: float) -> str:
        """
        Identify which power regime using the KerrBistabilityCalculator.

        Args:
            power: Drive power
            kerr_coeff: Kerr coefficient K (Hz)
            kappa: Resonator decay rate (Hz)

        Returns:
            Regime string: 'linear', 'bistable', or 'high_power_stable'
        """
        return self.kerr_calculator.identify_power_regime(power, kerr_coeff, kappa)

    def get_jump_points(self) -> Dict[str, Optional[float]]:
        """
        Get recorded jump points for hysteresis characterization.

        Returns:
            Dictionary with 'jump_up' and 'jump_down' power values
        """
        return {"jump_up": self._jump_up_power, "jump_down": self._jump_down_power}

    def reset_branch_state(self, initial_branch: str = "lower"):
        """
        Reset the branch tracking state.

        Args:
            initial_branch: Initial branch to start from ('lower' or 'upper')
        """
        if initial_branch not in ["lower", "upper"]:
            raise ValueError("Initial branch must be 'lower' or 'upper'")

        self.current_branch = initial_branch
        self._jump_up_power = None
        self._jump_down_power = None

    def _should_jump_to_upper(self, power: float, kerr_coeff: float, kappa: float) -> bool:
        """
        Determine if system should jump from lower to upper branch.

        Jump occurs when the lower branch becomes unstable or at critical power.

        Args:
            power: Current drive power
            kerr_coeff: Kerr coefficient K (Hz)
            kappa: Resonator decay rate (Hz)

        Returns:
            True if should jump to upper branch
        """
        P_c = self.kerr_calculator.find_bifurcation_power(kerr_coeff, kappa)

        # Jump to upper branch at approximately 1.2 * P_c
        # This is somewhat empirical and depends on the specific system
        jump_threshold = 1.2 * P_c

        return power > jump_threshold

    def _should_jump_to_lower(self, power: float, kerr_coeff: float, kappa: float) -> bool:
        """
        Determine if system should jump from upper to lower branch.

        Jump occurs when the upper branch becomes unstable or at critical power.

        Args:
            power: Current drive power
            kerr_coeff: Kerr coefficient K (Hz)
            kappa: Resonator decay rate (Hz)

        Returns:
            True if should jump to lower branch
        """
        P_c = self.kerr_calculator.find_bifurcation_power(kerr_coeff, kappa)

        # Jump to lower branch at approximately 0.8 * P_c
        # This creates the hysteresis loop
        jump_threshold = 0.8 * P_c

        return power < jump_threshold

    def simulate_s_curve(
        self,
        omega_drive: float,
        omega_r: float,
        kappa: float,
        kerr_coeff: float,
        power_range: Tuple[float, float],
        num_points: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Generate S-curve response showing all branches.

        This method generates the complete S-curve by finding all solutions
        at each power point, not just the hysteresis path.

        Args:
            omega_drive: Drive frequency (Hz)
            omega_r: Resonator frequency (Hz)
            kappa: Resonator decay rate (Hz)
            kerr_coeff: Kerr coefficient K (Hz)
            power_range: Tuple of (min_power, max_power)
            num_points: Number of power points

        Returns:
            Dictionary with 'powers', 'lower_branch', 'upper_branch', 'middle_branch'
        """
        powers = np.linspace(power_range[0], power_range[1], num_points)

        lower_branch = []
        middle_branch = []
        upper_branch = []

        for power in powers:
            solutions = self.kerr_calculator.find_steady_state_solutions(
                omega_drive, omega_r, kappa, kerr_coeff, np.sqrt(power)
            )

            # Sort solutions by amplitude magnitude
            solutions.sort(key=lambda x: abs(x))

            # Classify solutions by stability
            stable_solutions = []
            unstable_solutions = []

            for sol in solutions:
                if self.kerr_calculator.check_solution_stability(sol, omega_drive, omega_r, kappa, kerr_coeff):
                    stable_solutions.append(sol)
                else:
                    unstable_solutions.append(sol)

            # Assign to branches
            if len(solutions) >= 1:
                lower_branch.append(solutions[0])  # Lowest amplitude
            else:
                lower_branch.append(0j)

            if len(solutions) >= 2:
                middle_branch.append(solutions[1])  # Middle amplitude (usually unstable)
            else:
                middle_branch.append(0j)

            if len(solutions) >= 3:
                upper_branch.append(solutions[2])  # Highest amplitude
            else:
                upper_branch.append(0j)

        return {
            "powers": powers,
            "lower_branch": np.array(lower_branch),
            "middle_branch": np.array(middle_branch),
            "upper_branch": np.array(upper_branch),
        }
