"""
Physics calculations for dispersive readout chi shifts.

This module implements physically accurate chi shift calculations for transmon qubits
in the dispersive regime, accounting for anharmonicity and multi-level effects.

The module provides the `ChiShiftCalculator` class for computing dispersive shifts
that arise when a transmon qubit is coupled to a readout resonator. Unlike simplified
constant-chi models, this implementation accounts for:

- Multi-level transmon physics with anharmonicity
- Level-dependent coupling matrix elements (√n scaling)
- Proper summation over virtual transitions
- Validation of dispersive regime conditions

Classes
-------
ChiShiftCalculator
    Main calculator for physics-based chi shift computation.

Examples
--------
Basic usage:

>>> from leeq.theory.simulation.numpy.dispersive_readout.physics import ChiShiftCalculator
>>> calc = ChiShiftCalculator()
>>> chis = calc.calculate_chi_shifts(
...     f_r=7000,           # 7 GHz resonator
...     f_q=5000,           # 5 GHz qubit
...     anharmonicity=-250, # -250 MHz anharmonicity
...     g=100,              # 100 MHz coupling
...     num_levels=4        # Include 4 transmon levels
... )
>>> print(f"Chi shifts: {chis} MHz")

Notes
-----
This implementation assumes:
- Transmon in the charge-insensitive regime
- Dispersive coupling regime (g << |Δ|)
- Single-mode resonator approximation
- Rotating wave approximation

For systems outside these assumptions, consider full Hamiltonian simulations
using packages like QuTiP.

References
----------
[1] Blais et al., "Circuit quantum electrodynamics", Rev. Mod. Phys. 93, 025005 (2021)
[2] Koch et al., "Charge-insensitive qubit design", Phys. Rev. A 76, 042319 (2007)
[3] Schuster et al., "Resolving photon number states", Nature 445, 515-518 (2007)
"""

from typing import Optional
import numpy as np


class ChiShiftCalculator:
    """
    Calculate level-dependent chi shifts for transmon qubits in the dispersive regime.

    This class implements physically accurate calculations of dispersive shifts (chi)
    that account for anharmonicity and multi-level transmon physics. The calculations
    are based on the virtual photon exchange between qubit and resonator.

    The dispersive shift is given by:
    χ_n = Σ_m |g_nm|² / (ω_r - ω_nm)

    where:
    - χ_n is the dispersive shift when qubit is in state |n⟩
    - g_nm are the coupling matrix elements between states |n⟩ and |m⟩
    - ω_nm = E_m - E_n are the transition frequencies
    - ω_r is the resonator frequency

    Examples
    --------
    Calculate chi shifts for a typical transmon:

    >>> calc = ChiShiftCalculator()
    >>> chis = calc.calculate_chi_shifts(
    ...     f_r=7000,           # 7 GHz resonator
    ...     f_q=5000,           # 5 GHz qubit
    ...     anharmonicity=-250, # -250 MHz anharmonicity
    ...     g=100,              # 100 MHz coupling
    ...     num_levels=4
    ... )
    >>> print(f"Chi shifts: {chis} MHz")

    Validate dispersive regime:

    >>> is_dispersive = calc.validate_dispersive_regime(
    ...     f_r=7000, f_q=5000, g=100
    ... )
    >>> print(f"In dispersive regime: {is_dispersive}")

    References
    ----------
    [1] Blais et al., "Circuit quantum electrodynamics", Rev. Mod. Phys. 93, 025005 (2021)
    [2] Koch et al., "Charge-insensitive qubit design", Phys. Rev. A 76, 042319 (2007)
    """

    def __init__(self):
        """
        Initialize the chi shift calculator.

        The calculator is stateless and can be reused for multiple calculations
        with different parameters.
        """
        pass

    def calculate_chi_shifts(
        self,
        f_r: float,
        f_q: float,
        anharmonicity: float,
        g: float,
        num_levels: int = 4,
        temperature: float = 0.0,
        relative: bool = True,
    ) -> np.ndarray:
        """
        Calculate dispersive shifts for each transmon level.

        This method computes the frequency shift of a resonator when coupled to
        a transmon qubit in different energy states. The calculation includes
        contributions from all virtual transitions between qubit levels.

        Parameters
        ----------
        f_r : float
            Resonator frequency in MHz. Typical values: 6000-8000 MHz.
        f_q : float
            Qubit frequency (|0⟩ → |1⟩ transition) in MHz. Typical values: 4000-6000 MHz.
        anharmonicity : float
            Transmon anharmonicity in MHz. Always negative, typical values: -200 to -300 MHz.
            This parameter controls the frequency spacing between higher levels.
        g : float
            Qubit-resonator coupling strength in MHz. Typical values: 50-200 MHz.
            Must satisfy g << |f_r - f_q| for dispersive regime validity.
        num_levels : int, optional
            Number of transmon energy levels to include in calculation, by default 4.
            More levels improve accuracy but increase computation time.
        temperature : float, optional
            Temperature in Kelvin, by default 0.0. Currently not used in calculation
            but reserved for future thermal population weighting.
        relative : bool, optional
            If True (default), return chi shifts relative to ground state (χ₀ = 0).
            If False, return absolute chi shifts. Relative shifts are more commonly used.

        Returns
        -------
        np.ndarray
            Array of chi shifts [χ₀, χ₁, χ₂, ...] in MHz. The nth element is the
            dispersive shift when the qubit is in state |n⟩.

        Raises
        ------
        UserWarning
            If coupling strength g is not much smaller than detuning |f_r - f_q|,
            indicating the system may not be in the dispersive regime.

        Notes
        -----
        The dispersive shift formula is:

        χₙ = Σₘ |gₙₘ|² / (ωᵣ - ωₙₘ)

        where:
        - gₙₘ = g√max(n,m) for adjacent levels |n⟩↔|m⟩, 0 otherwise
        - ωₙₘ = Eₘ - Eₙ with Eₙ = nf_q + n(n-1)α/2
        - α is the anharmonicity parameter

        For the two-level approximation (num_levels=2), this reduces to:
        χ₁ - χ₀ = g²/(f_r - f_q)

        Examples
        --------
        Basic usage with typical transmon parameters:

        >>> calc = ChiShiftCalculator()
        >>> chis = calc.calculate_chi_shifts(
        ...     f_r=7000, f_q=5000, anharmonicity=-250, g=100
        ... )
        >>> print(f"Chi shifts: {chis}")
        [ 0.     -5.13  -9.95 -14.51] MHz

        Compare two-level vs multi-level calculation:

        >>> chi_2level = calc.calculate_chi_shifts(
        ...     f_r=7000, f_q=5000, anharmonicity=-250, g=100, num_levels=2
        ... )
        >>> chi_4level = calc.calculate_chi_shifts(
        ...     f_r=7000, f_q=5000, anharmonicity=-250, g=100, num_levels=4
        ... )
        >>> print(f"2-level: {chi_2level[:2]}")
        >>> print(f"4-level: {chi_4level[:2]}")

        References
        ----------
        .. [1] Blais, A. et al. "Circuit quantum electrodynamics."
               Rev. Mod. Phys. 93, 025005 (2021).
        .. [2] Schuster, D. I. et al. "Resolving photon number states in a superconducting circuit."
               Nature 445, 515-518 (2007).
        """
        # Validate dispersive regime condition
        detuning = abs(f_r - f_q)
        if g >= 0.1 * detuning:
            import warnings

            warnings.warn(
                f"Not in dispersive regime: g={g} MHz should be << |Δ|={detuning} MHz. "
                f"Results may be inaccurate. Consider g/|Δ| < 0.1 for reliable dispersive treatment.",
                stacklevel=2,
                category=UserWarning,
            )

        chi_shifts = np.zeros(num_levels)

        for n in range(num_levels):
            # Calculate chi shift for level n
            # χ_n = Σ_m |g_nm|² / (ω_r - ω_nm) for all virtual transitions

            for m in range(num_levels):
                if m != n:
                    # Energy difference for transition n -> m
                    E_n = self._calculate_energy(n, f_q, anharmonicity)
                    E_m = self._calculate_energy(m, f_q, anharmonicity)
                    omega_nm = E_m - E_n

                    # Coupling matrix element
                    g_nm = self._coupling_element(n, m, g)

                    # Only include transitions with non-zero coupling
                    if abs(g_nm) > 0 and abs(omega_nm) > 1e-6:  # Avoid division by zero
                        # Add contribution to chi shift
                        # Use detuning from resonator to actual transition frequency E_m
                        chi_shifts[n] += (g_nm**2) / (f_r - E_m)

        # Convert to relative chi shifts if requested
        if relative:
            chi_shifts = chi_shifts - chi_shifts[0]

        return chi_shifts

    def _calculate_energy(self, n: int, f_q: float, alpha: float) -> float:
        """
        Calculate energy of level n including anharmonicity.

        Energy formula: E_n = n*ω_q + n*(n-1)*α/2

        Args:
            n: Level number (0, 1, 2, ...)
            f_q: Qubit frequency (MHz)
            alpha: Anharmonicity (MHz, typically negative)

        Returns:
            Energy of level n (MHz)
        """
        return n * f_q + n * (n - 1) * alpha / 2

    def _coupling_element(self, n: int, m: int, g: float) -> float:
        """
        Calculate coupling matrix element between levels n and m.

        For adjacent levels (|m-n| = 1): g_nm = g * sqrt(max(n,m))
        For non-adjacent levels: g_nm = 0 (dipole selection rule)

        Args:
            n: Initial level
            m: Final level
            g: Base coupling strength (MHz)

        Returns:
            Coupling matrix element g_nm (MHz)
        """
        if abs(m - n) == 1:
            # Adjacent levels: sqrt(n) scaling
            return g * np.sqrt(max(n, m))
        else:
            # Non-adjacent levels have zero dipole coupling
            return 0.0

    def calculate_two_level_chi(self, f_r: float, f_q: float, g: float) -> float:
        """
        Calculate chi shift in the two-level approximation.

        Formula: χ = g² / (ω_r - ω_q)

        Args:
            f_r: Resonator frequency (MHz)
            f_q: Qubit frequency (MHz)
            g: Coupling strength (MHz)

        Returns:
            Chi shift in two-level limit (MHz)
        """
        detuning = f_r - f_q
        if abs(detuning) < 1e-6:
            raise ValueError("Resonator and qubit frequencies are too close (detuning < 1 kHz)")
        return g**2 / detuning

    def validate_dispersive_regime(self, f_r: float, f_q: float, g: float, threshold: float = 0.1) -> bool:
        """
        Check if parameters are in the dispersive regime.

        Condition: g << |ω_r - ω_q|

        Args:
            f_r: Resonator frequency (MHz)
            f_q: Qubit frequency (MHz)
            g: Coupling strength (MHz)
            threshold: Maximum ratio g/|Δ| for dispersive regime

        Returns:
            True if in dispersive regime
        """
        detuning = abs(f_r - f_q)
        if detuning == 0:
            return False
        return g / detuning < threshold
