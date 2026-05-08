"""
Transmon energy level calculations and physics utilities.

This module provides functions for calculating transmon energy levels,
transition frequencies, coupling matrix elements, and thermal populations.
It implements the standard transmon Hamiltonian with anharmonicity corrections.

The transmon is modeled as a Duffing oscillator with Hamiltonian:
H = ω_q a†a + (α/2) a†a†aa

where α < 0 is the anharmonicity that makes the transmon charge-insensitive.

Functions
---------
calculate_transmon_energies
    Compute energy levels En = nωq + n(n-1)α/2
calculate_transition_frequencies
    Compute transition frequencies ωn,n+1 = ωq + nα
calculate_coupling_matrix_elements
    Compute g_nm = g√max(n,m) for adjacent levels
get_level_populations
    Calculate thermal populations using Boltzmann distribution
effective_anharmonicity
    Get effective anharmonicity for specific transitions
ac_stark_shift
    Calculate AC Stark shifts from off-resonant driving

Examples
--------
Calculate first few energy levels:

>>> energies = calculate_transmon_energies(
...     f_q=5000,           # 5 GHz qubit frequency
...     anharmonicity=-250, # -250 MHz anharmonicity
...     num_levels=4
... )
>>> print(f"Energies: {energies} MHz")

Get transition frequencies:

>>> transitions = calculate_transition_frequencies(
...     f_q=5000, anharmonicity=-250, num_levels=4
... )
>>> print(f"Transitions: {transitions} MHz")
>>> # Output: [5000, 4750, 4500] MHz for |0⟩→|1⟩, |1⟩→|2⟩, |2⟩→|3⟩

Notes
-----
This module assumes:
- Transmon operates in charge-insensitive regime (EJ >> EC)
- Anharmonicity is approximately constant for first few levels
- Harmonic oscillator coupling matrix elements
- Rotating wave approximation for AC Stark shifts

References
----------
[1] Koch et al., "Charge-insensitive qubit design", Phys. Rev. A 76, 042319 (2007)
[2] Schreier et al., "Suppressing charge noise decoherence", Phys. Rev. B 77, 180502(R) (2008)
"""

import numpy as np
from typing import List, Tuple


def calculate_transmon_energies(f_q: float, anharmonicity: float, num_levels: int = 4) -> np.ndarray:
    """
    Calculate transmon energy levels including anharmonicity.

    Computes the energy eigenvalues of the transmon Hamiltonian with
    leading-order anharmonic correction. The ground state energy is
    set to zero for convenience.

    Parameters
    ----------
    f_q : float
        Qubit frequency (|0⟩ → |1⟩ transition frequency) in MHz.
        Typical values: 4000-6000 MHz.
    anharmonicity : float
        Anharmonicity parameter in MHz. Must be negative for transmons.
        Typical values: -200 to -300 MHz. Controls level spacing.
    num_levels : int, optional
        Number of energy levels to calculate, by default 4.

    Returns
    -------
    np.ndarray
        Array of energy levels [E_0, E_1, E_2, ...] in MHz.
        Ground state E_0 = 0 by convention.

    Notes
    -----
    The energy formula is:
    E_n = nω_q + n(n-1)α/2

    This gives level spacings:
    E_1 - E_0 = ω_q
    E_2 - E_1 = ω_q + α
    E_3 - E_2 = ω_q + 2α

    The anharmonicity causes red-detuning of higher transitions,
    which is essential for selective addressing of qubit levels.

    Examples
    --------
    >>> energies = calculate_transmon_energies(5000, -250, 4)
    >>> print(f"Energy levels: {energies}")
    Energy levels: [   0. 5000. 9750. 14250.] MHz

    >>> level_spacings = np.diff(energies)
    >>> print(f"Level spacings: {level_spacings}")
    Level spacings: [5000. 4750. 4500.] MHz
    """
    n = np.arange(num_levels)
    energies = n * f_q + n * (n - 1) * anharmonicity / 2
    return energies


def calculate_transition_frequencies(f_q: float, anharmonicity: float, num_levels: int = 4) -> np.ndarray:
    """
    Calculate transition frequencies between adjacent levels.

    For a transmon with anharmonicity, each transition frequency
    is red-detuned from the fundamental by nα.

    Parameters
    ----------
    f_q : float
        Qubit frequency (|0⟩ → |1⟩ transition) in MHz.
    anharmonicity : float
        Anharmonicity parameter in MHz (negative for transmons).
    num_levels : int, optional
        Number of levels (gives num_levels-1 transitions), by default 4.

    Returns
    -------
    np.ndarray
        Array of transition frequencies [ω_01, ω_12, ω_23, ...] in MHz.

    Notes
    -----
    The transition frequency formula is:
    ω_n,n+1 = E_{n+1} - E_n = ω_q + nα

    This gives:
    - |0⟩ → |1⟩: ω_01 = ω_q (fundamental transition)
    - |1⟩ → |2⟩: ω_12 = ω_q + α (red-detuned by |α|)
    - |2⟩ → |3⟩: ω_23 = ω_q + 2α (red-detuned by 2|α|)

    The anharmonicity enables selective addressing: a pulse resonant
    with |0⟩ → |1⟩ will be off-resonant with higher transitions.

    Examples
    --------
    >>> transitions = calculate_transition_frequencies(5000, -250, 4)
    >>> print(f"Transition frequencies: {transitions} MHz")
    Transition frequencies: [5000. 4750. 4500.] MHz

    >>> detunings = transitions - transitions[0]
    >>> print(f"Detunings from fundamental: {detunings} MHz")
    Detunings from fundamental: [   0. -250. -500.] MHz
    """
    n = np.arange(num_levels - 1)
    transitions = f_q + n * anharmonicity
    return transitions


def calculate_coupling_matrix_elements(g: float, num_levels: int = 4) -> np.ndarray:
    """
    Calculate coupling matrix elements for transmon-resonator interaction.

    The transmon-resonator coupling follows harmonic oscillator matrix
    elements with √n enhancement for transitions to higher levels.

    Parameters
    ----------
    g : float
        Base coupling strength (|0⟩ ↔ |1⟩ coupling) in MHz.
        Typical values: 50-200 MHz for circuit QED systems.
    num_levels : int, optional
        Number of transmon levels to include, by default 4.

    Returns
    -------
    np.ndarray
        Hermitian coupling matrix of shape (num_levels, num_levels).
        Non-zero elements only between adjacent levels.

    Notes
    -----
    The matrix elements are:
    g_n,n+1 = g √(n+1)

    where the √(n+1) factor comes from harmonic oscillator ladder operators:
    ⟨n|a†|n+1⟩ = √(n+1)

    This enhancement means higher-level transitions couple more strongly:
    - |0⟩ ↔ |1⟩: coupling = g
    - |1⟩ ↔ |2⟩: coupling = g√2 ≈ 1.41g
    - |2⟩ ↔ |3⟩: coupling = g√3 ≈ 1.73g

    Examples
    --------
    >>> coupling_matrix = calculate_coupling_matrix_elements(100, 3)
    >>> print(f"Coupling matrix:\n{coupling_matrix}")
    Coupling matrix:
    [[  0.    100.      0.  ]
     [100.      0.    141.42]
     [  0.    141.42    0.  ]]

    >>> # Extract coupling strengths
    >>> g01 = coupling_matrix[0, 1]
    >>> g12 = coupling_matrix[1, 2]
    >>> print(f"g_01 = {g01:.1f} MHz, g_12 = {g12:.1f} MHz")
    g_01 = 100.0 MHz, g_12 = 141.4 MHz
    """
    coupling_matrix = np.zeros((num_levels, num_levels))

    for n in range(num_levels - 1):
        # Coupling between adjacent levels
        g_nm = g * np.sqrt(n + 1)
        coupling_matrix[n, n + 1] = g_nm
        coupling_matrix[n + 1, n] = g_nm  # Hermitian matrix

    return coupling_matrix


def get_level_populations(temperature: float, f_q: float, anharmonicity: float, num_levels: int = 4) -> np.ndarray:
    """
    Calculate thermal level populations using Boltzmann distribution.

    At finite temperature, higher energy levels have non-zero occupation
    probability according to the Boltzmann distribution. This is important
    for understanding thermal effects in qubit measurements.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin. For T=0, only ground state is populated.
        Typical dilution refrigerator base temperature: ~10-20 mK.
    f_q : float
        Qubit frequency in MHz.
    anharmonicity : float
        Anharmonicity in MHz.
    num_levels : int, optional
        Number of levels to include, by default 4.

    Returns
    -------
    np.ndarray
        Array of level populations [P_0, P_1, P_2, ...] summing to 1.

    Notes
    -----
    The population formula is:
    P_n = exp(-E_n/k_B T) / Z

    where Z = Σ_n exp(-E_n/k_B T) is the partition function.

    For qubit frequencies ~5 GHz and mK temperatures:
    k_B T / hν ~ 10^-3, so thermal excitation is typically negligible.

    Examples
    --------
    Zero temperature (ideal case):

    >>> pops = get_level_populations(0.0, 5000, -250, 4)
    >>> print(f"T=0 populations: {pops}")
    T=0 populations: [1. 0. 0. 0.]

    Finite temperature:

    >>> pops = get_level_populations(0.02, 5000, -250, 4)  # 20 mK
    >>> print(f"T=20mK populations: {pops}")
    T=20mK populations: [0.999 0.001 0.000 0.000]  # Small thermal excitation
    """
    if temperature <= 0:
        # Zero temperature: only ground state populated
        populations = np.zeros(num_levels)
        populations[0] = 1.0
        return populations

    # Convert MHz to K using h/k_B ≈ 4.8e-5 K/MHz
    h_over_kB = 4.8e-5  # K/MHz
    beta = h_over_kB / temperature

    energies = calculate_transmon_energies(f_q, anharmonicity, num_levels)

    # Boltzmann weights (subtract ground state energy)
    weights = np.exp(-beta * (energies - energies[0]))

    # Normalize to get populations
    partition_function = np.sum(weights)
    populations = weights / partition_function

    return populations


def effective_anharmonicity(f_q: float, anharmonicity: float, level: int) -> float:
    """
    Calculate effective anharmonicity for transitions from a given level.

    For the transmon Duffing oscillator model, the anharmonicity is
    approximately constant for the first few levels. This function
    provides the effective anharmonicity for specific transitions.

    Parameters
    ----------
    f_q : float
        Qubit frequency in MHz (not used in current implementation).
    anharmonicity : float
        Anharmonicity parameter in MHz.
    level : int
        Starting level for the transition.

    Returns
    -------
    float
        Effective anharmonicity for transitions from this level in MHz.
        Currently returns the same value for all levels.

    Notes
    -----
    For the simple Duffing model E_n = nω_q + n(n-1)α/2,
    the anharmonicity α is constant across levels.

    In more sophisticated models (e.g., with higher-order corrections),
    the effective anharmonicity could depend on the level number.

    Examples
    --------
    >>> alpha_eff = effective_anharmonicity(5000, -250, 1)
    >>> print(f"Effective anharmonicity: {alpha_eff} MHz")
    Effective anharmonicity: -250 MHz

    Future extensions might include level-dependent corrections:
    >>> # alpha_eff = anharmonicity * (1 + correction_factor * level)
    """
    # For the Duffing oscillator model, anharmonicity is approximately constant
    # for the first few levels. Higher-order corrections could be added here.
    return anharmonicity


def ac_stark_shift(f_drive: float, f_q: float, anharmonicity: float, rabi_frequency: float, num_levels: int = 4) -> np.ndarray:
    """
    Calculate AC Stark shifts due to off-resonant driving.

    When a transmon is driven off-resonantly, virtual photon exchange
    causes energy level shifts proportional to drive intensity.
    This is a simplified second-order perturbation theory calculation.

    Parameters
    ----------
    f_drive : float
        Drive frequency in MHz.
    f_q : float
        Qubit frequency (|0⟩ → |1⟩ transition) in MHz.
    anharmonicity : float
        Anharmonicity in MHz.
    rabi_frequency : float
        Rabi frequency (Ω = gε/ħ) of the drive in MHz, where g is the
        drive coupling and ε is the drive amplitude.
    num_levels : int, optional
        Number of levels to include, by default 4.

    Returns
    -------
    np.ndarray
        Array of AC Stark shifts for each level in MHz.

    Notes
    -----
    This implements a simplified AC Stark shift formula:
    ΔE_n = Σ_m |Ω_nm|^2 / (4Δ_nm)

    where Ω_nm is the drive coupling between levels and Δ_nm is the detuning.

    Important limitations:
    - Only includes nearest-neighbor couplings
    - Uses first-order drive coupling approximation
    - Neglects drive-induced mixing between levels
    - Real AC Stark shifts can be more complex

    For accurate calculations, use full Floquet theory or
    dressed state analysis.

    Examples
    --------
    >>> stark_shifts = ac_stark_shift(
    ...     f_drive=4500,       # Off-resonant drive
    ...     f_q=5000,           # Qubit frequency
    ...     anharmonicity=-250,
    ...     rabi_frequency=10,  # Weak drive
    ...     num_levels=3
    ... )
    >>> print(f"AC Stark shifts: {stark_shifts} MHz")

    Warnings
    --------
    This is a simplified educational implementation. For quantitative
    predictions, use specialized packages like QuTiP with Floquet methods.
    """
    energies = calculate_transmon_energies(f_q, anharmonicity, num_levels)
    stark_shifts = np.zeros(num_levels)

    for n in range(num_levels):
        for m in range(num_levels):
            if m != n and abs(m - n) == 1:  # Adjacent levels only
                transition_freq = energies[m] - energies[n]
                detuning = f_drive - transition_freq

                if abs(detuning) > 1e-6:  # Avoid division by zero
                    # Simplified AC Stark shift formula
                    stark_shifts[n] += (rabi_frequency**2) / (4 * detuning)

    return stark_shifts
