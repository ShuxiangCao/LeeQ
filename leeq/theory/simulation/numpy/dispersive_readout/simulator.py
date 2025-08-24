from typing import List, Tuple, Union, Optional, Dict

import numpy as np

from leeq.theory.simulation.numpy.dispersive_readout.utils import *
from leeq.theory.simulation.numpy.dispersive_readout.physics import ChiShiftCalculator
from leeq.theory.simulation.numpy.dispersive_readout.kerr_physics import KerrBistabilityCalculator
from leeq.theory.simulation.numpy.dispersive_readout.power_sweep_manager import PowerSweepManager


class DispersiveReadoutSimulator:
    """
    Base class for dispersive readout simulators in quantum computing.

    This class provides the basic interface and common functionality for
    simulating dispersive readout of qubits. Concrete implementations
    should inherit from this class and implement the specific physics.

    Methods
    -------
    simulate_trace_with_decay(state, f_prob, noise_std=None, return_states_over_time=False)
        Simulate readout trace including T1 decay during measurement.
    plot_iq(states, f_prob, noise_std=0)
        Plot I-Q trajectories for different quantum states.
    _simulate_trace(state, f_prob, noise_std, return_states_over_time=False)
        Core simulation method (to be implemented by subclasses).

    Examples
    --------
    This is typically used through the concrete implementation:

    >>> simulator = DispersiveReadoutSimulatorSyntheticData(...)
    >>> trace = simulator.simulate_trace_with_decay(state=1, f_prob=7000)
    """

    def simulate_trace_with_decay(
        self,
        state: int,
        f_prob: float,
        noise_std: float = None,
        return_states_over_time: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate a signal trace for a given quantum state including T1 decay.

        This method simulates the realistic scenario where the qubit can decay
        during the readout process, changing the measured signal over time.

        Parameters
        ----------
        state : int
            Initial quantum state of the qubit (0, 1, 2, ...).
        f_prob : float
            Readout probe frequency in MHz.
        noise_std : float, optional
            Standard deviation for additive Gaussian noise. If None, uses
            a default noise level.
        return_states_over_time : bool, optional
            If True, also return the qubit state trajectory during readout.

        Returns
        -------
        np.ndarray or tuple
            If return_states_over_time=False: Complex signal trace.
            If return_states_over_time=True: Tuple of (signal_trace, state_trajectory).

        Notes
        -----
        The simulation includes:
        1. Exponential decay from initial state with rate 1/T1
        2. Random decay times sampled from exponential distribution
        3. State-dependent dispersive shifts affecting resonator frequency
        4. Finite resonator bandwidth and response
        5. Additive measurement noise

        Examples
        --------
        Basic usage:

        >>> trace = simulator.simulate_trace_with_decay(state=1, f_prob=7000)
        >>> print(f"Trace shape: {trace.shape}")

        Include state trajectory:

        >>> trace, states = simulator.simulate_trace_with_decay(
        ...     state=2, f_prob=7000, return_states_over_time=True
        ... )
        >>> print(f"Final state: {states[-1]}")
        """

        # Get the time list based on the sampling rate and the envelope width
        t_list = get_t_list(self.sampling_rate, width=self.width)

        t = 0
        current_state = state

        states_over_time = []

        while t < t_list[-1]:
            states_over_time.append((t, current_state))
            if current_state == 0:
                break
            expected_duration = self.t1s[current_state - 1]
            t += np.random.exponential(scale=expected_duration)
            current_state -= 1

        data, states_at_t = self._simulate_trace(
            state=states_over_time, f_prob=f_prob, noise_std=noise_std, return_states_over_time=True
        )

        if return_states_over_time:
            return data, np.asarray(states_at_t)
        else:
            return data

    def plot_iq(self, states: List[int], f_prob: float, noise_std: float = 0) -> None:
        """
        Plot I-Q trajectories for different quantum states.

        This method generates and plots the cumulative I-Q signal trajectories
        for multiple qubit states, allowing visualization of state discrimination.

        Parameters
        ----------
        states : List[int]
            List of quantum states to plot (e.g., [0, 1, 2]).
        f_prob : float
            Readout probe frequency in MHz.
        noise_std : float, optional
            Standard deviation for additive Gaussian noise, by default 0.

        Notes
        -----
        The plot shows cumulative I-Q trajectories where:
        - I (in-phase) and Q (quadrature) are real and imaginary parts
        - Different states trace different paths due to dispersive shifts
        - Final endpoints cluster by state, enabling discrimination
        - Noise causes trajectory spreading

        The cumulative sum amplifies small dispersive shifts, making
        state-dependent frequency differences visible as trajectory separation.

        Examples
        --------
        Plot ground and excited states:

        >>> simulator.plot_iq(states=[0, 1], f_prob=7000, noise_std=0.1)

        Plot multiple levels:

        >>> simulator.plot_iq(states=[0, 1, 2, 3], f_prob=7000)
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        for state in states:
            trace = self._simulate_trace(state, f_prob, noise_std)
            trajectory = np.cumsum(trace)
            Is = np.real(trajectory)
            Qs = np.imag(trajectory)
            plt.plot(Is, Qs, label=f"|{state}⟩", linewidth=2)

        plt.xlabel("In-phase (I)")
        plt.ylabel("Quadrature (Q)")
        plt.title("I-Q Trajectories for Different Qubit States")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.show()


class DispersiveReadoutSimulatorSyntheticData(DispersiveReadoutSimulator):
    """
    A simulator for dispersive readout in quantum computing with physics-based chi shifts.

    This class simulates the readout process of a transmon qubit coupled to a resonator
    in the dispersive regime. It can use either manually specified chi shifts (legacy mode)
    or physics-based calculations that account for anharmonicity and multi-level effects.

    The simulator generates synthetic I/Q traces that include:
    - Dispersive frequency shifts based on qubit state
    - Resonator response with finite bandwidth
    - T1 decay during readout
    - Additive noise

    Attributes
    ----------
    f_r : float
        Resonance frequency of the readout resonator in MHz.
    kappa : float
        Bandwidth (linewidth) of the readout resonator in MHz.
    chis : List[float]
        Chi shifts for different quantum states in MHz. Either provided
        explicitly or calculated using physics model.
    amp : float
        Amplitude of the readout signal (dimensionless).
    baseline : float
        Baseline level of the signal (dimensionless).
    width : float
        Width of the signal pulse in microseconds.
    rise : float
        Rise time of the signal pulse in microseconds.
    trunc : float
        Truncation factor for the signal pulse envelope.
    sampling_rate : float
        Sampling rate for generating the signal in MHz.
    use_physics_model : bool
        Whether to use physics-based chi calculation instead of manual chi values.
    anharmonicity : float
        Transmon anharmonicity in MHz (negative, typically -200 to -300).
    coupling_strength : float
        Qubit-resonator coupling strength in MHz (typically 50-200).
    f_q : float
        Qubit frequency (|0⟩ → |1⟩ transition) in MHz.
    num_levels : int
        Number of transmon energy levels to include in calculations.
    t1s : List[float]
        T1 relaxation times for each level in microseconds.

    Examples
    --------
    Create simulator with physics-based chi calculation:

    >>> simulator = DispersiveReadoutSimulatorSyntheticData.from_physics_model(
    ...     f_r=7000,              # 7 GHz resonator
    ...     f_q=5000,              # 5 GHz qubit
    ...     anharmonicity=-250,    # -250 MHz anharmonicity
    ...     coupling_strength=100, # 100 MHz coupling
    ...     kappa=1.0,            # 1 MHz linewidth
    ...     num_levels=4
    ... )

    Create simulator with manual chi shifts (legacy mode):

    >>> simulator = DispersiveReadoutSimulatorSyntheticData(
    ...     f_r=7000,
    ...     kappa=1.0,
    ...     chis=[0, -0.25, -0.5, -0.75],  # Manual chi shifts
    ...     use_physics_model=False
    ... )

    Generate readout traces:

    >>> trace_0 = simulator._simulate_trace(state=0, f_prob=7000, noise_std=0.1)
    >>> trace_1 = simulator._simulate_trace(state=1, f_prob=7000, noise_std=0.1)

    Notes
    -----
    The physics model calculates chi shifts using:
    χ_n = Σ_m |g_nm|² / (ω_r - ω_nm)

    where g_nm are coupling matrix elements and ω_nm are transition frequencies.
    This provides more accurate modeling than constant chi approximations.

    See Also
    --------
    ChiShiftCalculator : Physics-based chi shift calculation
    VirtualTransmon : Transmon device model with physics-based chi
    """

    def __init__(
        self,
        f_r: float,
        kappa: float,
        chis: Optional[List[float]] = None,
        amp: float = 1,
        baseline: float = 0.1,
        width: float = 10,
        rise: float = 0.0001,
        trunc: float = 1.2,
        sampling_rate: float = 1e3,
        t1s: List[float] = ([100.0, 50.0, 100 / 3],),
        use_physics_model: bool = False,
        anharmonicity: float = -250,
        coupling_strength: float = 50,
        f_q: Optional[float] = None,
        num_levels: int = 4,
        use_kerr_nonlinearity: bool = False,
        kerr_coefficient: Optional[float] = None,
    ):
        """
        Initialize dispersive readout simulator.

        Parameters
        ----------
        f_r : float
            Resonator frequency in MHz.
        kappa : float
            Resonator bandwidth (FWHM) in MHz.
        chis : Optional[List[float]], optional
            Manual chi shifts in MHz. If None and use_physics_model=True,
            chi shifts will be calculated automatically.
        amp : float, optional
            Signal amplitude (dimensionless), by default 1.
        baseline : float, optional
            Baseline signal level (dimensionless), by default 0.1.
        width : float, optional
            Pulse width in microseconds, by default 10.
        rise : float, optional
            Pulse rise time in microseconds, by default 0.0001.
        trunc : float, optional
            Pulse truncation factor, by default 1.2.
        sampling_rate : float, optional
            Sampling rate in MHz, by default 1000.
        t1s : List[float], optional
            T1 relaxation times for each level in microseconds, by default [100.0, 50.0, 33.33].
        use_physics_model : bool, optional
            Whether to calculate chi shifts using physics model, by default False.
        anharmonicity : float, optional
            Transmon anharmonicity in MHz, by default -250.
        coupling_strength : float, optional
            Qubit-resonator coupling in MHz, by default 50.
        f_q : Optional[float], optional
            Qubit frequency in MHz. If None, defaults to f_r - 1000.
        num_levels : int, optional
            Number of transmon levels, by default 4.
        use_kerr_nonlinearity : bool, optional
            Whether to enable Kerr nonlinearity effects, by default False.
        kerr_coefficient : Optional[float], optional
            Kerr coefficient in Hz. If None and use_kerr_nonlinearity=True,
            it will be calculated from transmon parameters.
        """
        # Store basic parameters
        self.f_r = f_r
        self.kappa = kappa
        self.amp = amp
        self.baseline = baseline
        self.sampling_rate = sampling_rate
        self.rise = rise
        self.trunc = trunc
        self.width = width
        self.t1s = t1s

        # Store physics parameters
        self.use_physics_model = use_physics_model
        self.anharmonicity = anharmonicity
        self.coupling_strength = coupling_strength
        self.num_levels = num_levels

        # Set qubit frequency - default to f_r - 1000 MHz if not provided
        if f_q is None:
            self.f_q = f_r - 1000  # Default detuning of 1 GHz
        else:
            self.f_q = f_q

        # Initialize Kerr nonlinearity
        self.use_kerr_nonlinearity = use_kerr_nonlinearity
        if use_kerr_nonlinearity:
            self.kerr_calculator = KerrBistabilityCalculator()
            self.power_sweep_manager = PowerSweepManager(self.kerr_calculator)
            # Calculate or use provided Kerr coefficient
            if kerr_coefficient is None:
                self.kerr_coefficient = self.kerr_calculator.calculate_kerr_coefficient(
                    f_r=self.f_r, f_q=self.f_q, anharmonicity=anharmonicity, g=coupling_strength, num_levels=num_levels
                )
            else:
                self.kerr_coefficient = kerr_coefficient
            # Track current branch for hysteresis
            self.current_branch = "lower"
        else:
            self.kerr_calculator = None
            self.power_sweep_manager = None
            self.kerr_coefficient = None
            self.current_branch = None

        # Calculate or use provided chi shifts
        if use_physics_model and chis is None:
            # Use physics model to calculate chi shifts
            calculator = ChiShiftCalculator()
            self.chis = calculator.calculate_chi_shifts(
                f_r=self.f_r, f_q=self.f_q, anharmonicity=anharmonicity, g=coupling_strength, num_levels=num_levels
            ).tolist()  # Convert to list for consistency
        elif chis is not None:
            # Use explicitly provided chi shifts (backward compatibility)
            self.chis = list(chis)  # Ensure it's a list
        else:
            # Default legacy chi shifts for backward compatibility
            self.chis = [0, -0.25, -0.5, -0.75]

        # Ensure we have enough chi values for num_levels
        if len(self.chis) < num_levels:
            # Extend with linear scaling if needed
            last_chi = self.chis[-1]
            step = last_chi / (len(self.chis) - 1) if len(self.chis) > 1 else -0.25
            while len(self.chis) < num_levels:
                self.chis.append(self.chis[-1] + step)

    @classmethod
    def from_physics_model(
        cls,
        f_r: float,
        f_q: float,
        anharmonicity: float,
        coupling_strength: float,
        kappa: float,
        num_levels: int = 4,
        use_kerr_nonlinearity: bool = False,
        **kwargs,
    ) -> "DispersiveReadoutSimulatorSyntheticData":
        """
        Factory method to create simulator using physics-based chi calculation.

        This convenience method creates a simulator instance with automatically
        calculated chi shifts based on transmon physics, rather than requiring
        manual specification of chi values.

        Parameters
        ----------
        f_r : float
            Resonator frequency in MHz. Typical values: 6000-8000 MHz.
        f_q : float
            Qubit frequency (|0⟩ → |1⟩ transition) in MHz. Typical: 4000-6000 MHz.
        anharmonicity : float
            Transmon anharmonicity in MHz. Must be negative, typical: -200 to -300 MHz.
        coupling_strength : float
            Qubit-resonator coupling strength in MHz. Typical: 50-200 MHz.
            Must satisfy dispersive condition g << |f_r - f_q|.
        kappa : float
            Resonator bandwidth (FWHM linewidth) in MHz. Typical: 0.5-2 MHz.
        num_levels : int, optional
            Number of transmon energy levels to include, by default 4.
            More levels improve accuracy but increase computation time.
        use_kerr_nonlinearity : bool, optional
            Whether to enable Kerr nonlinearity effects, by default False.
        **kwargs
            Additional keyword arguments passed to the constructor, such as:
            - amp : signal amplitude
            - width : pulse width
            - sampling_rate : digitization rate
            - t1s : relaxation times

        Returns
        -------
        DispersiveReadoutSimulatorSyntheticData
            Simulator instance with physics-calculated chi shifts.

        Examples
        --------
        Create simulator for typical transmon parameters:

        >>> simulator = DispersiveReadoutSimulatorSyntheticData.from_physics_model(
        ...     f_r=7000,              # 7 GHz resonator
        ...     f_q=5000,              # 5 GHz qubit
        ...     anharmonicity=-250,    # -250 MHz anharmonicity
        ...     coupling_strength=100, # 100 MHz coupling
        ...     kappa=1.0,            # 1 MHz linewidth
        ...     num_levels=4,          # Include |0⟩, |1⟩, |2⟩, |3⟩
        ...     width=2.0,            # 2 μs readout pulse
        ...     sampling_rate=1000     # 1 GS/s sampling
        ... )

        Check calculated chi shifts:

        >>> print(f"Chi shifts: {simulator.chis} MHz")

        Notes
        -----
        This method automatically enables the physics model by setting
        use_physics_model=True and chis=None, causing chi shifts to be
        calculated using the ChiShiftCalculator.
        """
        return cls(
            f_r=f_r,
            kappa=kappa,
            f_q=f_q,
            anharmonicity=anharmonicity,
            coupling_strength=coupling_strength,
            num_levels=num_levels,
            use_physics_model=True,
            use_kerr_nonlinearity=use_kerr_nonlinearity,
            chis=None,  # Will be calculated by physics model
            **kwargs,
        )

    def _simulate_trace(
        self,
        state: Union[int, List[Tuple[float, int]]],
        f_prob: float,
        noise_std: float = 0,
        return_states_over_time: bool = False,
    ) -> np.ndarray:
        """
        Core method to simulate dispersive readout signal trace.

        This method generates the complex-valued readout signal by:
        1. Computing state-dependent resonator frequency shifts (chi shifts)
        2. Modeling resonator response with finite bandwidth
        3. Adding measurement noise

        Parameters
        ----------
        state : Union[int, List[Tuple[float, int]]]
            Either a single quantum state (int) or a list of (time, state) tuples
            for time-dependent state evolution during readout.
        f_prob : float
            Probe frequency in MHz used for readout.
        noise_std : float, optional
            Standard deviation of additive Gaussian noise, by default 0.
        return_states_over_time : bool, optional
            If True, also return the state at each time point, by default False.

        Returns
        -------
        np.ndarray or tuple
            Complex signal trace. If return_states_over_time=True, returns
            tuple of (signal, states_at_each_time).

        Notes
        -----
        The signal generation process:
        1. Pulse envelope applied to probe
        2. State-dependent frequency shift: f_eff = f_prob + chi[state]
        3. Lorentzian resonator response with bandwidth kappa
        4. Complex amplitude calculation
        5. Noise addition

        For time-dependent states, the chi shift changes during the pulse
        according to the provided state trajectory.
        """
        # Generate pulse envelope
        envelope = soft_square(
            sampling_rate=self.sampling_rate,
            amp=self.amp,
            phase=0,
            width=self.width,
            rise=self.rise,
            trunc=self.trunc,
        )

        # Generate time points corresponding to envelope samples
        t_list = get_t_list(self.sampling_rate, len(envelope) / self.sampling_rate)

        assert len(t_list) == len(envelope), "Time and envelope arrays must have same length"

        if isinstance(state, int):
            state = [(0, state)]

        chis_ = {
            i: root_lorentzian(
                f=f_prob,
                f0=self.f_r + self.chis[i],
                kappa=self.kappa,
                amp=self.amp,
                baseline=self.baseline,
            )
            for i in range(len(self.chis))
        }
        lorentzian_values = chis_

        # Find the state at different time points
        state_at_t = np.zeros(len(t_list))
        resonator_frequency_at_t = np.zeros(len(t_list))
        lorentzian_value_at_t = np.zeros(len(t_list), dtype=complex)

        # sort the state by t0
        state = sorted(state, key=lambda x: x[0])

        for t0, s in state:
            t_idx = t_list >= t0
            state_at_t[t_idx] = s
            resonator_frequency_at_t[t_idx] = self.f_r + self.chis[s]
            lorentzian_value_at_t[t_idx] = lorentzian_values[s]

        signal = lorentzian_value_at_t * envelope

        noise = np.random.normal(scale=noise_std, size=signal.shape) + 1j * np.random.normal(
            scale=noise_std, size=signal.shape
        )

        if return_states_over_time:
            return signal + noise, np.asarray(state_at_t)

        return signal + noise

    def _simulate_trace_with_bistability(
        self, state: Union[int, List[Tuple[float, int]]], f_probe: float, power: float, noise_std: float = 0
    ) -> np.ndarray:
        """
        Simulate trace accounting for Kerr bistability and S-curve response.

        This method extends the basic trace simulation to include Kerr nonlinearity
        effects, which can cause bistable behavior and S-curve response at high powers.

        Parameters
        ----------
        state : Union[int, List[Tuple[float, int]]]
            Quantum state or list of (time, state) tuples.
        f_probe : float
            Probe frequency in MHz.
        power : float
            Drive power (dimensionless units).
        noise_std : float, optional
            Standard deviation of additive Gaussian noise, by default 0.

        Returns
        -------
        np.ndarray
            Complex signal trace including bistability effects.

        Notes
        -----
        When Kerr nonlinearity is disabled, this method falls back to the
        standard _simulate_trace method for backward compatibility.

        In the Kerr regime, the method:
        1. Finds all steady-state solutions to the cubic equation
        2. Identifies stable branches using stability analysis
        3. Selects appropriate branch based on hysteresis history
        4. Handles transitions between bistable branches
        """
        # Fall back to standard simulation if Kerr is disabled
        if not self.use_kerr_nonlinearity:
            return self._simulate_trace(state, f_probe, noise_std)

        # Convert drive power to amplitude for physics calculation
        # Power scaling: assume linear relationship for now
        drive_amplitude = np.sqrt(power * self.kappa)

        # Find all steady-state solutions
        solutions = self.kerr_calculator.find_steady_state_solutions(
            omega_drive=f_probe * 2 * np.pi * 1e6,  # Convert MHz to rad/s
            omega_r=self.f_r * 2 * np.pi * 1e6,  # Convert MHz to rad/s
            kappa=self.kappa * 2 * np.pi * 1e6,  # Convert MHz to rad/s
            kerr_coeff=self.kerr_coefficient,
            drive_amplitude=drive_amplitude,
        )

        # Filter for stable solutions
        stable_solutions = []
        for sol in solutions:
            if self.kerr_calculator.check_solution_stability(
                amplitude=sol,
                omega_drive=f_probe * 2 * np.pi * 1e6,
                omega_r=self.f_r * 2 * np.pi * 1e6,
                kappa=self.kappa * 2 * np.pi * 1e6,
                kerr_coeff=self.kerr_coefficient,
            ):
                stable_solutions.append(sol)

        # Select appropriate solution based on current branch and hysteresis
        if len(stable_solutions) == 1:
            # Single stable solution (linear or high-power regime)
            amplitude = stable_solutions[0]
        elif len(stable_solutions) == 2:
            # Bistable regime - use branch selection with hysteresis
            amplitude = self._select_branch_with_hysteresis(stable_solutions, power)
        else:
            # No stable solutions found - fallback to zero
            amplitude = 0.0

        # Generate signal using selected amplitude
        # Apply the same envelope and noise as standard simulation
        envelope = soft_square(
            sampling_rate=self.sampling_rate,
            amp=self.amp,
            phase=0,
            width=self.width,
            rise=self.rise,
            trunc=self.trunc,
        )

        # Scale amplitude by envelope and add noise
        signal = amplitude * envelope
        noise = np.random.normal(scale=noise_std, size=signal.shape) + 1j * np.random.normal(
            scale=noise_std, size=signal.shape
        )

        return signal + noise

    def _select_branch_with_hysteresis(self, stable_solutions: List[complex], power: float) -> complex:
        """
        Select branch in bistable regime using hysteresis logic.

        Parameters
        ----------
        stable_solutions : List[complex]
            List of stable steady-state solutions (should be 2 in bistable regime).
        power : float
            Drive power level.

        Returns
        -------
        complex
            Selected solution amplitude.

        Notes
        -----
        This method implements proper hysteresis behavior:
        - Forward sweep: stay on lower branch until upper jump point
        - Backward sweep: stay on upper branch until lower jump point
        - Jump thresholds differ, creating hysteresis loop
        """
        if len(stable_solutions) != 2:
            return stable_solutions[0] if stable_solutions else 0.0

        # Sort solutions by amplitude (lower and upper branches)
        solutions_by_amplitude = sorted(stable_solutions, key=lambda x: abs(x))
        lower_branch = solutions_by_amplitude[0]
        upper_branch = solutions_by_amplitude[1]

        # Simple hysteresis logic based on current branch
        # In a full implementation, this would track jump powers more precisely
        if self.current_branch == "lower":
            # Check if we should jump to upper branch
            # For now, use a simple power threshold
            P_c = self.kerr_calculator.find_bifurcation_power(self.kerr_coefficient, self.kappa * 2 * np.pi * 1e6)
            if power > 1.5 * P_c:  # Jump-up threshold
                self.current_branch = "upper"
                return upper_branch
            else:
                return lower_branch
        else:  # current_branch == 'upper'
            # Check if we should jump to lower branch
            P_c = self.kerr_calculator.find_bifurcation_power(self.kerr_coefficient, self.kappa * 2 * np.pi * 1e6)
            if power < 0.8 * P_c:  # Jump-down threshold
                self.current_branch = "lower"
                return lower_branch
            else:
                return upper_branch

    def simulate_power_sweep_with_hysteresis(
        self, f_probe: float, powers: np.ndarray, state: int = 0, direction: str = "up"
    ) -> List[complex]:
        """
        Simulate resonator response vs power showing S-curve and hysteresis.

        This method performs a power sweep at a fixed probe frequency to
        demonstrate the S-curve response and hysteresis characteristic of
        Kerr resonators in the bistable regime.

        Parameters
        ----------
        f_probe : float
            Fixed probe frequency for the sweep in MHz.
        powers : np.ndarray
            Array of power values to sweep.
        state : int, optional
            Quantum state of the system, by default 0.
        direction : str, optional
            Sweep direction - 'up' for increasing power, 'down' for decreasing,
            by default 'up'.

        Returns
        -------
        List[complex]
            Complex response amplitudes for each power point.

        Notes
        -----
        The direction parameter affects hysteresis behavior:
        - 'up': Start on lower branch, may jump to upper branch at high power
        - 'down': Start on upper branch, may jump to lower branch at low power

        This creates the characteristic hysteresis loop where the jump-up
        and jump-down powers are different.
        """
        if not self.use_kerr_nonlinearity:
            # Fall back to standard simulation without power dependence
            return [self._simulate_trace(state, f_probe, 0) for _ in powers]

        responses = []

        # Set initial branch based on sweep direction
        self.current_branch = "lower" if direction == "up" else "upper"

        power_sequence = powers if direction == "up" else np.flip(powers)

        for power in power_sequence:
            response = self._simulate_trace_with_bistability(state, f_probe, power, noise_std=0)
            # Extract the amplitude (average over pulse envelope)
            avg_amplitude = np.mean(response)
            responses.append(avg_amplitude)

        # Reverse responses for down sweep to maintain power order
        if direction == "down":
            responses = list(reversed(responses))

        return responses

    def simulate_three_regimes_demo(self, f_range: np.ndarray, state: int = 0) -> Dict[str, List[complex]]:
        """
        Demonstrate all three power regimes in one simulation.

        This method showcases the three distinct power regimes of Kerr resonators:
        1. Linear regime: Single Lorentzian response
        2. Bistable regime: S-curve with hysteresis
        3. High-power regime: Single response at shifted frequency

        Parameters
        ----------
        f_range : np.ndarray
            Array of frequencies to sweep in MHz.
        state : int, optional
            Quantum state of the system, by default 0.

        Returns
        -------
        Dict[str, List[complex]]
            Dictionary with keys 'linear', 'bistable', 'high_power' containing
            the response for each regime.

        Notes
        -----
        This method is useful for visualization and validation of the Kerr
        physics implementation. It automatically selects appropriate power
        levels for each regime based on the calculated bifurcation power.
        """
        if not self.use_kerr_nonlinearity:
            # Return identical responses for all regimes if Kerr is disabled
            standard_response = [self._simulate_trace(state, f, 0) for f in f_range]
            return {
                "linear": standard_response.copy(),
                "bistable": standard_response.copy(),
                "high_power": standard_response.copy(),
            }

        # Calculate critical power for regime boundaries
        P_c = self.kerr_calculator.find_bifurcation_power(
            self.kerr_coefficient,
            self.kappa * 2 * np.pi * 1e6,  # Convert to rad/s
        )

        # Select powers for each regime
        regimes = {
            "linear": 0.3 * P_c,  # Well below bifurcation
            "bistable": 1.2 * P_c,  # In bistable region
            "high_power": 10 * P_c,  # Well above bifurcation
        }

        results = {}
        for regime_name, power in regimes.items():
            # Reset to lower branch for consistent starting point
            self.current_branch = "lower"

            regime_responses = []
            for f in f_range:
                response = self._simulate_trace_with_bistability(state, f, power, noise_std=0)
                # Extract average amplitude
                avg_amplitude = np.mean(response)
                regime_responses.append(avg_amplitude)

            results[regime_name] = regime_responses

        return results

    pass


class DispersiveReadoutSimulatorRealData(DispersiveReadoutSimulator):
    """
    A simulator for dispersive readout in quantum computing.
    """

    def __init__(
        self,
        data_traces: np.ndarray,
        sampling_rate: float = 1e3,
        t1s: List[float] = (None),
    ):
        """
        Data trace is complex and in the following shape: (qubit_state, num_traces, num_samples)
        The qubit state is in the order of 0, 1, 2, 3.
        Args:
            data_traces: data traces for different qubit states,
            sampling_rate: sampling rate of the data traces,
            t1s: t1s for different qubit states,
        """
        if t1s is None:
            t1s = [100.0, 50.0, 100 / 3]
        self.data_traces = data_traces
        self.data_traces_mean = np.mean(data_traces, axis=1)
        self.data_traces_std = np.std(data_traces, axis=1)
        self.sampling_rate = sampling_rate
        self.width = self.data_traces_mean.shape[-1] / self.sampling_rate
        self.t1s = t1s

    def _simulate_trace(
        self,
        state: Union[int, List[Tuple[float, int]]],
        f_prob: float = 0,  # not used
        noise_std: float = None,
        return_states_over_time: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulates a signal trace for a given quantum state.

        Args:
            state (int): Quantum state to simulate.
            f_prob (float): The readout frequency used for probing the resonator.
            noise_std (float): Standard deviation for noise.
            return_states_over_time: whether to return the states over time.
        Returns:
            np.ndarray: Simulated signal trace with noise.

        """
        if noise_std is None:
            noise_std = self.data_traces_std.mean()

        if isinstance(state, int):
            state = [(0, state)]

        t_list = get_t_list(self.sampling_rate, self.width)  # np.zeros_like(self.data_traces[0, :, :], dtype=complex)

        # Find the state at different time points
        state_at_t = np.zeros(len(t_list))
        signal_at_t = np.zeros(len(t_list), dtype=complex)

        # sort the state by t0
        state = sorted(state, key=lambda x: x[0])

        for t0, s in state:
            t_idx = t_list >= t0
            state_at_t[t_idx] = s
            signal_at_t[t_idx] = self.data_traces_mean[s, t_idx]

        noise = np.random.normal(scale=noise_std, size=signal_at_t.shape) + 1j * np.random.normal(
            scale=noise_std, size=signal_at_t.shape
        )
        signal_at_t = signal_at_t + noise

        if return_states_over_time:
            return signal_at_t, np.asarray(state_at_t)

        return signal_at_t


if __name__ == "__main__":
    sim = DispersiveReadoutSimulatorSyntheticData(
        f_r=9000,
        kappa=0.5,
        chis=[0, -0.25, -0.5, -0.75],
        t1s=[1.0, 5.0, 10 / 3],
        width=10,
    )
    data = sim.simulate_trace_with_decay(state=1, f_prob=9000)

    from matplotlib import pyplot as plt

    plt.plot(data.real, data.imag)
    plt.show()

    # sim.plot_iq(
    #    states=[0, [(0, 1), (2, 0)], [(0, 2), (2, 1), (4, 0)], 3],
    #    f_prob=9000,
    #    noise_std=5,
    # )
