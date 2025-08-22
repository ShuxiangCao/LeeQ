from typing import List, Dict, Tuple, Union
import numpy as np
from .simulator import DispersiveReadoutSimulator


class MultiQubitDispersiveReadoutSimulator(DispersiveReadoutSimulator):
    """Multi-qubit extension of dispersive readout simulator
    
    This class extends the base DispersiveReadoutSimulator to handle multi-qubit
    systems with joint quantum states. It calculates chi shifts for all resonators
    based on the collective state of all qubits and generates realistic I/Q traces.
    
    The simulator supports arbitrary qubit-resonator coupling topologies and 
    includes qubit-qubit interaction effects in the chi shift calculations.
    """
    
    def __init__(self,
                 qubit_frequencies: List[float],
                 qubit_anharmonicities: List[float],
                 resonator_frequencies: List[float],
                 resonator_kappas: List[float],
                 coupling_matrix: Dict[Tuple[str, str], float],
                 **kwargs):
        """
        Initialize multi-qubit dispersive readout simulator.
        
        Parameters
        ----------
        qubit_frequencies : List[float]
            List of qubit frequencies in MHz
        qubit_anharmonicities : List[float]
            List of qubit anharmonicities in MHz (typically negative)
        resonator_frequencies : List[float]
            List of resonator frequencies in MHz
        resonator_kappas : List[float]
            List of resonator linewidths in MHz
        coupling_matrix : Dict[Tuple[str, str], float]
            Dictionary mapping element pairs to coupling strengths in MHz.
            Keys should be tuples like ('Q0', 'R0') for qubit-resonator coupling
            or ('Q0', 'Q1') for qubit-qubit coupling.
        **kwargs
            Additional parameters passed to parent DispersiveReadoutSimulator
        """
        # Store system configuration
        self.n_qubits = len(qubit_frequencies)
        self.n_resonators = len(resonator_frequencies)
        self.qubit_frequencies = np.array(qubit_frequencies)
        self.qubit_anharmonicities = np.array(qubit_anharmonicities)
        self.resonator_frequencies = np.array(resonator_frequencies)
        self.resonator_kappas = np.array(resonator_kappas)
        self.coupling_matrix = coupling_matrix
        
        # Set required properties for compatibility with parent class methods
        # The base DispersiveReadoutSimulator doesn't have __init__, so we set attributes directly
        
        # Default simulation parameters (can be overridden via kwargs)
        self.amp = kwargs.get('amp', 1.0)
        self.baseline = kwargs.get('baseline', 0.1)
        self.width = kwargs.get('width', 10.0)
        self.rise = kwargs.get('rise', 0.0001)
        self.trunc = kwargs.get('trunc', 1.2)
        self.sampling_rate = kwargs.get('sampling_rate', 1e3)
        self.t1s = kwargs.get('t1s', [100.0, 50.0, 100/3])
        
        # For compatibility with parent class methods that expect these
        if self.n_resonators > 0:
            self.f_r = self.resonator_frequencies[0]
            self.kappa = self.resonator_kappas[0]
    
    def _get_state_tuple(self, joint_state: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        """Convert integer state index to tuple representation.
        
        Parameters
        ----------
        joint_state : Union[int, Tuple[int, ...]]
            Either an integer index or tuple of qubit states
            
        Returns
        -------
        Tuple[int, ...]
            Tuple representation (n1, n2, ..., nk) where ni is state of qubit i
            
        Examples
        --------
        For 2 qubits:
        - 0 -> (0, 0)  # |00⟩
        - 1 -> (0, 1)  # |01⟩  
        - 2 -> (1, 0)  # |10⟩
        - 3 -> (1, 1)  # |11⟩
        """
        if isinstance(joint_state, tuple):
            return joint_state
            
        # Convert integer to binary representation
        state_list = []
        for i in range(self.n_qubits):
            state_list.append(joint_state % 2)
            joint_state //= 2
        return tuple(reversed(state_list))
    
    def _validate_state(self, joint_state: Union[int, Tuple[int, ...]]) -> bool:
        """Validate that joint state is within valid bounds.
        
        Parameters
        ----------
        joint_state : Union[int, Tuple[int, ...]]
            State to validate
            
        Returns
        -------
        bool
            True if state is valid, False otherwise
        """
        if isinstance(joint_state, int):
            return 0 <= joint_state < 2**self.n_qubits
        elif isinstance(joint_state, tuple):
            return (len(joint_state) == self.n_qubits and 
                   all(0 <= state <= 1 for state in joint_state))
        return False
    
    def _calculate_chi_shifts(self, joint_state: Tuple[int, ...]) -> np.ndarray:
        """
        Calculate chi shifts for all resonators given joint qubit state.
        
        Using the formula: χᵢ(n₁,n₂,...,nₖ) = Σⱼ |gᵢⱼ|² / (ωᵣᵢ - ωⱼ(n₁,n₂,...,nₖ))
        
        Parameters
        ----------
        joint_state : Tuple[int, ...]
            Tuple of qubit states (n1, n2, ..., nk) where ni ∈ {0, 1}
            
        Returns
        -------
        np.ndarray
            Array of chi shifts for each resonator in MHz
            
        Notes
        -----
        This method includes:
        1. Anharmonicity effects for excited qubit states
        2. Qubit-qubit coupling effects that shift effective qubit frequencies
        3. Numerical stability checks for near-zero detunings
        """
        import warnings
        
        chi_shifts = np.zeros(self.n_resonators)
        
        for res_idx in range(self.n_resonators):
            for q_idx in range(self.n_qubits):
                # Check if this qubit is coupled to this resonator
                coupling_key = (f'Q{q_idx}', f'R{res_idx}')
                if coupling_key in self.coupling_matrix:
                    g = self.coupling_matrix[coupling_key]
                    
                    # Calculate effective qubit frequency including anharmonicity
                    qubit_level = joint_state[q_idx]
                    omega_q = self.qubit_frequencies[q_idx] + \
                              qubit_level * self.qubit_anharmonicities[q_idx]
                    
                    # Add qubit-qubit coupling effects
                    for other_q_idx in range(self.n_qubits):
                        if other_q_idx != q_idx:
                            # Create sorted tuple for consistent key lookup
                            qq_key = tuple(sorted([f'Q{q_idx}', f'Q{other_q_idx}']))
                            if qq_key in self.coupling_matrix and joint_state[other_q_idx] > 0:
                                # Add ZZ-type interaction effect
                                omega_q += self.coupling_matrix[qq_key] * joint_state[other_q_idx]
                    
                    # Calculate detuning and chi contribution
                    detuning = self.resonator_frequencies[res_idx] - omega_q
                    
                    # Check for numerical stability and dispersive regime
                    if abs(detuning) < 1e-6:
                        warnings.warn(
                            f"Near-zero detuning detected for Q{q_idx}-R{res_idx}: "
                            f"Δ = {detuning:.1e} MHz. Chi calculation may be unstable.",
                            UserWarning
                        )
                        continue
                    
                    # Check dispersive regime validity (g/Δ << 1)
                    if abs(g / detuning) > 0.1:
                        warnings.warn(
                            f"Dispersive regime approximation may be invalid for Q{q_idx}-R{res_idx}: "
                            f"g/Δ = {abs(g/detuning):.3f} > 0.1",
                            UserWarning
                        )
                    
                    # Add chi contribution
                    chi_shifts[res_idx] += g**2 / detuning
                    
        return chi_shifts
    
    def simulate_trace(self,
                      joint_state: Union[int, Tuple[int, ...]],
                      resonator_id: int,
                      f_probe: float,
                      noise_std: float = 0,
                      **kwargs) -> np.ndarray:
        """
        Generate readout trace for a single resonator given joint qubit state.
        
        Parameters
        ----------
        joint_state : Union[int, Tuple[int, ...]]
            Either integer index or tuple (n1, n2, ...) of qubit states
        resonator_id : int
            Index of resonator to simulate (0 to n_resonators-1)
        f_probe : float
            Probe frequency in MHz
        noise_std : float, optional
            Noise standard deviation for complex Gaussian noise, by default 0
        **kwargs
            Additional simulation parameters that override instance defaults
            
        Returns
        -------
        np.ndarray
            Complex-valued I/Q trace for the resonator
            
        Raises
        ------
        ValueError
            If joint_state is invalid or resonator_id is out of bounds
            
        Examples
        --------
        >>> sim = MultiQubitDispersiveReadoutSimulator(...)
        >>> trace = sim.simulate_trace((0, 1), 0, 7000, noise_std=0.1)
        >>> print(f"Trace shape: {trace.shape}, dtype: {trace.dtype}")
        """
        # Validate inputs
        if not self._validate_state(joint_state):
            raise ValueError(f"Invalid joint state: {joint_state}")
        if not (0 <= resonator_id < self.n_resonators):
            raise ValueError(f"Resonator ID {resonator_id} out of range [0, {self.n_resonators-1}]")
        
        # Convert to tuple representation
        state_tuple = self._get_state_tuple(joint_state)
        
        # Calculate chi shifts for all resonators
        chi_shifts = self._calculate_chi_shifts(state_tuple)
        
        # Get shifted resonator parameters
        f_shifted = self.resonator_frequencies[resonator_id] - chi_shifts[resonator_id]
        kappa = self.resonator_kappas[resonator_id]
        
        # Import utilities
        from .utils import root_lorentzian, soft_square, get_t_list
        
        # Get simulation parameters (kwargs override instance defaults)
        amp = kwargs.get('amp', getattr(self, 'amp', 1.0))
        baseline = kwargs.get('baseline', getattr(self, 'baseline', 0.1))
        width = kwargs.get('width', getattr(self, 'width', 10.0))
        rise = kwargs.get('rise', getattr(self, 'rise', 0.0001))
        trunc = kwargs.get('trunc', getattr(self, 'trunc', 1.2))
        sampling_rate = kwargs.get('sampling_rate', getattr(self, 'sampling_rate', 1e3))
        phase = kwargs.get('phase', 0.0)
        
        # Generate envelope using soft_square
        envelope = soft_square(
            sampling_rate=int(sampling_rate),
            amp=amp,
            phase=phase,
            width=width,
            rise=rise,
            trunc=trunc
        )
        
        # Calculate resonator response using root_lorentzian
        lorentzian = root_lorentzian(
            f=f_probe,
            f0=f_shifted,
            kappa=kappa,
            amp=amp,
            baseline=baseline
        )
        
        # Combine envelope and lorentzian response
        signal = lorentzian * envelope
        
        # Add noise if requested
        if noise_std > 0:
            # Generate complex Gaussian noise
            noise_real = np.random.normal(0, noise_std, signal.shape)
            noise_imag = np.random.normal(0, noise_std, signal.shape)
            noise = noise_real + 1j * noise_imag
            signal += noise
            
        return signal
    
    def simulate_multiplexed_readout(self,
                                    joint_state: Union[int, Tuple[int, ...]],
                                    probe_frequencies: Union[List[float], np.ndarray],
                                    noise_std: float = 0,
                                    **kwargs) -> List[np.ndarray]:
        """
        Simulate simultaneous readout of multiple resonators.
        
        This method generates I/Q traces for all resonators simultaneously,
        each with their own probe frequency. This simulates frequency-multiplexed
        readout where multiple resonators are probed at once.
        
        Parameters
        ----------
        joint_state : Union[int, Tuple[int, ...]]
            Joint state of all qubits
        probe_frequencies : Union[List[float], np.ndarray]
            List of probe frequencies for each resonator in MHz.
            Length must match number of resonators.
        noise_std : float, optional
            Noise standard deviation for complex Gaussian noise, by default 0
        **kwargs
            Additional simulation parameters passed to simulate_trace
            
        Returns
        -------
        List[np.ndarray]
            List of complex-valued I/Q traces, one for each resonator
            
        Raises
        ------
        ValueError
            If number of probe frequencies doesn't match number of resonators
            
        Examples
        --------
        >>> sim = MultiQubitDispersiveReadoutSimulator(...)
        >>> traces = sim.simulate_multiplexed_readout(
        ...     joint_state=(1, 0),
        ...     probe_frequencies=[7000, 7500],
        ...     noise_std=0.1
        ... )
        >>> print(f"Generated {len(traces)} traces")
        """
        # Validate inputs
        if len(probe_frequencies) != self.n_resonators:
            raise ValueError(
                f"Expected {self.n_resonators} probe frequencies, "
                f"got {len(probe_frequencies)}"
            )
        
        # Generate trace for each resonator
        traces = []
        for res_idx, f_probe in enumerate(probe_frequencies):
            trace = self.simulate_trace(
                joint_state=joint_state,
                resonator_id=res_idx,
                f_probe=f_probe,
                noise_std=noise_std,
                **kwargs
            )
            traces.append(trace)
        
        return traces