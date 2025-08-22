import numpy as np
import qutip as qt
from typing import List, Dict, Tuple, Optional


class CWSpectroscopySimulator:
    """
    Continuous wave spectroscopy simulator for multi-qubit systems.

    This simulator uses a crosstalk approximation where coupling between qubits
    transfers drive amplitude rather than creating full entanglement. Each qubit
    is simulated independently in its rotating frame after calculating effective
    drives including crosstalk contributions.

    Parameters
    ----------
    simulation_setup : HighLevelSimulationSetup
        Setup containing VirtualTransmon objects and coupling information

    Attributes
    ----------
    virtual_qubits : Dict[int, VirtualTransmon]
        Mapping from channel to virtual qubit
    channels : List[int]
        Sorted list of available channels
    truncation : int
        Truncation level for qubit simulations (max 5)

    Examples
    --------
    >>> setup = HighLevelSimulationSetup(...)
    >>> sim = CWSpectroscopySimulator(setup)
    >>> iq = sim.simulate_spectroscopy_iq(
    ...     drives=[(1, 5000.0, 10.0)],
    ...     readout_params={1: {'frequency': 7000.0, 'amplitude': 5.0}}
    ... )
    """

    def __init__(self, simulation_setup):
        """
        Initialize simulator from HighLevelSimulationSetup.

        Parameters
        ----------
        simulation_setup : HighLevelSimulationSetup
            Setup containing virtual qubits and coupling information

        Raises
        ------
        ValueError
            If simulation_setup is invalid or contains no virtual qubits
        """
        # Validation
        if not hasattr(simulation_setup, '_virtual_qubits'):
            raise ValueError("Invalid simulation setup")

        # Store setup and extract components
        self.setup = simulation_setup
        self.virtual_qubits = simulation_setup._virtual_qubits
        self.channels = sorted(self.virtual_qubits.keys())

        # Validate non-empty
        if not self.channels:
            raise ValueError("No virtual qubits found")

        # Set truncation level
        self.truncation = min(5, list(self.virtual_qubits.values())[0].truncate_level)

        # Cache for single-qubit Hamiltonians
        self._hamiltonian_cache = {}

    def _get_cached_hamiltonian(self, channel: int, freq: float, amp: float):
        """
        Get cached Hamiltonian for single qubit simulation.

        Parameters
        ----------
        channel : int
            Qubit channel
        freq : float
            Drive frequency in MHz
        amp : float
            Drive amplitude in MHz

        Returns
        -------
        qutip.Qobj
            Hamiltonian for the single qubit system
        """
        cache_key = (channel, round(freq, 2), round(amp, 4))

        if cache_key not in self._hamiltonian_cache:
            # Build and cache Hamiltonian
            vqubit = self.virtual_qubits[channel]
            N = self.truncation
            a = qt.destroy(N)
            n = a.dag() * a
            detuning = freq - vqubit.qubit_frequency

            H = 2 * np.pi * (detuning * n
                             + vqubit.anharmonicity / 2 * (n**2 - n)
                             + amp * (a + a.dag()))

            self._hamiltonian_cache[cache_key] = H

        return self._hamiltonian_cache[cache_key]

    def _simulate_single_qubit(self, channel: int, freq: float, amp: float) -> np.ndarray:
        """
        Simulate single qubit response in rotating frame.

        Parameters
        ----------
        channel : int
            Qubit channel
        freq : float
            Drive frequency in MHz
        amp : float
            Drive amplitude in MHz

        Returns
        -------
        np.ndarray
            Population array [P0, P1, P2, ...] for energy levels
        """
        N = self.truncation

        # Get cached Hamiltonian
        H = self._get_cached_hamiltonian(channel, freq, amp)

        # Find dressed ground state
        eigenvalues, eigenstates = H.eigenstates()
        ground_bare = qt.basis(N, 0)
        overlaps = [np.abs(es.overlap(ground_bare))**2 for es in eigenstates]
        dressed_ground = eigenstates[np.argmax(overlaps)]

        # Extract populations
        populations = np.zeros(N)
        for i in range(N):
            populations[i] = np.abs(dressed_ground.overlap(qt.basis(N, i)))**2

        return populations

    def _calculate_effective_drives(self, drives: List[Tuple[int, float, float]]) -> Dict[int, Tuple[float, float]]:
        """
        Calculate effective drives including coupling-induced crosstalk.

        Coupling transfers a fraction of drive amplitude to neighboring qubits:
        effective_amp_neighbor = coupling_strength / detuning * primary_amp

        Parameters
        ----------
        drives : List[Tuple[int, float, float]]
            List of (channel, frequency, amplitude) tuples for qubit drives

        Returns
        -------
        Dict[int, Tuple[float, float]]
            Dictionary mapping channel to (frequency, effective_amplitude)

        Raises
        ------
        ValueError
            If any drive channel is not found in the simulation setup
        """
        # Initialize with direct drives
        effective_drives = {}
        for channel, freq, amp in drives:
            if channel not in self.channels:
                raise ValueError(f"Channel {channel} not found")
            effective_drives[channel] = (freq, amp)

        # Add crosstalk contributions
        for ch_driven, freq_drive, amp_drive in drives:
            vq_driven = self.virtual_qubits[ch_driven]

            for ch_other in self.channels:
                if ch_other == ch_driven:
                    continue

                vq_other = self.virtual_qubits[ch_other]
                coupling = self.setup.get_coupling_strength_by_qubit(vq_driven, vq_other)

                if coupling != 0:
                    detuning = max(abs(freq_drive - vq_other.qubit_frequency), 1.0)
                    transfer = coupling / detuning * amp_drive

                    if ch_other in effective_drives:
                        freq_existing, amp_existing = effective_drives[ch_other]
                        effective_drives[ch_other] = (freq_drive, amp_existing + transfer)
                    else:
                        effective_drives[ch_other] = (freq_drive, transfer)

        return effective_drives

    def simulate_spectroscopy_iq(self,
                                 drives: List[Tuple[int, float, float]],
                                 readout_params: Dict[int, Dict]) -> Dict[int, complex]:
        """
        Simulate full spectroscopy with IQ readout response.

        Parameters
        ----------
        drives : List[Tuple[int, float, float]]
            List of (channel, frequency, amplitude) tuples for qubit drives
        readout_params : Dict[int, Dict]
            Dictionary mapping channel to {'frequency': MHz, 'amplitude': MHz}
            for readout parameters

        Returns
        -------
        Dict[int, complex]
            Dictionary mapping channel to complex IQ response

        Raises
        ------
        ValueError
            If drives list is empty or readout_params is empty
        """
        # Input validation
        if not drives:
            raise ValueError("At least one drive must be specified")
        if not readout_params:
            raise ValueError("Readout parameters must be specified")

        # Calculate effective drives
        effective_drives = self._calculate_effective_drives(drives)

        # Simulate each qubit and get IQ
        iq_responses = {}
        for channel in self.channels:
            # Get populations
            if channel in effective_drives:
                freq, amp = effective_drives[channel]
                populations = self._simulate_single_qubit(channel, freq, amp)
            else:
                populations = np.zeros(self.truncation)
                populations[0] = 1.0

            # Convert to IQ if readout requested
            if channel in readout_params:
                vqubit = self.virtual_qubits[channel]
                f_readout = readout_params[channel]['frequency']
                amp_readout = readout_params[channel]['amplitude']
                
                # Use VirtualTransmon's resonator response with population weighting
                resonator_responses = vqubit.get_resonator_response(
                    f=f_readout, 
                    amp=amp_readout, 
                    baseline=0
                )
                
                # Weight by populations to get effective IQ response
                # Each element of resonator_responses corresponds to a qubit state
                iq = np.dot(populations, resonator_responses)
                iq_responses[channel] = iq

        return iq_responses
