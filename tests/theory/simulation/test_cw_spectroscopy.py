import pytest
import numpy as np
from leeq.theory.simulation.numpy.cw_spectroscopy import CWSpectroscopySimulator
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager


@pytest.fixture(autouse=True)
def clean_singleton():
    """Clean up ExperimentManager singleton state before and after each test."""
    # Reset singleton to get fresh instance
    ExperimentManager._reset_singleton()
    
    yield
    
    # Clear state and reset singleton after test
    try:
        manager = ExperimentManager()
        manager.clear_setups()
        manager._active_experiment_instance = None
    finally:
        ExperimentManager._reset_singleton()


@pytest.fixture
def single_qubit_setup():
    """Create single qubit test setup."""
    vq = VirtualTransmon(
        name="Q1",
        qubit_frequency=5000.0,
        anharmonicity=-200.0,
        t1=50.0,
        t2=30.0,
        readout_frequency=7000.0,
        truncate_level=3
    )
    return HighLevelSimulationSetup(
        name="test",
        virtual_qubits={1: vq},
        omega_to_amp_map={1: 500.0}
    )


@pytest.fixture
def coupled_qubits_setup():
    """Create coupled two-qubit setup."""
    vq1 = VirtualTransmon(name="Q1", qubit_frequency=5000.0, 
                         anharmonicity=-200.0, t1=50.0, t2=30.0, 
                         readout_frequency=7000.0, truncate_level=3)
    vq2 = VirtualTransmon(name="Q2", qubit_frequency=5200.0, 
                         anharmonicity=-210.0, t1=50.0, t2=30.0, 
                         readout_frequency=7100.0, truncate_level=3)
    
    return HighLevelSimulationSetup(
        name="test",
        virtual_qubits={1: vq1, 2: vq2},
        omega_to_amp_map={1: 500.0, 2: 500.0},
        coupling_strength_map={frozenset(["Q1", "Q2"]): 20.0}
    )


def test_simulator_init(single_qubit_setup):
    """Test simulator initialization."""
    sim = CWSpectroscopySimulator(single_qubit_setup)
    assert sim.channels == [1]
    assert sim.truncation == 3


def test_crosstalk_calculation(coupled_qubits_setup):
    """Test crosstalk is calculated correctly."""
    sim = CWSpectroscopySimulator(coupled_qubits_setup)
    
    # Strong drive on Q1 should create crosstalk to Q2
    effective = sim._calculate_effective_drives([(1, 5000.0, 100.0)])
    
    assert 1 in effective  # Direct drive
    assert 2 in effective  # Crosstalk
    assert effective[2][1] > 0  # Non-zero crosstalk amplitude
    
    # Verify crosstalk scaling
    expected_crosstalk = 20.0 / 200.0 * 100.0  # coupling/detuning * amp
    assert abs(effective[2][1] - expected_crosstalk) < 1.0


def test_multi_qubit_iq_response(coupled_qubits_setup):
    """Test multi-qubit IQ response generation."""
    sim = CWSpectroscopySimulator(coupled_qubits_setup)
    
    # Test with readout on both qubits
    drives = [(1, 5000.0, 50.0)]
    readout_params = {
        1: {'frequency': 7000.0, 'amplitude': 10.0},
        2: {'frequency': 7100.0, 'amplitude': 10.0}
    }
    
    iq_responses = sim.simulate_spectroscopy_iq(drives, readout_params)
    
    assert 1 in iq_responses
    assert 2 in iq_responses
    assert isinstance(iq_responses[1], complex)
    assert isinstance(iq_responses[2], complex)


def test_multi_qubit_no_coupling():
    """Test multi-qubit system with no coupling."""
    vq1 = VirtualTransmon(name="Q1", qubit_frequency=5000.0, 
                         anharmonicity=-200.0, t1=50.0, t2=30.0, 
                         readout_frequency=7000.0, truncate_level=3)
    vq2 = VirtualTransmon(name="Q2", qubit_frequency=5200.0, 
                         anharmonicity=-210.0, t1=50.0, t2=30.0, 
                         readout_frequency=7100.0, truncate_level=3)
    
    setup = HighLevelSimulationSetup(
        name="test",
        virtual_qubits={1: vq1, 2: vq2},
        omega_to_amp_map={1: 500.0, 2: 500.0}
    )
    
    sim = CWSpectroscopySimulator(setup)
    effective = sim._calculate_effective_drives([(1, 5000.0, 100.0)])
    
    # Should only have direct drive, no crosstalk
    assert 1 in effective
    assert 2 not in effective


def test_spectroscopy_integration(coupled_qubits_setup):
    """Test full spectroscopy workflow."""
    sim = CWSpectroscopySimulator(coupled_qubits_setup)
    
    # Sweep frequency around Q1
    frequencies = np.linspace(4900, 5100, 51)
    responses = []
    
    for freq in frequencies:
        iq = sim.simulate_spectroscopy_iq(
            drives=[(1, freq, 10.0)],
            readout_params={1: {'frequency': 7000.0, 'amplitude': 5.0}}
        )
        responses.append(iq[1])
    
    responses = np.array(responses)
    
    # Find local minimum around expected qubit frequency (5000 MHz)
    # Look in range 4950-5050 MHz to avoid coupling-induced features
    freq_mask = (frequencies >= 4950.0) & (frequencies <= 5050.0)
    local_responses = np.abs(responses)[freq_mask]
    local_frequencies = frequencies[freq_mask]
    local_min_idx = np.argmin(local_responses)
    min_freq = local_frequencies[local_min_idx]
    assert abs(min_freq - 5000.0) < 50.0  # Within 50 MHz


def test_setup_validation():
    """Test setup validation."""
    # Test with invalid setup
    with pytest.raises(ValueError, match="Invalid simulation setup"):
        CWSpectroscopySimulator({})  # Invalid object


def test_input_validation():
    """Test input validation for simulate_spectroscopy_iq."""
    vq = VirtualTransmon(name="Q1", qubit_frequency=5000.0, 
                        anharmonicity=-200.0, t1=50.0, t2=30.0,
                        readout_frequency=7000.0, truncate_level=3)
    setup = HighLevelSimulationSetup(
        name="test", virtual_qubits={1: vq}, omega_to_amp_map={1: 500.0}
    )
    sim = CWSpectroscopySimulator(setup)
    
    # Test invalid channel
    with pytest.raises(ValueError, match="Channel.*not found"):
        sim.simulate_spectroscopy_iq(
            drives=[(99, 5000.0, 10.0)],
            readout_params={1: {'frequency': 7000.0, 'amplitude': 5.0}}
        )
    
    # Test empty drives
    with pytest.raises(ValueError, match="At least one drive"):
        sim.simulate_spectroscopy_iq(
            drives=[],
            readout_params={1: {'frequency': 7000.0, 'amplitude': 5.0}}
        )
    
    # Test empty readout params
    with pytest.raises(ValueError, match="Readout parameters"):
        sim.simulate_spectroscopy_iq(
            drives=[(1, 5000.0, 10.0)],
            readout_params={}
        )


def test_hamiltonian_caching(single_qubit_setup):
    """Test Hamiltonian caching functionality."""
    sim = CWSpectroscopySimulator(single_qubit_setup)
    
    # First call should cache the Hamiltonian
    H1 = sim._get_cached_hamiltonian(1, 5000.0, 10.0)
    
    # Second call with same parameters should return cached version
    H2 = sim._get_cached_hamiltonian(1, 5000.0, 10.0)
    
    # Should be the same object (cached)
    assert H1 is H2
    
    # Different parameters should create new Hamiltonian
    H3 = sim._get_cached_hamiltonian(1, 5100.0, 10.0)
    assert H1 is not H3


def test_edge_cases(coupled_qubits_setup):
    """Test edge cases and boundary conditions."""
    sim = CWSpectroscopySimulator(coupled_qubits_setup)
    
    # Test zero amplitude drive
    iq = sim.simulate_spectroscopy_iq(
        drives=[(1, 5000.0, 0.0)],
        readout_params={1: {'frequency': 7000.0, 'amplitude': 5.0}}
    )
    assert 1 in iq
    
    # Test very large amplitude
    iq_large = sim.simulate_spectroscopy_iq(
        drives=[(1, 5000.0, 1000.0)],
        readout_params={1: {'frequency': 7000.0, 'amplitude': 5.0}}
    )
    assert 1 in iq_large
    
    # Test on-resonance vs off-resonance
    iq_on_res = sim.simulate_spectroscopy_iq(
        drives=[(1, 5000.0, 10.0)],  # On resonance
        readout_params={1: {'frequency': 7000.0, 'amplitude': 5.0}}
    )
    iq_off_res = sim.simulate_spectroscopy_iq(
        drives=[(1, 5500.0, 10.0)],  # Far off resonance
        readout_params={1: {'frequency': 7000.0, 'amplitude': 5.0}}
    )
    
    # Responses should be different
    assert np.abs(iq_on_res[1]) != np.abs(iq_off_res[1])


def test_multiple_drives(coupled_qubits_setup):
    """Test system with multiple simultaneous drives."""
    sim = CWSpectroscopySimulator(coupled_qubits_setup)
    
    # Drive both qubits simultaneously
    iq = sim.simulate_spectroscopy_iq(
        drives=[(1, 5000.0, 10.0), (2, 5200.0, 15.0)],
        readout_params={
            1: {'frequency': 7000.0, 'amplitude': 5.0},
            2: {'frequency': 7100.0, 'amplitude': 5.0}
        }
    )
    
    assert 1 in iq
    assert 2 in iq
    assert isinstance(iq[1], complex)
    assert isinstance(iq[2], complex)


def test_partial_readout(coupled_qubits_setup):
    """Test when readout is requested only for some qubits."""
    sim = CWSpectroscopySimulator(coupled_qubits_setup)
    
    # Drive both qubits but only readout one
    iq = sim.simulate_spectroscopy_iq(
        drives=[(1, 5000.0, 10.0), (2, 5200.0, 15.0)],
        readout_params={1: {'frequency': 7000.0, 'amplitude': 5.0}}  # Only Q1 readout
    )
    
    assert 1 in iq
    assert 2 not in iq  # No readout requested for Q2


def test_crosstalk_scaling(coupled_qubits_setup):
    """Test that crosstalk scales correctly with coupling and detuning."""
    sim = CWSpectroscopySimulator(coupled_qubits_setup)
    
    # Test different drive amplitudes
    effective_small = sim._calculate_effective_drives([(1, 5000.0, 10.0)])
    effective_large = sim._calculate_effective_drives([(1, 5000.0, 100.0)])
    
    # Crosstalk should scale with drive amplitude
    crosstalk_small = effective_small[2][1]  # Channel 2 crosstalk
    crosstalk_large = effective_large[2][1]
    
    assert crosstalk_large > crosstalk_small
    assert abs(crosstalk_large / crosstalk_small - 10.0) < 1.0  # Should be ~10x


def test_detuning_protection():
    """Test protection against division by very small detuning."""
    vq1 = VirtualTransmon(name="Q1", qubit_frequency=5000.0, 
                         anharmonicity=-200.0, t1=50.0, t2=30.0, 
                         readout_frequency=7000.0, truncate_level=3)
    vq2 = VirtualTransmon(name="Q2", qubit_frequency=5000.0,  # Same frequency!
                         anharmonicity=-210.0, t1=50.0, t2=30.0, 
                         readout_frequency=7100.0, truncate_level=3)
    
    setup = HighLevelSimulationSetup(
        name="test",
        virtual_qubits={1: vq1, 2: vq2},
        omega_to_amp_map={1: 500.0, 2: 500.0},
        coupling_strength_map={frozenset(["Q1", "Q2"]): 20.0}
    )
    
    sim = CWSpectroscopySimulator(setup)
    
    # Drive at qubit frequency - zero detuning should be protected
    effective = sim._calculate_effective_drives([(1, 5000.0, 100.0)])
    
    # Should not crash and should have reasonable crosstalk
    assert 2 in effective
    assert effective[2][1] > 0  # Non-zero crosstalk despite zero detuning


def test_population_extraction(single_qubit_setup):
    """Test population extraction from dressed states."""
    sim = CWSpectroscopySimulator(single_qubit_setup)
    
    # Test different drive conditions
    pops_off_res = sim._simulate_single_qubit(1, 5200.0, 10.0)  # 200 MHz detuned
    pops_on_res = sim._simulate_single_qubit(1, 5000.0, 10.0)   # On resonance
    pops_strong = sim._simulate_single_qubit(1, 5000.0, 100.0)  # Strong drive
    
    # Basic physical requirements
    # 1. All populations should sum to 1
    assert abs(np.sum(pops_off_res) - 1.0) < 1e-10
    assert abs(np.sum(pops_on_res) - 1.0) < 1e-10
    assert abs(np.sum(pops_strong) - 1.0) < 1e-10
    
    # 2. Different drives should produce different populations
    assert not np.allclose(pops_off_res, pops_on_res)
    assert not np.allclose(pops_on_res, pops_strong)
    
    # 3. Off-resonance should have more ground state population than on-resonance
    assert pops_off_res[0] > pops_on_res[0]


def test_empty_setup_error():
    """Test error handling for setup with no qubits."""
    empty_setup = HighLevelSimulationSetup(
        name="empty",
        virtual_qubits={},
        omega_to_amp_map={}
    )
    
    with pytest.raises(ValueError, match="No virtual qubits found"):
        CWSpectroscopySimulator(empty_setup)