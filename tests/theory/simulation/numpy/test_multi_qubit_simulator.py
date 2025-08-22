import pytest
import numpy as np
from leeq.theory.simulation.numpy.dispersive_readout.multi_qubit_simulator import (
    MultiQubitDispersiveReadoutSimulator
)


def create_test_simulator():
    """Create a simple test simulator for testing."""
    return MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200],
        qubit_anharmonicities=[-250, -240],
        resonator_frequencies=[7000, 7500],
        resonator_kappas=[1.0, 1.2],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            ('Q1', 'R1'): 120,
        }
    )


def test_simulator_initialization():
    """Test basic instantiation with proper inheritance."""
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200],
        qubit_anharmonicities=[-250, -240],
        resonator_frequencies=[7000, 7500],
        resonator_kappas=[1.0, 1.2],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            ('Q1', 'R1'): 120,
        }
    )
    
    # Test multi-qubit specific properties
    assert sim.n_qubits == 2
    assert sim.n_resonators == 2
    assert len(sim.qubit_frequencies) == 2
    assert len(sim.qubit_anharmonicities) == 2
    assert len(sim.resonator_frequencies) == 2
    assert len(sim.resonator_kappas) == 2
    
    # Test arrays are properly converted
    assert isinstance(sim.qubit_frequencies, np.ndarray)
    assert isinstance(sim.qubit_anharmonicities, np.ndarray)
    assert isinstance(sim.resonator_frequencies, np.ndarray)
    assert isinstance(sim.resonator_kappas, np.ndarray)
    
    # Test coupling matrix
    assert ('Q0', 'R0') in sim.coupling_matrix
    assert ('Q1', 'R1') in sim.coupling_matrix
    assert sim.coupling_matrix[('Q0', 'R0')] == 100
    assert sim.coupling_matrix[('Q1', 'R1')] == 120
    
    # Test parent class properties are set
    assert hasattr(sim, 'f_r')
    assert hasattr(sim, 'kappa')
    assert hasattr(sim, 'amp')
    assert hasattr(sim, 'width')
    assert hasattr(sim, 'sampling_rate')


def test_state_conversion():
    """Test state tuple conversion for different cases."""
    sim = create_test_simulator()
    
    # Test 2-qubit cases
    assert sim._get_state_tuple(0) == (0, 0)
    assert sim._get_state_tuple(1) == (0, 1)
    assert sim._get_state_tuple(2) == (1, 0)
    assert sim._get_state_tuple(3) == (1, 1)
    
    # Test tuple passthrough
    assert sim._get_state_tuple((1, 0)) == (1, 0)
    assert sim._get_state_tuple((0, 1)) == (0, 1)
    
    # Test edge cases
    assert sim._get_state_tuple((0, 0)) == (0, 0)
    assert sim._get_state_tuple((1, 1)) == (1, 1)


def test_state_validation():
    """Test state validation for various inputs."""
    sim = create_test_simulator()
    
    # Test valid integer states
    assert sim._validate_state(0) == True
    assert sim._validate_state(1) == True
    assert sim._validate_state(2) == True
    assert sim._validate_state(3) == True
    
    # Test invalid integer states
    assert sim._validate_state(-1) == False
    assert sim._validate_state(4) == False  # 2^2 = 4 is out of bounds
    assert sim._validate_state(10) == False
    
    # Test valid tuple states
    assert sim._validate_state((0, 0)) == True
    assert sim._validate_state((0, 1)) == True
    assert sim._validate_state((1, 0)) == True
    assert sim._validate_state((1, 1)) == True
    
    # Test invalid tuple states
    assert sim._validate_state((0,)) == False  # Wrong length
    assert sim._validate_state((0, 0, 0)) == False  # Wrong length
    assert sim._validate_state((2, 0)) == False  # Invalid state values
    assert sim._validate_state((0, -1)) == False  # Invalid state values
    assert sim._validate_state((-1, 1)) == False  # Invalid state values
    
    # Test invalid types
    assert sim._validate_state([0, 1]) == False
    assert sim._validate_state("01") == False
    assert sim._validate_state(1.5) == False


def test_single_qubit_initialization():
    """Test that single qubit case works properly."""
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000],
        qubit_anharmonicities=[-250],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={('Q0', 'R0'): 100}
    )
    
    assert sim.n_qubits == 1
    assert sim.n_resonators == 1
    assert sim._get_state_tuple(0) == (0,)
    assert sim._get_state_tuple(1) == (1,)
    assert sim._validate_state(0) == True
    assert sim._validate_state(1) == True
    assert sim._validate_state(2) == False


def test_three_qubit_state_conversion():
    """Test state conversion for three qubits."""
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5100, 5200],
        qubit_anharmonicities=[-250, -245, -240],
        resonator_frequencies=[7000, 7100, 7200],
        resonator_kappas=[1.0, 1.0, 1.0],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            ('Q1', 'R1'): 100,
            ('Q2', 'R2'): 100,
        }
    )
    
    # Test 3-qubit binary conversion
    assert sim._get_state_tuple(0) == (0, 0, 0)  # |000⟩
    assert sim._get_state_tuple(1) == (0, 0, 1)  # |001⟩
    assert sim._get_state_tuple(2) == (0, 1, 0)  # |010⟩
    assert sim._get_state_tuple(3) == (0, 1, 1)  # |011⟩
    assert sim._get_state_tuple(4) == (1, 0, 0)  # |100⟩
    assert sim._get_state_tuple(5) == (1, 0, 1)  # |101⟩
    assert sim._get_state_tuple(6) == (1, 1, 0)  # |110⟩
    assert sim._get_state_tuple(7) == (1, 1, 1)  # |111⟩
    
    # Test validation
    assert sim._validate_state(7) == True  # Max valid state
    assert sim._validate_state(8) == False  # Out of bounds
    assert sim._validate_state((1, 1, 1)) == True
    assert sim._validate_state((1, 1)) == False  # Wrong length


def create_two_qubit_two_resonator_simulator():
    """Create a two-qubit, two-resonator simulator for testing multiplexed readout."""
    return MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200],
        qubit_anharmonicities=[-250, -240],
        resonator_frequencies=[7000, 7500],
        resonator_kappas=[1.0, 1.2],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            ('Q1', 'R1'): 120,
            ('Q0', 'Q1'): 10,  # Add some qubit-qubit coupling
        }
    )


def test_trace_generation():
    """Test I/Q trace generation for signal simulation."""
    sim = create_test_simulator()
    
    # Test trace generation for states that affect R0 (coupled to Q0)
    trace_00 = sim.simulate_trace(
        joint_state=(0, 0),
        resonator_id=0,
        f_probe=7000,
        noise_std=0
    )
    
    trace_10 = sim.simulate_trace(
        joint_state=(1, 0),  # Change Q0 state, which affects R0
        resonator_id=0,
        f_probe=7000,
        noise_std=0
    )
    
    # Basic validation
    assert isinstance(trace_00, np.ndarray)
    assert isinstance(trace_10, np.ndarray)
    assert trace_00.dtype == np.complex128
    assert trace_10.dtype == np.complex128
    assert len(trace_00) > 0
    assert len(trace_10) > 0
    
    # Traces should have same length
    assert len(trace_00) == len(trace_10)
    
    # Traces should be different for different states (due to chi shifts)
    assert not np.allclose(trace_00, trace_10)
    
    # Test with integer state input
    trace_int = sim.simulate_trace(
        joint_state=2,  # Should be equivalent to (1, 0)
        resonator_id=0,
        f_probe=7000,
        noise_std=0
    )
    
    # Should match tuple version
    assert np.allclose(trace_10, trace_int)
    
    # Test that R1 responds to Q1 state changes
    trace_01_r1 = sim.simulate_trace(
        joint_state=(0, 1),
        resonator_id=1,  # R1 is coupled to Q1
        f_probe=7500,
        noise_std=0
    )
    
    trace_00_r1 = sim.simulate_trace(
        joint_state=(0, 0),
        resonator_id=1,
        f_probe=7500,
        noise_std=0
    )
    
    # R1 should respond differently to Q1 state change
    assert not np.allclose(trace_01_r1, trace_00_r1)


def test_trace_generation_with_noise():
    """Test trace generation with noise addition."""
    sim = create_test_simulator()
    
    # Generate traces with and without noise
    trace_no_noise = sim.simulate_trace((0, 1), 0, 7000, noise_std=0)
    trace_with_noise = sim.simulate_trace((0, 1), 0, 7000, noise_std=0.1)
    
    # Should have same length
    assert len(trace_no_noise) == len(trace_with_noise)
    
    # Should be different due to noise
    assert not np.allclose(trace_no_noise, trace_with_noise)
    
    # Noise should be complex
    assert trace_with_noise.dtype == np.complex128


def test_trace_generation_parameters():
    """Test trace generation with different simulation parameters."""
    sim = create_test_simulator()
    
    # Test with custom parameters
    trace_custom = sim.simulate_trace(
        joint_state=(1, 0),
        resonator_id=1,
        f_probe=7500,
        noise_std=0,
        amp=2.0,
        width=20.0,
        sampling_rate=2000
    )
    
    # Test with default parameters
    trace_default = sim.simulate_trace(
        joint_state=(1, 0),
        resonator_id=1,
        f_probe=7500,
        noise_std=0
    )
    
    # Custom trace should have different length due to different width/sampling_rate
    assert len(trace_custom) != len(trace_default)
    
    # Both should be valid complex arrays
    assert isinstance(trace_custom, np.ndarray)
    assert isinstance(trace_default, np.ndarray)
    assert trace_custom.dtype == np.complex128
    assert trace_default.dtype == np.complex128
    
    # Both should have non-zero signal
    assert np.any(np.abs(trace_custom) > 0)
    assert np.any(np.abs(trace_default) > 0)


def test_trace_generation_validation():
    """Test validation errors for trace generation."""
    sim = create_test_simulator()
    
    # Test invalid state
    with pytest.raises(ValueError, match="Invalid joint state"):
        sim.simulate_trace((2, 0), 0, 7000)  # Invalid state value
    
    with pytest.raises(ValueError, match="Invalid joint state"):
        sim.simulate_trace(4, 0, 7000)  # Out of bounds integer state
    
    # Test invalid resonator ID
    with pytest.raises(ValueError, match="Resonator ID .* out of range"):
        sim.simulate_trace((0, 1), 5, 7000)  # Out of bounds resonator
    
    with pytest.raises(ValueError, match="Resonator ID .* out of range"):
        sim.simulate_trace((0, 1), -1, 7000)  # Negative resonator ID


def test_multiplexed_readout():
    """Test simultaneous multi-resonator readout."""
    sim = create_two_qubit_two_resonator_simulator()
    
    # Test multiplexed readout
    traces = sim.simulate_multiplexed_readout(
        joint_state=(1, 0),
        probe_frequencies=[7000, 7500],
        noise_std=0
    )
    
    # Should return list of traces
    assert isinstance(traces, list)
    assert len(traces) == 2
    
    # Each trace should be a complex array
    for trace in traces:
        assert isinstance(trace, np.ndarray)
        assert trace.dtype == np.complex128
        assert len(trace) > 0
    
    # Traces should be different for different resonators
    assert not np.allclose(traces[0], traces[1])
    
    # Test different states produce different results
    traces_01 = sim.simulate_multiplexed_readout(
        joint_state=(0, 1),
        probe_frequencies=[7000, 7500],
        noise_std=0
    )
    
    assert len(traces_01) == 2
    assert not np.allclose(traces[0], traces_01[0])  # Different states should give different results


def test_multiplexed_readout_with_noise():
    """Test multiplexed readout with noise."""
    sim = create_two_qubit_two_resonator_simulator()
    
    # Test with noise
    traces_with_noise = sim.simulate_multiplexed_readout(
        joint_state=(1, 1),
        probe_frequencies=[7000, 7500],
        noise_std=0.2
    )
    
    traces_no_noise = sim.simulate_multiplexed_readout(
        joint_state=(1, 1),
        probe_frequencies=[7000, 7500],
        noise_std=0
    )
    
    assert len(traces_with_noise) == 2
    assert len(traces_no_noise) == 2
    
    # Should be different due to noise
    assert not np.allclose(traces_with_noise[0], traces_no_noise[0])
    assert not np.allclose(traces_with_noise[1], traces_no_noise[1])


def test_multiplexed_readout_validation():
    """Test validation for multiplexed readout."""
    sim = create_two_qubit_two_resonator_simulator()
    
    # Test wrong number of probe frequencies
    with pytest.raises(ValueError, match="Expected 2 probe frequencies, got 1"):
        sim.simulate_multiplexed_readout(
            joint_state=(0, 1),
            probe_frequencies=[7000],  # Should be 2 frequencies
            noise_std=0
        )
    
    with pytest.raises(ValueError, match="Expected 2 probe frequencies, got 3"):
        sim.simulate_multiplexed_readout(
            joint_state=(0, 1),
            probe_frequencies=[7000, 7500, 8000],  # Too many frequencies
            noise_std=0
        )


def test_signal_properties():
    """Test basic properties of generated signals."""
    sim = create_test_simulator()
    
    # Generate trace
    trace = sim.simulate_trace(
        joint_state=(0, 1),
        resonator_id=0,
        f_probe=7000,
        noise_std=0
    )
    
    # Signal should be complex
    assert trace.dtype == np.complex128
    
    # Signal should have both real and imaginary parts (I/Q)
    assert np.any(np.real(trace) != 0)
    assert np.any(np.imag(trace) != 0)
    
    # Signal amplitude should be finite
    assert np.all(np.isfinite(trace))
    assert np.all(np.isfinite(np.abs(trace)))
    
    # Test that signal has some structure (not all zeros)
    assert np.any(np.abs(trace) > 0)


def test_different_resonator_responses():
    """Test that different resonators give different responses."""
    sim = create_two_qubit_two_resonator_simulator()
    
    # Same state, same probe frequency, different resonators
    trace_r0 = sim.simulate_trace((1, 1), 0, 7000, noise_std=0)
    trace_r1 = sim.simulate_trace((1, 1), 1, 7000, noise_std=0)
    
    # Should be different due to different resonator parameters and chi shifts
    assert not np.allclose(trace_r0, trace_r1)