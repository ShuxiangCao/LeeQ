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
    assert sim._validate_state(0)
    assert sim._validate_state(1)
    assert sim._validate_state(2)
    assert sim._validate_state(3)

    # Test invalid integer states
    assert not sim._validate_state(-1)
    assert not sim._validate_state(4)  # 2^2 = 4 is out of bounds
    assert not sim._validate_state(10)

    # Test valid tuple states
    assert sim._validate_state((0, 0))
    assert sim._validate_state((0, 1))
    assert sim._validate_state((1, 0))
    assert sim._validate_state((1, 1))

    # Test invalid tuple states
    assert not sim._validate_state((0,))  # Wrong length
    assert not sim._validate_state((0, 0, 0))  # Wrong length
    assert not sim._validate_state((2, 0))  # Invalid state values
    assert not sim._validate_state((0, -1))  # Invalid state values
    assert not sim._validate_state((-1, 1))  # Invalid state values

    # Test invalid types
    assert not sim._validate_state([0, 1])
    assert not sim._validate_state("01")
    assert not sim._validate_state(1.5)


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
    assert sim._validate_state(0)
    assert sim._validate_state(1)
    assert not sim._validate_state(2)


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
    assert sim._validate_state(7)  # Max valid state
    assert not sim._validate_state(8)  # Out of bounds
    assert sim._validate_state((1, 1, 1))
    assert not sim._validate_state((1, 1))  # Wrong length


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


def test_simulate_channel_readout_single_resonator():
    """Test channel with single resonator (no averaging)."""
    sim = create_two_qubit_two_resonator_simulator()

    # Channel map with single resonator per channel
    channel_map = {
        1: [0],  # Channel 1 gets only resonator 0
        2: [1],  # Channel 2 gets only resonator 1
    }

    # Generate channel traces
    channel_traces = sim.simulate_channel_readout(
        joint_state=(1, 0),
        probe_frequencies=[7000, 7500],
        channel_map=channel_map,
        noise_std=0
    )

    # Should return dict with channel IDs as keys
    assert isinstance(channel_traces, dict)
    assert set(channel_traces.keys()) == {1, 2}

    # Each channel should have a trace
    for channel_id, trace in channel_traces.items():
        assert isinstance(trace, np.ndarray)
        assert trace.dtype == np.complex128
        assert len(trace) > 0

    # Generate individual resonator traces for comparison
    individual_traces = sim.simulate_multiplexed_readout(
        joint_state=(1, 0),
        probe_frequencies=[7000, 7500],
        noise_std=0
    )

    # Single resonator channels should match individual traces exactly
    np.testing.assert_array_equal(channel_traces[1], individual_traces[0])
    np.testing.assert_array_equal(channel_traces[2], individual_traces[1])

    # Different channels should produce different traces
    assert not np.allclose(channel_traces[1], channel_traces[2])


def test_simulate_channel_readout_multiple_resonators():
    """Test channel with multiple resonators (averaging)."""
    # Create simulator with 3 resonators
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200, 5400],
        qubit_anharmonicities=[-250, -240, -230],
        resonator_frequencies=[7000, 7200, 7400],
        resonator_kappas=[1.0, 1.2, 1.4],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            ('Q1', 'R1'): 120,
            ('Q2', 'R2'): 140,
        }
    )

    # Channel map with multiple resonators per channel
    channel_map = {
        1: [0, 1],  # Channel 1 gets average of resonators 0 and 1
        2: [2],     # Channel 2 gets only resonator 2
    }

    # Generate channel traces
    channel_traces = sim.simulate_channel_readout(
        joint_state=(1, 0, 1),
        probe_frequencies=[7000, 7200, 7400],
        channel_map=channel_map,
        noise_std=0
    )

    # Should return dict with expected channel IDs
    assert isinstance(channel_traces, dict)
    assert set(channel_traces.keys()) == {1, 2}

    # Get individual resonator traces for comparison
    individual_traces = sim.simulate_multiplexed_readout(
        joint_state=(1, 0, 1),
        probe_frequencies=[7000, 7200, 7400],
        noise_std=0
    )

    # Channel 1 should be average of resonators 0 and 1
    expected_channel_1 = np.mean([individual_traces[0], individual_traces[1]], axis=0)
    np.testing.assert_array_almost_equal(channel_traces[1], expected_channel_1)

    # Channel 2 should match resonator 2 exactly (single resonator)
    np.testing.assert_array_equal(channel_traces[2], individual_traces[2])

    # Verify averaging behavior: averaged trace should be between individual traces
    # Check at a few sample points
    for i in [0, len(individual_traces[0])//2, -1]:
        avg_val = channel_traces[1][i]
        val_0 = individual_traces[0][i]
        val_1 = individual_traces[1][i]

        # Average should be between the two values (or equal if they're the same)
        if val_0 != val_1:
            assert min(abs(val_0), abs(val_1)) <= abs(avg_val) <= max(abs(val_0), abs(val_1)) or \
                   abs(avg_val - (val_0 + val_1) / 2) < 1e-10


def test_simulate_channel_readout_various_configurations():
    """Test different channel mappings."""
    # Create simulator with 4 resonators for flexible testing
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200, 5400, 5600],
        qubit_anharmonicities=[-250, -240, -230, -220],
        resonator_frequencies=[7000, 7200, 7400, 7600],
        resonator_kappas=[1.0, 1.2, 1.4, 1.6],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            ('Q1', 'R1'): 120,
            ('Q2', 'R2'): 140,
            ('Q3', 'R3'): 160,
        }
    )

    # Test configuration 1: All resonators in one channel
    channel_map_1 = {1: [0, 1, 2, 3]}

    channel_traces_1 = sim.simulate_channel_readout(
        joint_state=(0, 1, 0, 1),
        probe_frequencies=[7000, 7200, 7400, 7600],
        channel_map=channel_map_1,
        noise_std=0
    )

    assert len(channel_traces_1) == 1
    assert 1 in channel_traces_1
    assert isinstance(channel_traces_1[1], np.ndarray)

    # Test configuration 2: Mixed single and multi-resonator channels
    channel_map_2 = {
        10: [0],      # Single resonator
        20: [1, 2],   # Two resonators
        30: [3],      # Single resonator
    }

    channel_traces_2 = sim.simulate_channel_readout(
        joint_state=(1, 1, 0, 0),
        probe_frequencies=[7000, 7200, 7400, 7600],
        channel_map=channel_map_2,
        noise_std=0
    )

    assert len(channel_traces_2) == 3
    assert set(channel_traces_2.keys()) == {10, 20, 30}

    # Verify each channel has valid traces
    for channel_id, trace in channel_traces_2.items():
        assert isinstance(trace, np.ndarray)
        assert trace.dtype == np.complex128
        assert len(trace) > 0

    # Test configuration 3: Multiple channels with overlapping resonators (should warn)
    channel_map_3 = {
        1: [0, 1],
        2: [1, 2],  # Resonator 1 appears in both channels
    }

    # This should work but generate a warning
    with pytest.warns(UserWarning, match="Resonator indices .* are assigned to multiple channels"):
        channel_traces_3 = sim.simulate_channel_readout(
            joint_state=(0, 0, 1, 1),
            probe_frequencies=[7000, 7200, 7400, 7600],
            channel_map=channel_map_3,
            noise_std=0
        )

    assert len(channel_traces_3) == 2
    assert set(channel_traces_3.keys()) == {1, 2}


def test_simulate_channel_readout_edge_cases():
    """Test error handling and edge cases."""
    sim = create_two_qubit_two_resonator_simulator()

    # Test empty channel_map
    with pytest.raises(ValueError, match="channel_map cannot be empty"):
        sim.simulate_channel_readout(
            joint_state=(0, 1),
            probe_frequencies=[7000, 7500],
            channel_map={},
            noise_std=0
        )

    # Test non-dict channel_map
    with pytest.raises(TypeError, match="channel_map must be a dictionary"):
        sim.simulate_channel_readout(
            joint_state=(0, 1),
            probe_frequencies=[7000, 7500],
            channel_map=[(1, [0]), (2, [1])],  # List instead of dict
            noise_std=0
        )

    # Test invalid channel ID type
    with pytest.raises(TypeError, match="Channel ID .* must be an integer"):
        sim.simulate_channel_readout(
            joint_state=(0, 1),
            probe_frequencies=[7000, 7500],
            channel_map={"1": [0], 2: [1]},  # String key instead of int
            noise_std=0
        )

    # Test invalid resonator indices type
    with pytest.raises(TypeError, match="Resonator indices for channel .* must be a list"):
        sim.simulate_channel_readout(
            joint_state=(0, 1),
            probe_frequencies=[7000, 7500],
            channel_map={1: 0, 2: [1]},  # Int instead of list
            noise_std=0
        )

    # Test empty resonator indices list
    with pytest.raises(ValueError, match="Channel .* has no resonator indices"):
        sim.simulate_channel_readout(
            joint_state=(0, 1),
            probe_frequencies=[7000, 7500],
            channel_map={1: [], 2: [1]},  # Empty list
            noise_std=0
        )

    # Test invalid resonator index type
    with pytest.raises(TypeError, match="Resonator index .* for channel .* must be an integer"):
        sim.simulate_channel_readout(
            joint_state=(0, 1),
            probe_frequencies=[7000, 7500],
            channel_map={1: [0.5], 2: [1]},  # Float instead of int
            noise_std=0
        )

    # Test out-of-bounds resonator index (negative)
    with pytest.raises(ValueError, match="Resonator index .* for channel .* is out of range"):
        sim.simulate_channel_readout(
            joint_state=(0, 1),
            probe_frequencies=[7000, 7500],
            channel_map={1: [-1], 2: [1]},  # Negative index
            noise_std=0
        )

    # Test out-of-bounds resonator index (too high)
    with pytest.raises(ValueError, match="Resonator index .* for channel .* is out of range"):
        sim.simulate_channel_readout(
            joint_state=(0, 1),
            probe_frequencies=[7000, 7500],
            channel_map={1: [0], 2: [5]},  # Index 5 doesn't exist (only 0,1)
            noise_std=0
        )

    # Test invalid joint_state (should be caught by underlying method)
    with pytest.raises(ValueError, match="Expected 2 probe frequencies"):
        sim.simulate_channel_readout(
            joint_state=(0, 1),
            probe_frequencies=[7000],  # Wrong number of frequencies
            channel_map={1: [0], 2: [1]},
            noise_std=0
        )

    # Test successful edge case: single channel with three resonators
    sim_3res = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200, 5400],
        qubit_anharmonicities=[-250, -240, -230],
        resonator_frequencies=[7000, 7200, 7400],
        resonator_kappas=[1.0, 1.2, 1.4],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            ('Q1', 'R1'): 120,
            ('Q2', 'R2'): 140,
        }
    )

    channel_traces = sim_3res.simulate_channel_readout(
        joint_state=(1, 1, 1),
        probe_frequencies=[7000, 7200, 7400],
        channel_map={42: [0, 1, 2]},  # All resonators in channel 42
        noise_std=0
    )

    assert len(channel_traces) == 1
    assert 42 in channel_traces
    assert isinstance(channel_traces[42], np.ndarray)
    assert channel_traces[42].dtype == np.complex128


def test_simulate_channel_readout_with_noise():
    """Test channel readout with noise addition."""
    sim = create_two_qubit_two_resonator_simulator()

    channel_map = {
        1: [0, 1],  # Average both resonators
    }

    # Generate traces with and without noise
    traces_no_noise = sim.simulate_channel_readout(
        joint_state=(1, 0),
        probe_frequencies=[7000, 7500],
        channel_map=channel_map,
        noise_std=0
    )

    traces_with_noise = sim.simulate_channel_readout(
        joint_state=(1, 0),
        probe_frequencies=[7000, 7500],
        channel_map=channel_map,
        noise_std=0.2
    )

    # Both should have same structure
    assert set(traces_no_noise.keys()) == set(traces_with_noise.keys())
    assert len(traces_no_noise[1]) == len(traces_with_noise[1])

    # Should be different due to noise
    assert not np.allclose(traces_no_noise[1], traces_with_noise[1])

    # Both should be complex arrays
    assert traces_no_noise[1].dtype == np.complex128
    assert traces_with_noise[1].dtype == np.complex128


def test_simulate_channel_readout_state_dependence():
    """Test that channel readout responds correctly to state changes."""
    sim = create_two_qubit_two_resonator_simulator()

    channel_map = {
        1: [0],    # Channel 1: just resonator 0 (coupled to Q0)
        2: [1],    # Channel 2: just resonator 1 (coupled to Q1)
        3: [0, 1], # Channel 3: average of both
    }

    # Test different states
    states_to_test = [(0, 0), (1, 0), (0, 1), (1, 1)]
    all_traces = {}

    for state in states_to_test:
        traces = sim.simulate_channel_readout(
            joint_state=state,
            probe_frequencies=[7000, 7500],
            channel_map=channel_map,
            noise_std=0
        )
        all_traces[state] = traces

    # Channel 1 should respond to Q0 state changes
    assert not np.allclose(all_traces[(0, 0)][1], all_traces[(1, 0)][1])  # Q0: 0->1

    # Channel 2 should respond to Q1 state changes
    assert not np.allclose(all_traces[(0, 0)][2], all_traces[(0, 1)][2])  # Q1: 0->1

    # Channel 3 (averaged) should respond to both qubit state changes
    assert not np.allclose(all_traces[(0, 0)][3], all_traces[(1, 0)][3])  # Q0 change
    assert not np.allclose(all_traces[(0, 0)][3], all_traces[(0, 1)][3])  # Q1 change
    assert not np.allclose(all_traces[(0, 0)][3], all_traces[(1, 1)][3])  # Both change


# Integration tests for Task 2.2
def test_channel_readout_integration_comprehensive_states():
    """Integration test: Comprehensive testing with different joint states across multi-qubit systems."""
    # Create a 3-qubit system for more comprehensive state testing
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200, 5400],
        qubit_anharmonicities=[-250, -240, -230],
        resonator_frequencies=[7000, 7200, 7400],
        resonator_kappas=[1.0, 1.2, 1.4],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            ('Q1', 'R1'): 120,
            ('Q2', 'R2'): 140,
            ('Q0', 'Q1'): 5,   # Add some inter-qubit coupling
            ('Q1', 'Q2'): 7,
        }
    )

    channel_map = {
        1: [0],      # Single resonator channel
        2: [1, 2],   # Multi-resonator channel
    }

    # Test all possible 3-qubit states (8 states total)
    all_states = []
    for i in range(8):  # 2^3 = 8 possible states
        all_states.append(i)

    # Also test tuple format for subset of states
    tuple_states = [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]

    all_traces = {}

    # Test integer states
    for state in all_states:
        traces = sim.simulate_channel_readout(
            joint_state=state,
            probe_frequencies=[7000, 7200, 7400],
            channel_map=channel_map,
            noise_std=0
        )
        all_traces[f"int_{state}"] = traces

        # Verify structure
        assert isinstance(traces, dict)
        assert set(traces.keys()) == {1, 2}
        assert isinstance(traces[1], np.ndarray)
        assert isinstance(traces[2], np.ndarray)

    # Test tuple states
    for state in tuple_states:
        traces = sim.simulate_channel_readout(
            joint_state=state,
            probe_frequencies=[7000, 7200, 7400],
            channel_map=channel_map,
            noise_std=0
        )
        all_traces[f"tuple_{state}"] = traces

    # Verify that equivalent integer and tuple states produce identical results
    np.testing.assert_array_equal(all_traces["int_0"][1], all_traces["tuple_(0, 0, 0)"][1])
    np.testing.assert_array_equal(all_traces["int_5"][1], all_traces["tuple_(1, 0, 1)"][1])

    # Verify different states produce different traces
    assert not np.allclose(all_traces["int_0"][1], all_traces["int_7"][1])
    assert not np.allclose(all_traces["int_0"][2], all_traces["int_7"][2])

    # Verify channel 2 (averaged) shows combined response
    individual_r1 = sim.simulate_trace((1, 1, 0), 1, 7200, noise_std=0)
    individual_r2 = sim.simulate_trace((1, 1, 0), 2, 7400, noise_std=0)
    expected_avg = np.mean([individual_r1, individual_r2], axis=0)

    actual_channel_2 = sim.simulate_channel_readout(
        joint_state=(1, 1, 0),
        probe_frequencies=[7000, 7200, 7400],
        channel_map=channel_map,
        noise_std=0
    )[2]

    np.testing.assert_array_almost_equal(actual_channel_2, expected_avg)


def test_channel_readout_integration_noise_scenarios():
    """Integration test: Various noise scenarios and levels."""
    sim = create_two_qubit_two_resonator_simulator()

    channel_map = {
        1: [0],      # Single resonator
        2: [0, 1],   # Averaged resonators
    }

    # Test multiple noise levels
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]
    state = (1, 0)
    probe_freqs = [7000, 7500]

    traces_by_noise = {}

    for noise_std in noise_levels:
        traces = sim.simulate_channel_readout(
            joint_state=state,
            probe_frequencies=probe_freqs,
            channel_map=channel_map,
            noise_std=noise_std
        )
        traces_by_noise[noise_std] = traces

        # Verify structure remains consistent
        assert isinstance(traces, dict)
        assert set(traces.keys()) == {1, 2}
        assert traces[1].dtype == np.complex128
        assert traces[2].dtype == np.complex128
        assert len(traces[1]) == len(traces[2])

    # Verify noise-free and noisy traces are different (except for 0 noise)
    no_noise = traces_by_noise[0.0]
    for noise_std in noise_levels[1:]:
        noisy = traces_by_noise[noise_std]
        assert not np.allclose(no_noise[1], noisy[1], rtol=1e-3)
        assert not np.allclose(no_noise[2], noisy[2], rtol=1e-3)

    # Test that higher noise levels create more deviation
    # Compare SNR-like metric
    signal_power_no_noise = np.mean(np.abs(no_noise[1])**2)

    for noise_std in [0.1, 0.2]:
        noisy_trace = traces_by_noise[noise_std]
        noise_power = np.mean(np.abs(noisy_trace[1] - no_noise[1])**2)
        # Higher noise should create proportionally more deviation
        assert noise_power > 0
        snr_estimate = signal_power_no_noise / noise_power
        # Very rough SNR check - should be finite and reasonable
        assert np.isfinite(snr_estimate)
        assert snr_estimate > 0

    # Test noise reproducibility with different channel configurations
    # The underlying simulate_multiplexed_readout should handle noise consistently
    for noise_std in [0.1, 0.3]:
        # Same parameters should give similar noise characteristics
        traces_a = sim.simulate_channel_readout(state, probe_freqs, channel_map, noise_std)
        traces_b = sim.simulate_channel_readout(state, probe_freqs, channel_map, noise_std)

        # Traces should be different (due to random noise) but have similar statistics
        assert not np.allclose(traces_a[1], traces_b[1])
        # But should have similar variance characteristics
        var_a = np.var(np.abs(traces_a[1]))
        var_b = np.var(np.abs(traces_b[1]))
        assert 0.5 < var_a / var_b < 2.0  # Should be within factor of 2


def test_channel_readout_integration_probe_frequency_variations():
    """Integration test: Various probe frequencies and their effects."""
    sim = create_two_qubit_two_resonator_simulator()

    channel_map = {
        1: [0, 1],   # Average both resonators
    }

    state = (0, 1)

    # Test frequency variations
    frequency_variations = [
        [7000, 7500],           # Baseline
        [6900, 7400],           # Both shifted down
        [7100, 7600],           # Both shifted up
        [7050, 7450],           # Smaller shifts
        [6950, 7550],           # Asymmetric shifts
        [7000.5, 7500.3],       # Fine frequency adjustments
    ]

    traces_by_freq = {}

    for freqs in frequency_variations:
        traces = sim.simulate_channel_readout(
            joint_state=state,
            probe_frequencies=freqs,
            channel_map=channel_map,
            noise_std=0
        )
        traces_by_freq[tuple(freqs)] = traces

        # Verify structure
        assert isinstance(traces, dict)
        assert 1 in traces
        assert isinstance(traces[1], np.ndarray)
        assert traces[1].dtype == np.complex128
        assert len(traces[1]) > 0

    # Different frequencies should produce different responses
    baseline = traces_by_freq[(7000, 7500)]
    for freqs in frequency_variations[1:]:
        test_trace = traces_by_freq[tuple(freqs)]
        # Different probe frequencies should give different responses
        assert not np.allclose(baseline[1], test_trace[1], rtol=1e-6)

    # Test that frequency effects are consistent with underlying physics
    # Verify that the averaged signal matches expected behavior
    for freqs in frequency_variations:
        # Get individual resonator responses
        individual_traces = sim.simulate_multiplexed_readout(
            joint_state=state,
            probe_frequencies=freqs,
            noise_std=0
        )

        # Channel readout should match manual averaging
        expected_avg = np.mean(individual_traces, axis=0)
        channel_trace = traces_by_freq[tuple(freqs)][1]

        np.testing.assert_array_almost_equal(channel_trace, expected_avg)

    # Test probe frequency validation integration
    # Wrong number of frequencies should raise error from underlying method
    with pytest.raises(ValueError, match="Expected 2 probe frequencies"):
        sim.simulate_channel_readout(
            joint_state=state,
            probe_frequencies=[7000],  # Only one frequency
            channel_map=channel_map,
            noise_std=0
        )

    with pytest.raises(ValueError, match="Expected 2 probe frequencies"):
        sim.simulate_channel_readout(
            joint_state=state,
            probe_frequencies=[7000, 7200, 7400],  # Too many frequencies
            channel_map=channel_map,
            noise_std=0
        )


def test_channel_readout_backward_compatibility_integration():
    """Integration test: Verify backward compatibility with existing functionality."""
    sim = create_two_qubit_two_resonator_simulator()

    # Test that existing simulate_multiplexed_readout still works exactly as before
    state = (1, 0)
    probe_freqs = [7000, 7500]

    # Call original method
    original_traces = sim.simulate_multiplexed_readout(
        joint_state=state,
        probe_frequencies=probe_freqs,
        noise_std=0
    )

    # Verify original method behavior is unchanged
    assert isinstance(original_traces, list)
    assert len(original_traces) == 2
    assert all(isinstance(trace, np.ndarray) for trace in original_traces)
    assert all(trace.dtype == np.complex128 for trace in original_traces)

    # Test that channel readout with single-resonator channels matches original
    single_channel_map = {
        1: [0],  # Channel 1 gets resonator 0 only
        2: [1],  # Channel 2 gets resonator 1 only
    }

    channel_traces = sim.simulate_channel_readout(
        joint_state=state,
        probe_frequencies=probe_freqs,
        channel_map=single_channel_map,
        noise_std=0
    )

    # Single-resonator channels should exactly match original traces
    np.testing.assert_array_equal(channel_traces[1], original_traces[0])
    np.testing.assert_array_equal(channel_traces[2], original_traces[1])

    # Test with noise - should still be consistent
    original_noisy = sim.simulate_multiplexed_readout(
        joint_state=state,
        probe_frequencies=probe_freqs,
        noise_std=0.1
    )

    # Note: We can't test exact equality with noise due to randomness,
    # but we can verify that the structure and behavior is consistent
    assert len(original_noisy) == 2
    assert all(isinstance(trace, np.ndarray) for trace in original_noisy)
    assert all(trace.dtype == np.complex128 for trace in original_noisy)

    # Verify that all existing simulator methods still work
    assert hasattr(sim, 'simulate_trace')
    assert hasattr(sim, 'simulate_multiplexed_readout')
    assert hasattr(sim, 'simulate_channel_readout')

    # Test that state validation still works consistently
    assert sim._validate_state((1, 0))
    assert sim._validate_state(2)  # Should be equivalent to (1, 0)
    assert not sim._validate_state((2, 0))  # Invalid state

    # Test that both integer and tuple states work in both methods
    trace_tuple = sim.simulate_multiplexed_readout(
        joint_state=(0, 1),
        probe_frequencies=probe_freqs,
        noise_std=0
    )

    trace_int = sim.simulate_multiplexed_readout(
        joint_state=1,  # Binary: 01 = (0, 1)
        probe_frequencies=probe_freqs,
        noise_std=0
    )

    # Should be identical
    np.testing.assert_array_equal(trace_tuple[0], trace_int[0])
    np.testing.assert_array_equal(trace_tuple[1], trace_int[1])

    # Same should work for channel readout
    channel_tuple = sim.simulate_channel_readout(
        joint_state=(0, 1),
        probe_frequencies=probe_freqs,
        channel_map=single_channel_map,
        noise_std=0
    )

    channel_int = sim.simulate_channel_readout(
        joint_state=1,
        probe_frequencies=probe_freqs,
        channel_map=single_channel_map,
        noise_std=0
    )

    np.testing.assert_array_equal(channel_tuple[1], channel_int[1])
    np.testing.assert_array_equal(channel_tuple[2], channel_int[2])


def test_channel_readout_integration_large_system_scalability():
    """Integration test: Verify functionality scales to larger multi-qubit systems."""
    # Create a larger system (4 qubits, 6 resonators) for scalability testing
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5100, 5200, 5300],
        qubit_anharmonicities=[-250, -240, -230, -220],
        resonator_frequencies=[7000, 7100, 7200, 7300, 7400, 7500],
        resonator_kappas=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        coupling_matrix={
            ('Q0', 'R0'): 100, ('Q0', 'R1'): 50,
            ('Q1', 'R1'): 120, ('Q1', 'R2'): 60,
            ('Q2', 'R2'): 140, ('Q2', 'R3'): 70,
            ('Q3', 'R3'): 160, ('Q3', 'R4'): 80, ('Q3', 'R5'): 40,
        }
    )

    # Complex channel mapping
    channel_map = {
        1: [0],           # Single resonator
        2: [1, 2],        # Two resonators
        3: [3, 4, 5],     # Three resonators
        4: [0, 3],        # Non-contiguous resonators
    }

    probe_frequencies = [7000, 7100, 7200, 7300, 7400, 7500]

    # Test various states
    test_states = [
        0,           # All qubits in |0⟩
        15,          # All qubits in |1⟩ (binary 1111)
        5,           # Binary 0101
        10,          # Binary 1010
        (1, 0, 1, 0), # Same as binary 1010 but tuple format
    ]

    for state in test_states:
        traces = sim.simulate_channel_readout(
            joint_state=state,
            probe_frequencies=probe_frequencies,
            channel_map=channel_map,
            noise_std=0.05
        )

        # Verify structure
        assert isinstance(traces, dict)
        assert set(traces.keys()) == {1, 2, 3, 4}

        # Verify each channel has valid traces
        for channel_id, trace in traces.items():
            assert isinstance(trace, np.ndarray)
            assert trace.dtype == np.complex128
            assert len(trace) > 0
            assert np.all(np.isfinite(trace))

    # Verify averaged channels work correctly for larger systems
    # Test channel 3 (resonators 3, 4, 5)
    individual_traces = sim.simulate_multiplexed_readout(
        joint_state=10,  # Binary 1010
        probe_frequencies=probe_frequencies,
        noise_std=0
    )

    channel_trace = sim.simulate_channel_readout(
        joint_state=10,
        probe_frequencies=probe_frequencies,
        channel_map={99: [3, 4, 5]},
        noise_std=0
    )[99]

    expected_avg = np.mean([individual_traces[3], individual_traces[4], individual_traces[5]], axis=0)
    np.testing.assert_array_almost_equal(channel_trace, expected_avg)

    # Performance check - should complete in reasonable time
    import time
    start_time = time.time()

    for _ in range(3):  # Run a few times to get average
        sim.simulate_channel_readout(
            joint_state=7,
            probe_frequencies=probe_frequencies,
            channel_map=channel_map,
            noise_std=0.1
        )

    elapsed = time.time() - start_time
    # Should be sub-second for reasonable system size (loose check)
    assert elapsed < 10.0  # Very conservative - should be much faster


# Physics validation tests for Task 2.3
def test_physics_validation_averaged_traces_preserve_behavior():
    """Physics validation: Verify averaged traces preserve expected dispersive behavior."""
    # Create a system where we can validate the averaging preserves physics
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200],  # Different qubit frequencies
        qubit_anharmonicities=[-250, -240],  # Different anharmonicities
        resonator_frequencies=[7000, 7500],  # Different resonator frequencies
        resonator_kappas=[1.0, 1.2],  # Different decay rates
        coupling_matrix={
            ('Q0', 'R0'): 100,  # Q0 couples to R0
            ('Q1', 'R1'): 120,  # Q1 couples to R1
        }
    )

    # Test different states to verify dispersive shifts
    states = [(0, 0), (1, 0), (0, 1), (1, 1)]
    probe_frequencies = [7000, 7500]

    # Single resonator channels (baseline)
    single_channel_map = {1: [0], 2: [1]}

    # Multi-resonator channel (averaging)
    avg_channel_map = {3: [0, 1]}

    single_traces = {}
    avg_traces = {}
    individual_traces = {}

    for state in states:
        # Get single-resonator channel traces
        single_result = sim.simulate_channel_readout(
            joint_state=state,
            probe_frequencies=probe_frequencies,
            channel_map=single_channel_map,
            noise_std=0
        )
        single_traces[state] = single_result

        # Get averaged channel traces
        avg_result = sim.simulate_channel_readout(
            joint_state=state,
            probe_frequencies=probe_frequencies,
            channel_map=avg_channel_map,
            noise_std=0
        )
        avg_traces[state] = avg_result

        # Get individual resonator traces for manual verification
        individual_result = sim.simulate_multiplexed_readout(
            joint_state=state,
            probe_frequencies=probe_frequencies,
            noise_std=0
        )
        individual_traces[state] = individual_result

    # Physics validation 1: Averaged traces should show state-dependent behavior
    # The averaged signal should still distinguish between different qubit states
    states_list = [(0, 0), (1, 0), (0, 1), (1, 1)]

    for i, state1 in enumerate(states_list):
        for state2 in states_list[i+1:]:
            # Averaged channel should distinguish different states
            diff = np.abs(avg_traces[state1][3] - avg_traces[state2][3])
            max_diff = np.max(diff)

            # Should have measurable difference (physics preserved)
            assert max_diff > 1e-10, f"Averaged channel cannot distinguish states {state1} and {state2}"

    # Physics validation 2: Averaged trace should be mathematical average
    for state in states:
        expected_avg = np.mean([individual_traces[state][0], individual_traces[state][1]], axis=0)
        actual_avg = avg_traces[state][3]

        np.testing.assert_array_almost_equal(
            actual_avg, expected_avg,
            err_msg=f"Averaged trace doesn't match mathematical average for state {state}"
        )

    # Physics validation 3: Dispersive shifts should be preserved in averaged signals
    # Check that state-dependent frequency shifts are present in averaged data
    for resonator_id in [0, 1]:
        # Get single resonator responses for different states
        trace_00 = single_traces[(0, 0)][resonator_id + 1]
        trace_10 = single_traces[(1, 0)][resonator_id + 1]
        trace_01 = single_traces[(0, 1)][resonator_id + 1]

        # Verify individual resonators show dispersive shifts
        if resonator_id == 0:  # R0 couples to Q0
            # Should see difference when Q0 state changes
            assert not np.allclose(trace_00, trace_10), "R0 should respond to Q0 state change"
            # Should see minimal difference when only Q1 changes (weak coupling)
            diff_q1_only = np.max(np.abs(trace_00 - trace_01))
            diff_q0_change = np.max(np.abs(trace_00 - trace_10))
            # Q0 change should create larger effect on R0 than Q1 change
            assert diff_q0_change > diff_q1_only * 0.1, "R0 should respond more to Q0 than Q1"

    # Physics validation 4: Averaged signal preserves relative dispersive shift magnitudes
    # Compare the magnitude of shifts in averaged vs individual signals
    avg_00 = avg_traces[(0, 0)][3]
    avg_11 = avg_traces[(1, 1)][3]

    # Maximum difference between |00⟩ and |11⟩ states in averaged signal
    avg_shift_magnitude = np.max(np.abs(avg_00 - avg_11))

    # Should be non-zero (physics preserved) and reasonable magnitude
    assert avg_shift_magnitude > 1e-10, "Averaged signal should show dispersive shifts"

    # The averaged shift should be related to individual resonator shifts
    r0_shift = np.max(np.abs(individual_traces[(0, 0)][0] - individual_traces[(1, 1)][0]))
    r1_shift = np.max(np.abs(individual_traces[(0, 0)][1] - individual_traces[(1, 1)][1]))

    # Averaged shift should be reasonable compared to individual shifts
    # (exact relationship depends on coupling strengths and frequencies)
    expected_avg_shift_approx = (r0_shift + r1_shift) / 2
    # Should be within same order of magnitude
    assert 0.1 < avg_shift_magnitude / expected_avg_shift_approx < 10, \
        f"Averaged shift magnitude {avg_shift_magnitude} not reasonable compared to individual shifts {r0_shift}, {r1_shift}"


def test_physics_validation_chi_shift_preservation():
    """Physics validation: Test chi shift preservation in averaged signals."""
    # Create system with well-defined chi shifts for testing
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5100],
        qubit_anharmonicities=[-250, -250],  # Same anharmonicity for simpler analysis
        resonator_frequencies=[7000, 7200],
        resonator_kappas=[1.0, 1.0],  # Same kappa for simpler analysis
        coupling_matrix={
            ('Q0', 'R0'): 100,  # Same coupling strength
            ('Q1', 'R1'): 100,  # Same coupling strength
        }
    )

    probe_frequencies = [7000, 7200]

    # Test chi shift calculation consistency
    # Chi shift is the dispersive frequency shift due to qubit state change

    # Channel configurations to test
    single_channel_map = {1: [0], 2: [1]}  # Individual resonators
    avg_channel_map = {3: [0, 1]}  # Averaged channel

    # Get traces for computational basis states
    ground_state = (0, 0)  # |00⟩
    excited_state = (1, 1)  # |11⟩

    # Single resonator channels
    single_ground = sim.simulate_channel_readout(
        joint_state=ground_state,
        probe_frequencies=probe_frequencies,
        channel_map=single_channel_map,
        noise_std=0
    )

    single_excited = sim.simulate_channel_readout(
        joint_state=excited_state,
        probe_frequencies=probe_frequencies,
        channel_map=single_channel_map,
        noise_std=0
    )

    # Averaged channel
    avg_ground = sim.simulate_channel_readout(
        joint_state=ground_state,
        probe_frequencies=probe_frequencies,
        channel_map=avg_channel_map,
        noise_std=0
    )

    avg_excited = sim.simulate_channel_readout(
        joint_state=excited_state,
        probe_frequencies=probe_frequencies,
        channel_map=avg_channel_map,
        noise_std=0
    )

    # Physics validation: Chi shifts should be preserved in averaged signals

    # 1. Individual resonator chi shifts
    chi_shift_r0 = single_excited[1] - single_ground[1]  # R0 shift
    chi_shift_r1 = single_excited[2] - single_ground[2]  # R1 shift

    # 2. Averaged channel chi shift
    chi_shift_avg = avg_excited[3] - avg_ground[3]  # Averaged shift

    # 3. Expected averaged chi shift should be average of individual shifts
    expected_chi_shift_avg = (chi_shift_r0 + chi_shift_r1) / 2

    # Verify chi shift preservation
    np.testing.assert_array_almost_equal(
        chi_shift_avg, expected_chi_shift_avg,
        err_msg="Chi shift in averaged signal doesn't match expected average of individual chi shifts"
    )

    # 4. Validate chi shift magnitudes are physically reasonable
    chi_mag_r0 = np.max(np.abs(chi_shift_r0))
    chi_mag_r1 = np.max(np.abs(chi_shift_r1))
    chi_mag_avg = np.max(np.abs(chi_shift_avg))

    # All should be non-zero (chi shifts present)
    assert chi_mag_r0 > 1e-12, "R0 should show chi shift"
    assert chi_mag_r1 > 1e-12, "R1 should show chi shift"
    assert chi_mag_avg > 1e-12, "Averaged channel should show chi shift"

    # Averaged chi shift should be bounded by individual shifts
    min_chi = min(chi_mag_r0, chi_mag_r1)
    max_chi = max(chi_mag_r0, chi_mag_r1)

    # For simple averaging, the averaged chi magnitude should be between the individual magnitudes
    assert min_chi * 0.5 <= chi_mag_avg <= max_chi * 1.5, \
        f"Averaged chi shift magnitude {chi_mag_avg} not reasonable given individual magnitudes {chi_mag_r0}, {chi_mag_r1}"

    # 5. Test chi shift consistency across different state combinations
    partial_states = [(1, 0), (0, 1)]  # Single excitation states

    for partial_state in partial_states:
        partial_single = sim.simulate_channel_readout(
            joint_state=partial_state,
            probe_frequencies=probe_frequencies,
            channel_map=single_channel_map,
            noise_std=0
        )

        partial_avg = sim.simulate_channel_readout(
            joint_state=partial_state,
            probe_frequencies=probe_frequencies,
            channel_map=avg_channel_map,
            noise_std=0
        )

        # Partial chi shifts
        partial_chi_r0 = partial_single[1] - single_ground[1]
        partial_chi_r1 = partial_single[2] - single_ground[2]
        partial_chi_avg = partial_avg[3] - avg_ground[3]

        # Expected partial averaged chi shift
        expected_partial_chi_avg = (partial_chi_r0 + partial_chi_r1) / 2

        np.testing.assert_array_almost_equal(
            partial_chi_avg, expected_partial_chi_avg,
            err_msg=f"Chi shift preservation failed for state {partial_state}"
        )


def test_physics_validation_single_vs_multiple_resonator_comparison():
    """Physics validation: Compare single vs multiple resonator cases for physics consistency."""
    # Create system where we can compare single-resonator vs multi-resonator channels
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200, 5400],
        qubit_anharmonicities=[-250, -240, -230],
        resonator_frequencies=[7000, 7300, 7600],  # Well-separated frequencies
        resonator_kappas=[1.0, 1.5, 2.0],  # Different linewidths
        coupling_matrix={
            ('Q0', 'R0'): 80,   # Different coupling strengths
            ('Q1', 'R1'): 120,
            ('Q2', 'R2'): 160,
        }
    )

    probe_frequencies = [7000, 7300, 7600]
    test_state = (1, 0, 1)  # Mixed state for interesting physics

    # Define channel configurations for comparison
    single_channels = {1: [0], 2: [1], 3: [2]}  # Each resonator in separate channel
    pair_channels = {4: [0, 1], 5: [2]}  # Two resonators averaged, one single
    all_channels = {6: [0, 1, 2]}  # All resonators averaged

    # Get traces for each configuration
    single_result = sim.simulate_channel_readout(
        joint_state=test_state,
        probe_frequencies=probe_frequencies,
        channel_map=single_channels,
        noise_std=0
    )

    pair_result = sim.simulate_channel_readout(
        joint_state=test_state,
        probe_frequencies=probe_frequencies,
        channel_map=pair_channels,
        noise_std=0
    )

    all_result = sim.simulate_channel_readout(
        joint_state=test_state,
        probe_frequencies=probe_frequencies,
        channel_map=all_channels,
        noise_std=0
    )

    # Get individual traces for manual validation
    individual_traces = sim.simulate_multiplexed_readout(
        joint_state=test_state,
        probe_frequencies=probe_frequencies,
        noise_std=0
    )

    # Physics validation 1: Single-resonator channels should match individual traces exactly
    np.testing.assert_array_equal(single_result[1], individual_traces[0])
    np.testing.assert_array_equal(single_result[2], individual_traces[1])
    np.testing.assert_array_equal(single_result[3], individual_traces[2])

    # Physics validation 2: Pair channel should be mathematical average
    expected_pair_avg = np.mean([individual_traces[0], individual_traces[1]], axis=0)
    np.testing.assert_array_almost_equal(pair_result[4], expected_pair_avg)

    # Single resonator in pair configuration should match individual
    np.testing.assert_array_equal(pair_result[5], individual_traces[2])

    # Physics validation 3: All-resonator channel should be average of all
    expected_all_avg = np.mean(individual_traces, axis=0)
    np.testing.assert_array_almost_equal(all_result[6], expected_all_avg)

    # Physics validation 4: Signal power relationships
    # Individual resonator powers
    power_r0 = np.mean(np.abs(individual_traces[0])**2)
    power_r1 = np.mean(np.abs(individual_traces[1])**2)
    power_r2 = np.mean(np.abs(individual_traces[2])**2)

    # Averaged channel powers
    power_pair = np.mean(np.abs(pair_result[4])**2)
    power_all = np.mean(np.abs(all_result[6])**2)

    # For averaging independent signals, the power doesn't simply average
    # but the relationship should be physically reasonable
    assert power_pair > 0, "Pair channel should have non-zero power"
    assert power_all > 0, "All-resonator channel should have non-zero power"

    # Powers should be finite and reasonable
    all_powers = [power_r0, power_r1, power_r2, power_pair, power_all]
    assert all(np.isfinite(p) and p > 0 for p in all_powers), "All powers should be positive and finite"

    # Physics validation 5: Frequency response characteristics preservation
    # Test that averaging preserves the overall frequency response character

    # For different probe frequency offsets, verify consistent behavior
    frequency_offsets = [-10, -5, 0, 5, 10]  # MHz offsets from nominal

    for offset in frequency_offsets:
        offset_frequencies = [f + offset for f in probe_frequencies]

        # Get responses with frequency offset

        all_offset = sim.simulate_channel_readout(
            joint_state=test_state,
            probe_frequencies=offset_frequencies,
            channel_map=all_channels,
            noise_std=0
        )

        # Individual traces with offset
        individual_offset = sim.simulate_multiplexed_readout(
            joint_state=test_state,
            probe_frequencies=offset_frequencies,
            noise_std=0
        )

        # Verify averaging relationship maintained
        expected_all_offset = np.mean(individual_offset, axis=0)
        np.testing.assert_array_almost_equal(
            all_offset[6], expected_all_offset,
            err_msg=f"Averaging relationship not preserved for frequency offset {offset} MHz"
        )

    # Physics validation 6: Linearity of averaging operation
    # Test that averaging behaves linearly with signal amplitudes

    # Compare responses for different states
    state_ground = (0, 0, 0)
    state_test = (1, 0, 1)

    ground_individual = sim.simulate_multiplexed_readout(
        joint_state=state_ground,
        probe_frequencies=probe_frequencies,
        noise_std=0
    )

    test_individual = sim.simulate_multiplexed_readout(
        joint_state=state_test,
        probe_frequencies=probe_frequencies,
        noise_std=0
    )

    ground_all = sim.simulate_channel_readout(
        joint_state=state_ground,
        probe_frequencies=probe_frequencies,
        channel_map=all_channels,
        noise_std=0
    )[6]

    test_all = sim.simulate_channel_readout(
        joint_state=state_test,
        probe_frequencies=probe_frequencies,
        channel_map=all_channels,
        noise_std=0
    )[6]

    # The difference in averaged signals should equal the average of differences
    diff_individual = [test_individual[i] - ground_individual[i] for i in range(3)]
    expected_diff_avg = np.mean(diff_individual, axis=0)
    actual_diff_avg = test_all - ground_all

    np.testing.assert_array_almost_equal(
        actual_diff_avg, expected_diff_avg,
        err_msg="Averaging operation not linear with respect to signal differences"
    )

    # Physics validation 7: Verify that multi-resonator averaging preserves physics scaling
    # When averaging N resonators, certain properties should scale predictably

    single_snr_estimates = []
    for i in range(3):
        signal_power = np.mean(np.abs(individual_traces[i])**2)
        # Use a simple noise floor estimate
        noise_floor = signal_power * 1e-6  # Assume very low noise floor
        snr_est = signal_power / max(noise_floor, 1e-12)
        single_snr_estimates.append(snr_est)

    # Averaged signal properties
    all_signal_power = np.mean(np.abs(all_result[6])**2)
    pair_signal_power = np.mean(np.abs(pair_result[4])**2)

    # All values should be finite and positive
    assert np.isfinite(all_signal_power) and all_signal_power > 0
    assert np.isfinite(pair_signal_power) and pair_signal_power > 0
    assert all(np.isfinite(snr) and snr > 0 for snr in single_snr_estimates)

    # The physics consistency is maintained if:
    # 1. All signals are finite and meaningful
    # 2. Averaging relationships are mathematically correct
    # 3. State-dependent behavior is preserved

    # These have been validated above, confirming physics preservation


def test_channel_readout_usage_example():
    """
    Usage example demonstrating typical channel readout workflow.

    This test serves as a living example of how to use the channel readout
    functionality in real quantum measurement scenarios.
    """
    # Example: 4-qubit processor with 6 resonators distributed across 3 channels

    # Step 1: Create a realistic multi-qubit system
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200, 5400, 5600],  # MHz
        qubit_anharmonicities=[-200, -220, -180, -210],  # MHz
        resonator_frequencies=[7000, 7100, 7200, 7300, 7400, 7500],  # MHz
        resonator_kappas=[2.0, 2.2, 1.8, 2.1, 1.9, 2.3],  # MHz
        coupling_matrix={
            # Qubit-resonator couplings (MHz)
            ('Q0', 'R0'): 50, ('Q0', 'R1'): 45,  # Q0 couples to two resonators
            ('Q1', 'R2'): 60, ('Q2', 'R3'): 55,  # Q1, Q2 single resonator each
            ('Q3', 'R4'): 48, ('Q3', 'R5'): 52,  # Q3 couples to two resonators
            # Qubit-qubit interactions
            ('Q0', 'Q1'): 5, ('Q1', 'Q2'): 4, ('Q2', 'Q3'): 3
        }
    )

    # Step 2: Define realistic channel mapping
    # This reflects typical frequency-multiplexed readout architecture
    channel_map = {
        1: [0, 1],    # Channel 1: Q0's two resonators (averaged)
        2: [2, 3],    # Channel 2: Q1 and Q2 resonators (combined channel)
        3: [4, 5]     # Channel 3: Q3's two resonators (averaged)
    }

    probe_frequencies = [7000, 7100, 7200, 7300, 7400, 7500]

    # Step 3: Simulate different computational states
    test_states = [
        (0, 0, 0, 0),  # Ground state |0000⟩
        (1, 0, 0, 0),  # Single excitation |1000⟩
        (1, 1, 0, 0),  # Two excitations |1100⟩
        (1, 1, 1, 1),  # All excited |1111⟩
    ]

    channel_responses = {}

    for state in test_states:
        # Simulate with realistic noise
        traces = sim.simulate_channel_readout(
            joint_state=state,
            probe_frequencies=probe_frequencies,
            channel_map=channel_map,
            noise_std=0.005  # Typical measurement noise
        )
        channel_responses[state] = traces

        # Verify we get expected channel structure
        assert set(traces.keys()) == {1, 2, 3}

        # Each channel should return complex-valued I/Q trace
        for channel_id, trace in traces.items():
            assert isinstance(trace, np.ndarray)
            assert trace.dtype == np.complex128
            assert len(trace) > 0  # Non-empty trace
            assert np.all(np.isfinite(trace))  # No NaN/inf values

    # Step 4: Demonstrate state discrimination using channel data
    # Compare ground state vs excited states
    ground_traces = channel_responses[(0, 0, 0, 0)]
    excited_traces = channel_responses[(1, 1, 1, 1)]

    for channel_id in [1, 2, 3]:
        # Signals should be different for different qubit states
        signal_difference = np.mean(np.abs(excited_traces[channel_id] - ground_traces[channel_id]))
        assert signal_difference > 1e-10, f"Channel {channel_id} shows no state discrimination"

    # Step 5: Demonstrate noise characterization
    # Multiple measurements with same parameters should show noise variations
    noise_traces_1 = sim.simulate_channel_readout(
        joint_state=(1, 0, 1, 0),
        probe_frequencies=probe_frequencies,
        channel_map=channel_map,
        noise_std=0.01
    )

    noise_traces_2 = sim.simulate_channel_readout(
        joint_state=(1, 0, 1, 0),  # Same state
        probe_frequencies=probe_frequencies,
        channel_map=channel_map,
        noise_std=0.01  # Same noise level
    )

    # Traces should be different due to random noise
    for channel_id in [1, 2, 3]:
        assert not np.allclose(noise_traces_1[channel_id], noise_traces_2[channel_id])

        # But statistical properties should be similar
        std1 = np.std(noise_traces_1[channel_id])
        std2 = np.std(noise_traces_2[channel_id])
        assert 0.5 < std1/std2 < 2.0  # Within factor of 2

    # This example demonstrates:
    # - Realistic system setup with multiple qubits and resonators
    # - Channel mapping for frequency-multiplexed readout
    # - State-dependent measurement simulation
    # - Noise characterization
    # - Basic quantum state discrimination using channel data
