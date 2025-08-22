import pytest
import numpy as np
from leeq.theory.simulation.numpy.dispersive_readout.multi_qubit_simulator import (
    MultiQubitDispersiveReadoutSimulator
)


def test_bell_state_simulation():
    """Example: Simulate Bell state |00⟩ + |11⟩ basis states"""
    # Create 2-qubit, 2-resonator system
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200],
        qubit_anharmonicities=[-250, -240],
        resonator_frequencies=[7000, 7500],
        resonator_kappas=[1.0, 1.2],
        coupling_matrix={
            ('Q0', 'R0'): 100,   # Q0 couples to R0
            ('Q1', 'R1'): 120,   # Q1 couples to R1  
            ('Q0', 'Q1'): 10     # Qubit-qubit coupling
        }
    )
    
    # Simulate both basis states of Bell state (|00⟩ + |11⟩)/√2
    trace_00_r0 = sim.simulate_trace((0, 0), 0, 7000, noise_std=0.05)
    trace_11_r0 = sim.simulate_trace((1, 1), 0, 7000, noise_std=0.05)
    
    # Basic validation
    assert isinstance(trace_00_r0, np.ndarray)
    assert isinstance(trace_11_r0, np.ndarray)
    assert trace_00_r0.dtype == np.complex128
    assert trace_11_r0.dtype == np.complex128
    
    # Traces should be different due to state-dependent chi shifts
    assert not np.allclose(trace_00_r0, trace_11_r0, rtol=1e-3)
    
    # Test resonator 1 as well
    trace_00_r1 = sim.simulate_trace((0, 0), 1, 7500, noise_std=0.05)
    trace_11_r1 = sim.simulate_trace((1, 1), 1, 7500, noise_std=0.05)
    
    # R1 traces should also be different for different states
    assert not np.allclose(trace_00_r1, trace_11_r1, rtol=1e-3)
    
    # R0 and R1 traces should be different (different resonators)
    assert not np.allclose(trace_00_r0, trace_00_r1, rtol=1e-3)


def test_three_qubit_ghz_state():
    """Example: Three-qubit GHZ state simulation"""
    # Create 3-qubit, 3-resonator system
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
    
    # Simulate GHZ basis states (|000⟩ + |111⟩)/√2
    traces_000 = sim.simulate_multiplexed_readout(
        joint_state=(0, 0, 0), 
        probe_frequencies=[7000, 7100, 7200],
        noise_std=0.05
    )
    traces_111 = sim.simulate_multiplexed_readout(
        joint_state=(1, 1, 1), 
        probe_frequencies=[7000, 7100, 7200],
        noise_std=0.05
    )
    
    # Should get 3 traces for 3 resonators
    assert len(traces_000) == 3
    assert len(traces_111) == 3
    
    # All traces should be complex arrays
    for trace in traces_000 + traces_111:
        assert isinstance(trace, np.ndarray)
        assert trace.dtype == np.complex128
    
    # Traces should be different between |000⟩ and |111⟩ states
    for i in range(3):
        assert not np.allclose(traces_000[i], traces_111[i], rtol=1e-3)
    
    # Each resonator should produce different traces
    assert not np.allclose(traces_000[0], traces_000[1], rtol=1e-3)
    assert not np.allclose(traces_000[1], traces_000[2], rtol=1e-3)


def test_asymmetric_coupling_example():
    """Example: Asymmetric coupling where qubits couple to multiple resonators"""
    # 2 qubits, 2 resonators with cross-coupling
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200],
        qubit_anharmonicities=[-250, -240],
        resonator_frequencies=[7000, 7500],
        resonator_kappas=[1.0, 1.2],
        coupling_matrix={
            ('Q0', 'R0'): 100,   # Strong coupling
            ('Q0', 'R1'): 30,    # Weak cross-coupling
            ('Q1', 'R0'): 20,    # Weak cross-coupling
            ('Q1', 'R1'): 120,   # Strong coupling
        }
    )
    
    # Test all 4 computational basis states
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    all_traces = {}
    
    for state in states:
        traces = sim.simulate_multiplexed_readout(
            joint_state=state,
            probe_frequencies=[7000, 7500],
            noise_std=0.02
        )
        all_traces[state] = traces
        
        assert len(traces) == 2
        for trace in traces:
            assert isinstance(trace, np.ndarray)
            assert trace.dtype == np.complex128
    
    # All states should produce different readout signatures
    # due to different chi shift combinations
    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states[i+1:], i+1):
            # At least one resonator should show difference
            r0_different = not np.allclose(
                all_traces[state1][0], all_traces[state2][0], rtol=1e-3
            )
            r1_different = not np.allclose(
                all_traces[state1][1], all_traces[state2][1], rtol=1e-3
            )
            assert r0_different or r1_different, f"States {state1} and {state2} produce identical traces"


def test_dispersive_shift_example():
    """Example: Demonstrate chi shift differences for different states"""
    # Simple 2-qubit system
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
    
    # Calculate chi shifts for all 4 computational basis states
    chi_shifts = {}
    for state_int in range(4):
        state_tuple = sim._get_state_tuple(state_int)
        chi = sim._calculate_chi_shifts(state_tuple)
        chi_shifts[state_tuple] = chi
    
    # Chi shifts should be different for different states
    assert not np.allclose(chi_shifts[(0, 0)], chi_shifts[(0, 1)])
    assert not np.allclose(chi_shifts[(0, 0)], chi_shifts[(1, 0)])
    assert not np.allclose(chi_shifts[(0, 0)], chi_shifts[(1, 1)])
    
    # R0 chi shift should depend on Q0 state (since Q0 couples to R0)
    assert not np.isclose(chi_shifts[(0, 0)][0], chi_shifts[(1, 0)][0])
    # But R0 chi shift should be same for different Q1 states (Q1 doesn't couple to R0)
    assert np.isclose(chi_shifts[(0, 0)][0], chi_shifts[(0, 1)][0], rtol=1e-10)
    
    # Similarly for R1 and Q1
    assert not np.isclose(chi_shifts[(0, 0)][1], chi_shifts[(0, 1)][1])
    # R1 chi shift should be same for different Q0 states
    assert np.isclose(chi_shifts[(0, 0)][1], chi_shifts[(1, 0)][1], rtol=1e-10)


def test_crosstalk_coupling_example():
    """Example: System with crosstalk between qubits and resonators"""
    # 2 qubits, 1 resonator - both qubits couple to same resonator
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200],
        qubit_anharmonicities=[-250, -240],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={
            ('Q0', 'R0'): 100,   # Q0 couples to R0
            ('Q1', 'R0'): 80,    # Q1 also couples to R0 (crosstalk)
        }
    )
    
    # Chi shifts should depend on both qubit states
    chi_00 = sim._calculate_chi_shifts((0, 0))
    chi_01 = sim._calculate_chi_shifts((0, 1))
    chi_10 = sim._calculate_chi_shifts((1, 0))
    chi_11 = sim._calculate_chi_shifts((1, 1))
    
    # All chi shifts should be different
    assert not np.isclose(chi_00[0], chi_01[0])
    assert not np.isclose(chi_00[0], chi_10[0])
    assert not np.isclose(chi_00[0], chi_11[0])
    assert not np.isclose(chi_01[0], chi_10[0])
    assert not np.isclose(chi_01[0], chi_11[0])
    assert not np.isclose(chi_10[0], chi_11[0])
    
    # Generate traces for all states
    traces = {}
    for state in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        trace = sim.simulate_trace(state, 0, 7000, noise_std=0.01)
        traces[state] = trace
        assert isinstance(trace, np.ndarray)
        assert trace.dtype == np.complex128
    
    # All traces should be distinguishable
    state_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, state1 in enumerate(state_list):
        for state2 in state_list[i+1:]:
            assert not np.allclose(traces[state1], traces[state2], rtol=1e-3)


def test_integer_vs_tuple_state_consistency():
    """Example: Verify integer and tuple state representations are equivalent"""
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
    
    # Test that integer and tuple representations give same results
    state_pairs = [
        (0, (0, 0)),  # |00⟩
        (1, (0, 1)),  # |01⟩
        (2, (1, 0)),  # |10⟩
        (3, (1, 1)),  # |11⟩
    ]
    
    for state_int, state_tuple in state_pairs:
        # Chi shifts should be identical
        chi_int = sim._calculate_chi_shifts(sim._get_state_tuple(state_int))
        chi_tuple = sim._calculate_chi_shifts(state_tuple)
        assert np.allclose(chi_int, chi_tuple)
        
        # Traces should be identical
        trace_int = sim.simulate_trace(state_int, 0, 7000, noise_std=0)
        trace_tuple = sim.simulate_trace(state_tuple, 0, 7000, noise_std=0)
        assert np.allclose(trace_int, trace_tuple)
        
        # Multiplexed readout should be identical
        traces_int = sim.simulate_multiplexed_readout(
            state_int, [7000, 7500], noise_std=0
        )
        traces_tuple = sim.simulate_multiplexed_readout(
            state_tuple, [7000, 7500], noise_std=0
        )
        for t_int, t_tuple in zip(traces_int, traces_tuple):
            assert np.allclose(t_int, t_tuple)