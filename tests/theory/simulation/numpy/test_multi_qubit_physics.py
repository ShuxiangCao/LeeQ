import pytest
import numpy as np
import warnings
from leeq.theory.simulation.numpy.dispersive_readout.multi_qubit_simulator import (
    MultiQubitDispersiveReadoutSimulator
)


def test_chi_shift_single_qubit():
    """Test that chi shifts reduce to single-qubit case for N=1."""
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000],
        qubit_anharmonicities=[-250],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={('Q0', 'R0'): 100}
    )
    
    # Calculate chi shifts for ground and excited states
    chi_0 = sim._calculate_chi_shifts((0,))
    chi_1 = sim._calculate_chi_shifts((1,))
    
    # Verify physics: chi shifts should be different for different states
    # For ground state: ω_q = 5000, Δ = 7000 - 5000 = 2000
    # For excited state: ω_q = 5000 + (-250) = 4750, Δ = 7000 - 4750 = 2250
    # So chi_0 = g²/2000, chi_1 = g²/2250
    expected_chi_0 = 100**2 / 2000  # g²/Δ_ground
    expected_chi_1 = 100**2 / 2250   # g²/Δ_excited
    
    # Should be close to expected values (within 1%)
    assert abs(chi_0[0] - expected_chi_0) < 0.01 * abs(expected_chi_0)
    assert abs(chi_1[0] - expected_chi_1) < 0.01 * abs(expected_chi_1)
    assert chi_0[0] != 0  # Ground state should have non-zero chi
    assert chi_1[0] != chi_0[0]  # Excited state should be different


def test_chi_shift_two_qubits():
    """Test chi shifts for two-qubit states."""
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200],
        qubit_anharmonicities=[-250, -240],
        resonator_frequencies=[7000, 7500],
        resonator_kappas=[1.0, 1.2],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            ('Q1', 'R1'): 120,
            ('Q0', 'Q1'): 10  # Qubit-qubit coupling
        }
    )
    
    # Calculate chi shifts for all basis states
    chi_00 = sim._calculate_chi_shifts((0, 0))
    chi_01 = sim._calculate_chi_shifts((0, 1))
    chi_10 = sim._calculate_chi_shifts((1, 0))
    chi_11 = sim._calculate_chi_shifts((1, 1))
    
    # Chi shifts should be different for different states
    assert not np.allclose(chi_00, chi_01)
    assert not np.allclose(chi_00, chi_10)
    assert not np.allclose(chi_00, chi_11)
    assert not np.allclose(chi_01, chi_10)
    
    # Each resonator should have non-zero chi shifts
    assert chi_00[0] != 0  # R0 coupled to Q0
    assert chi_00[1] != 0  # R1 coupled to Q1
    
    # Verify qubit-qubit coupling affects chi shifts
    # When Q1 is excited, it should affect Q0's effective frequency
    # and thus R0's chi shift should be different
    assert chi_01[0] != chi_00[0]  # R0 chi changes when Q1 excited


def test_dispersive_regime_warning():
    """Test warning when not in dispersive regime (g/Δ > 0.1)."""
    with pytest.warns(UserWarning, match="Dispersive regime approximation"):
        sim = MultiQubitDispersiveReadoutSimulator(
            qubit_frequencies=[5000],
            qubit_anharmonicities=[-250],
            resonator_frequencies=[5100],  # Very close to qubit frequency
            resonator_kappas=[1.0],
            coupling_matrix={('Q0', 'R0'): 200}  # Strong coupling
        )
        # This should trigger warning: g/Δ = 200/100 = 2.0 > 0.1
        chi = sim._calculate_chi_shifts((0,))


def test_near_zero_detuning_warning():
    """Test warning for near-zero detuning that could cause instability."""
    with pytest.warns(UserWarning, match="Near-zero detuning"):
        sim = MultiQubitDispersiveReadoutSimulator(
            qubit_frequencies=[6999.9999995],  # Very close to resonator
            qubit_anharmonicities=[-250],
            resonator_frequencies=[7000],
            resonator_kappas=[1.0],
            coupling_matrix={('Q0', 'R0'): 100}
        )
        # This should trigger near-zero detuning warning (|Δ| < 1e-6)
        chi = sim._calculate_chi_shifts((0,))


def test_chi_shift_scaling():
    """Test that chi shifts scale correctly with coupling strength."""
    base_coupling = 50
    strong_coupling = 100  # 2x stronger
    
    # Create two simulators with different coupling strengths
    sim_weak = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000],
        qubit_anharmonicities=[-250],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={('Q0', 'R0'): base_coupling}
    )
    
    sim_strong = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000],
        qubit_anharmonicities=[-250],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={('Q0', 'R0'): strong_coupling}
    )
    
    chi_weak = sim_weak._calculate_chi_shifts((1,))
    chi_strong = sim_strong._calculate_chi_shifts((1,))
    
    # Chi should scale as g² (quadratic in coupling)
    expected_ratio = (strong_coupling / base_coupling)**2  # Should be 4
    actual_ratio = chi_strong[0] / chi_weak[0]
    
    assert abs(actual_ratio - expected_ratio) < 0.01


def test_chi_shift_state_dependence():
    """Test that chi shifts depend on the specific qubit state configuration."""
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200, 5400],
        qubit_anharmonicities=[-250, -240, -230],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={
            ('Q0', 'R0'): 80,
            ('Q1', 'R0'): 90,
            ('Q2', 'R0'): 100,
        }
    )
    
    # All qubits couple to same resonator
    chi_000 = sim._calculate_chi_shifts((0, 0, 0))
    chi_001 = sim._calculate_chi_shifts((0, 0, 1))
    chi_010 = sim._calculate_chi_shifts((0, 1, 0))
    chi_100 = sim._calculate_chi_shifts((1, 0, 0))
    chi_111 = sim._calculate_chi_shifts((1, 1, 1))
    
    # All states should give different chi shifts
    states = [chi_000[0], chi_001[0], chi_010[0], chi_100[0], chi_111[0]]
    
    # No two states should be identical
    for i in range(len(states)):
        for j in range(i+1, len(states)):
            assert abs(states[i] - states[j]) > 1e-10  # Allow for small numerical errors


def test_no_coupling_zero_chi():
    """Test that uncoupled qubits don't contribute to chi shifts."""
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200],
        qubit_anharmonicities=[-250, -240],
        resonator_frequencies=[7000, 7500],
        resonator_kappas=[1.0, 1.2],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            # Q1 is not coupled to any resonator
            # R1 is not coupled to any qubit
        }
    )
    
    chi_00 = sim._calculate_chi_shifts((0, 0))
    chi_01 = sim._calculate_chi_shifts((0, 1))
    chi_10 = sim._calculate_chi_shifts((1, 0))
    chi_11 = sim._calculate_chi_shifts((1, 1))
    
    # R0 should have different chi for different Q0 states
    assert chi_00[0] != chi_10[0]  # Q0 state matters for R0
    assert chi_01[0] != chi_11[0]  # Q0 state matters for R0
    
    # Q1 state should not affect R0 chi (no coupling)
    assert abs(chi_00[0] - chi_01[0]) < 1e-12
    assert abs(chi_10[0] - chi_11[0]) < 1e-12
    
    # R1 should have zero chi (no coupling to any qubit)
    assert abs(chi_00[1]) < 1e-12
    assert abs(chi_01[1]) < 1e-12
    assert abs(chi_10[1]) < 1e-12
    assert abs(chi_11[1]) < 1e-12


def test_qubit_qubit_coupling_effect():
    """Test that qubit-qubit coupling affects chi shifts."""
    # Create simulator with qubit-qubit coupling
    sim_with_qq = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200],
        qubit_anharmonicities=[-250, -240],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            ('Q1', 'R0'): 90,
            ('Q0', 'Q1'): 20  # Qubit-qubit coupling
        }
    )
    
    # Create simulator without qubit-qubit coupling
    sim_no_qq = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200],
        qubit_anharmonicities=[-250, -240],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={
            ('Q0', 'R0'): 100,
            ('Q1', 'R0'): 90,
            # No qubit-qubit coupling
        }
    )
    
    # Compare chi shifts for |11⟩ state
    chi_with_qq = sim_with_qq._calculate_chi_shifts((1, 1))
    chi_no_qq = sim_no_qq._calculate_chi_shifts((1, 1))
    
    # Chi shifts should be different when qubit-qubit coupling is present
    assert not np.allclose(chi_with_qq, chi_no_qq)
    assert abs(chi_with_qq[0] - chi_no_qq[0]) > 1e-6  # Should be measurable difference