import pytest
import numpy as np
import warnings
from unittest.mock import Mock
from leeq.theory.simulation.numpy.dispersive_readout.multi_qubit_simulator import (
    MultiQubitDispersiveReadoutSimulator
)
from leeq.experiments.builtin.basic.calibrations.resonator_spectroscopy import ResonatorSweepTransmissionWithExtraInitialLPB


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


def test_chi_shift_formula_validation():
    """
    Test chi shift formula validation: χ = g²/Δ
    
    This test validates that the fundamental dispersive readout formula
    is correctly implemented across different system configurations.
    """
    # Test various coupling strengths and detunings
    test_cases = [
        {'g': 50, 'qubit_freq': 5000, 'resonator_freq': 7000},   # Standard case
        {'g': 100, 'qubit_freq': 4800, 'resonator_freq': 6800},  # Different frequencies
        {'g': 80, 'qubit_freq': 5200, 'resonator_freq': 7500},   # Large detuning
        {'g': 120, 'qubit_freq': 5100, 'resonator_freq': 6100},  # Small detuning
    ]
    
    for case in test_cases:
        g = case['g']
        qubit_freq = case['qubit_freq']
        resonator_freq = case['resonator_freq']
        delta = resonator_freq - qubit_freq
        
        # Create simulator
        sim = MultiQubitDispersiveReadoutSimulator(
            qubit_frequencies=[qubit_freq],
            qubit_anharmonicities=[-250],
            resonator_frequencies=[resonator_freq],
            resonator_kappas=[1.0],
            coupling_matrix={('Q0', 'R0'): g}
        )
        
        # Calculate chi shift for ground state
        chi_calculated = sim._calculate_chi_shifts((0,))[0]
        
        # Expected chi from formula: χ = g²/Δ
        chi_expected = g**2 / delta
        
        # Validate formula within 1% tolerance
        relative_error = abs(chi_calculated - chi_expected) / abs(chi_expected)
        assert relative_error < 0.01, f"Chi formula validation failed: g={g}, Δ={delta}, χ_calc={chi_calculated}, χ_exp={chi_expected}, rel_error={relative_error}"
        
        print(f"✅ Chi formula validated: g={g}, Δ={delta}, χ={chi_calculated:.3f}")


def test_multi_qubit_scaling_1_to_4():
    """
    Test multi-qubit system scaling from 1 to 4 qubits.
    
    Validates that the physics remains consistent as system size increases
    and that coupling effects are properly included.
    """
    base_freq = 5000
    freq_spacing = 200  # MHz
    coupling_strength = 100  # MHz
    readout_base = 7000
    
    # Test systems with 1, 2, 3, and 4 qubits
    for n_qubits in range(1, 5):
        # Create system parameters
        qubit_frequencies = [base_freq + i * freq_spacing for i in range(n_qubits)]
        resonator_frequencies = [readout_base + i * 500 for i in range(n_qubits)]  # Well separated
        anharmonicities = [-250] * n_qubits
        kappas = [1.0] * n_qubits
        
        # Build coupling matrix
        coupling_matrix = {}
        
        # Qubit-resonator couplings
        for i in range(n_qubits):
            coupling_matrix[(f"Q{i}", f"R{i}")] = coupling_strength
            
        # Nearest-neighbor qubit-qubit couplings for multi-qubit systems
        if n_qubits > 1:
            for i in range(n_qubits - 1):
                coupling_matrix[(f"Q{i}", f"Q{i+1}")] = 10.0  # 10 MHz coupling
        
        # Create simulator
        sim = MultiQubitDispersiveReadoutSimulator(
            qubit_frequencies=qubit_frequencies,
            qubit_anharmonicities=anharmonicities,
            resonator_frequencies=resonator_frequencies,
            resonator_kappas=kappas,
            coupling_matrix=coupling_matrix
        )
        
        # Test ground state chi shifts
        ground_state = tuple([0] * n_qubits)
        chi_ground = sim._calculate_chi_shifts(ground_state)
        
        # Validate basic properties
        assert len(chi_ground) == n_qubits, f"Chi array length should match number of resonators for n_qubits={n_qubits}"
        assert all(abs(chi) > 0 for chi in chi_ground), f"All chi shifts should be non-zero for n_qubits={n_qubits}"
        
        # For single qubit, validate against formula
        if n_qubits == 1:
            delta = resonator_frequencies[0] - qubit_frequencies[0]
            expected_chi = coupling_strength**2 / delta
            assert abs(chi_ground[0] - expected_chi) < 0.01 * abs(expected_chi)
        
        # For multi-qubit systems, test that different qubit states produce different chi shifts
        if n_qubits >= 2:
            # Create excited state for first qubit
            excited_state = tuple([1] + [0] * (n_qubits - 1))
            chi_excited = sim._calculate_chi_shifts(excited_state)
            
            # Chi shifts should be different
            assert not np.allclose(chi_ground, chi_excited), f"Ground and excited states should have different chi shifts for n_qubits={n_qubits}"
            
            # At least the first resonator should be affected
            assert abs(chi_ground[0] - chi_excited[0]) > 1e-6, f"R0 chi shift should change when Q0 is excited for n_qubits={n_qubits}"
        
        print(f"✅ {n_qubits}-qubit system validated: χ_ground = {[f'{chi:.3f}' for chi in chi_ground]}")


def test_coupling_matrix_construction_accuracy():
    """
    Test coupling matrix construction accuracy from parameter extraction.
    
    This test validates that the coupling matrix is correctly constructed
    from HighLevelSimulationSetup configurations.
    """
    # Create mock experiment instance
    experiment = ResonatorSweepTransmissionWithExtraInitialLPB.__new__(ResonatorSweepTransmissionWithExtraInitialLPB)
    
    # Test cases with known dispersive shifts and expected couplings
    test_cases = [
        # Single qubit test
        {
            'virtual_qubits': {
                'Q1': {
                    'qubit_frequency': 5000,
                    'readout_frequency': 7000,
                    'readout_dipsersive_shift': 2.0,
                    'anharmonicity': -200,
                    'readout_linewidth': 1.0
                }
            },
            'couplings': {},
            'expected_g': {('Q0', 'R0'): np.sqrt(2.0 * 2000)},  # g = sqrt(chi * delta)
            'expected_qq': {}
        },
        # Two-qubit test with qubit-qubit coupling
        {
            'virtual_qubits': {
                'Q1': {
                    'qubit_frequency': 5000,
                    'readout_frequency': 7000,
                    'readout_dipsersive_shift': 1.5,
                    'anharmonicity': -200,
                    'readout_linewidth': 1.0
                },
                'Q2': {
                    'qubit_frequency': 5200,
                    'readout_frequency': 7500,
                    'readout_dipsersive_shift': 1.2,
                    'anharmonicity': -180,
                    'readout_linewidth': 1.2
                }
            },
            'couplings': {('Q1', 'Q2'): 8.0},  # 8 MHz coupling
            'expected_g': {
                ('Q0', 'R0'): np.sqrt(1.5 * 2000),
                ('Q1', 'R1'): np.sqrt(1.2 * 2300)
            },
            'expected_qq': {('Q0', 'Q1'): 8.0}
        }
    ]
    
    for i, case in enumerate(test_cases):
        # Create mock virtual qubits
        virtual_qubits = {}
        for qubit_id, params in case['virtual_qubits'].items():
            vq = Mock()
            for attr, value in params.items():
                setattr(vq, attr, value)
            virtual_qubits[qubit_id] = vq
        
        # Create mock setup
        setup = Mock()
        setup._virtual_qubits = virtual_qubits
        
        # Mock coupling strength method
        def get_coupling(vq_a, vq_b):
            qubit_list = list(virtual_qubits.values())
            idx_a = qubit_list.index(vq_a)
            idx_b = qubit_list.index(vq_b)
            qubit_keys = list(virtual_qubits.keys())
            key = (qubit_keys[idx_a], qubit_keys[idx_b])
            reverse_key = (qubit_keys[idx_b], qubit_keys[idx_a])
            
            return case['couplings'].get(key, case['couplings'].get(reverse_key, 0))
        
        setup.get_coupling_strength_by_qubit.side_effect = get_coupling
        
        # Extract parameters
        mock_dut_qubit = Mock()
        params_dict, channel_map, string_to_int_channel_map = experiment._extract_params(setup, mock_dut_qubit)
        
        # Validate coupling matrix construction
        coupling_matrix = params_dict['coupling_matrix']
        
        # Check qubit-resonator couplings
        for key, expected_g in case['expected_g'].items():
            assert key in coupling_matrix, f"Missing coupling {key} in test case {i+1}"
            actual_g = coupling_matrix[key]
            relative_error = abs(actual_g - expected_g) / expected_g
            assert relative_error < 0.01, f"Coupling {key} mismatch: expected {expected_g:.3f}, got {actual_g:.3f}"
        
        # Check qubit-qubit couplings
        for key, expected_J in case['expected_qq'].items():
            assert key in coupling_matrix, f"Missing qubit-qubit coupling {key} in test case {i+1}"
            actual_J = coupling_matrix[key]
            assert abs(actual_J - expected_J) < 1e-10, f"Qubit-qubit coupling {key} mismatch: expected {expected_J}, got {actual_J}"
        
        print(f"✅ Test case {i+1}: Coupling matrix construction validated")


def test_multi_qubit_coupling_effects_comprehensive():
    """
    Test comprehensive multi-qubit coupling effects.
    
    This test validates that different qubit states produce different chi shifts
    and that coupling effects are correctly included in the physics.
    """
    # Create 3-qubit system with all-to-all coupling to one resonator
    # This maximizes coupling effects for validation
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000, 5200, 5400],
        qubit_anharmonicities=[-250, -240, -230],
        resonator_frequencies=[7000],  # Single resonator coupled to all qubits
        resonator_kappas=[1.0],
        coupling_matrix={
            ('Q0', 'R0'): 80,   # All qubits couple to same resonator
            ('Q1', 'R0'): 90,
            ('Q2', 'R0'): 100,
            ('Q0', 'Q1'): 5,    # Qubit-qubit couplings
            ('Q1', 'Q2'): 7,
            ('Q0', 'Q2'): 3,
        }
    )
    
    # Generate all possible 3-qubit states
    all_states = []
    chi_shifts = []
    
    for i in range(2**3):  # 8 states total
        state = tuple(int(x) for x in format(i, '03b'))
        all_states.append(state)
        chi = sim._calculate_chi_shifts(state)
        chi_shifts.append(chi[0])  # Only one resonator
    
    # All states should produce different chi shifts
    unique_chi_count = len(set([round(chi, 10) for chi in chi_shifts]))
    assert unique_chi_count == len(all_states), f"Expected {len(all_states)} unique chi shifts, got {unique_chi_count}"
    
    # Test specific state dependencies
    chi_000 = sim._calculate_chi_shifts((0, 0, 0))[0]
    chi_001 = sim._calculate_chi_shifts((0, 0, 1))[0]
    chi_010 = sim._calculate_chi_shifts((0, 1, 0))[0]
    chi_100 = sim._calculate_chi_shifts((1, 0, 0))[0]
    chi_111 = sim._calculate_chi_shifts((1, 1, 1))[0]
    
    # Each single excitation should produce a different chi shift
    single_excitation_chis = [chi_001, chi_010, chi_100]
    for i, chi_i in enumerate(single_excitation_chis):
        for j, chi_j in enumerate(single_excitation_chis):
            if i != j:
                assert abs(chi_i - chi_j) > 1e-8, f"Single excitation states should have different chi shifts: state {i} vs {j}"
    
    # Ground state vs all excited should be very different
    assert abs(chi_000 - chi_111) > abs(chi_000) * 0.1, "Ground and all-excited states should have significantly different chi shifts"
    
    # Validate that chi shifts depend on specific qubit configurations
    state_chi_pairs = list(zip(all_states, chi_shifts))
    state_chi_pairs.sort(key=lambda x: x[1])  # Sort by chi shift
    
    print("Chi shifts by state (sorted):")
    for state, chi in state_chi_pairs:
        print(f"  |{''.join(map(str, state))}⟩: χ = {chi:.6f}")
    
    print(f"✅ Multi-qubit coupling effects validated across {len(all_states)} states")


def test_physics_consistency_across_scales():
    """
    Test physics consistency across different system scales.
    
    This test ensures that the fundamental physics remains consistent
    whether we have 1, 2, 3, or 4 qubits, with proper scaling behavior.
    """
    # Base parameters for consistency testing
    base_coupling = 100  # MHz
    base_freq = 5000     # MHz
    readout_freq = 7000  # MHz
    anharmonicity = -250 # MHz
    
    results = {}
    
    for n_qubits in range(1, 5):
        # Create identical qubits for clean comparison
        sim = MultiQubitDispersiveReadoutSimulator(
            qubit_frequencies=[base_freq] * n_qubits,
            qubit_anharmonicities=[anharmonicity] * n_qubits,
            resonator_frequencies=[readout_freq + i*1000 for i in range(n_qubits)],  # Separate resonators
            resonator_kappas=[1.0] * n_qubits,
            coupling_matrix={('Q0', 'R0'): base_coupling}  # Only first qubit coupled
        )
        
        # Test ground state chi for first resonator
        ground_state = (0,) * n_qubits
        chi_ground = sim._calculate_chi_shifts(ground_state)[0]  # First resonator
        
        # Test excited state chi for first resonator
        excited_state = (1,) + (0,) * (n_qubits - 1)
        chi_excited = sim._calculate_chi_shifts(excited_state)[0]
        
        # Store results
        results[n_qubits] = {
            'chi_ground': chi_ground,
            'chi_excited': chi_excited,
            'delta_chi': chi_excited - chi_ground
        }
        
        # Validate against single-qubit formula for first resonator
        delta_ground = readout_freq - base_freq
        delta_excited = readout_freq - (base_freq + anharmonicity)
        
        expected_chi_ground = base_coupling**2 / delta_ground
        expected_chi_excited = base_coupling**2 / delta_excited
        
        assert abs(chi_ground - expected_chi_ground) < 0.01 * abs(expected_chi_ground)
        assert abs(chi_excited - expected_chi_excited) < 0.01 * abs(expected_chi_excited)
        
        print(f"✅ {n_qubits}-qubit physics consistent: χ_ground={chi_ground:.3f}, χ_excited={chi_excited:.3f}")
    
    # Verify consistency across scales - chi shifts should be identical for uncoupled qubits
    for n in range(2, 5):
        chi_1q = results[1]['chi_ground']
        chi_nq = results[n]['chi_ground']
        
        # Should be identical since only first qubit is coupled
        assert abs(chi_1q - chi_nq) < 1e-10, f"Chi shifts should be identical for uncoupled systems: 1-qubit={chi_1q}, {n}-qubit={chi_nq}"
    
    print("✅ Physics consistency validated across all system scales")


def test_coupling_strength_physics_validation():
    """
    Test that coupling strengths follow proper physics relationships.
    
    Validates g²/Δ scaling, anharmonicity effects, and detuning dependence.
    """
    # Test g²/Δ scaling with different coupling strengths
    coupling_strengths = [50, 75, 100, 125, 150]
    qubit_freq = 5000
    resonator_freq = 7000
    delta = resonator_freq - qubit_freq
    
    chi_values = []
    
    for g in coupling_strengths:
        sim = MultiQubitDispersiveReadoutSimulator(
            qubit_frequencies=[qubit_freq],
            qubit_anharmonicities=[-250],
            resonator_frequencies=[resonator_freq],
            resonator_kappas=[1.0],
            coupling_matrix={('Q0', 'R0'): g}
        )
        
        chi = sim._calculate_chi_shifts((0,))[0]
        chi_values.append(chi)
        
        # Validate against formula
        expected_chi = g**2 / delta
        assert abs(chi - expected_chi) < 0.01 * abs(expected_chi)
    
    # Chi should scale quadratically with coupling strength
    for i in range(1, len(coupling_strengths)):
        g1, g2 = coupling_strengths[0], coupling_strengths[i]
        chi1, chi2 = chi_values[0], chi_values[i]
        
        expected_ratio = (g2/g1)**2
        actual_ratio = chi2/chi1
        
        assert abs(actual_ratio - expected_ratio) < 0.01, f"Chi scaling failed: g_ratio={(g2/g1):.2f}, expected_chi_ratio={expected_ratio:.2f}, actual_chi_ratio={actual_ratio:.2f}"
    
    # Test detuning dependence
    detunings = [1000, 1500, 2000, 2500, 3000]  # Different Δ values
    g = 100
    
    for delta in detunings:
        resonator_freq = qubit_freq + delta
        
        sim = MultiQubitDispersiveReadoutSimulator(
            qubit_frequencies=[qubit_freq],
            qubit_anharmonicities=[-250],
            resonator_frequencies=[resonator_freq],
            resonator_kappas=[1.0],
            coupling_matrix={('Q0', 'R0'): g}
        )
        
        chi = sim._calculate_chi_shifts((0,))[0]
        expected_chi = g**2 / delta
        
        assert abs(chi - expected_chi) < 0.01 * abs(expected_chi), f"Detuning dependence failed for Δ={delta}"
    
    print(f"✅ Coupling strength physics validated: g² scaling and Δ dependence")


def test_anharmonicity_effects_on_chi_shifts():
    """
    Test that anharmonicity correctly affects chi shifts for excited states.
    
    This validates that the effective qubit frequency changes by the
    anharmonicity when the qubit is in an excited state.
    """
    anharmonicity_values = [-200, -250, -300, -180, -220]
    qubit_freq = 5000
    resonator_freq = 7000
    g = 100
    
    for anharm in anharmonicity_values:
        sim = MultiQubitDispersiveReadoutSimulator(
            qubit_frequencies=[qubit_freq],
            qubit_anharmonicities=[anharm],
            resonator_frequencies=[resonator_freq],
            resonator_kappas=[1.0],
            coupling_matrix={('Q0', 'R0'): g}
        )
        
        # Ground state chi
        chi_ground = sim._calculate_chi_shifts((0,))[0]
        expected_chi_ground = g**2 / (resonator_freq - qubit_freq)
        
        # Excited state chi
        chi_excited = sim._calculate_chi_shifts((1,))[0]
        effective_excited_freq = qubit_freq + anharm  # Qubit frequency shifts by anharmonicity
        expected_chi_excited = g**2 / (resonator_freq - effective_excited_freq)
        
        # Validate both states
        assert abs(chi_ground - expected_chi_ground) < 0.01 * abs(expected_chi_ground)
        assert abs(chi_excited - expected_chi_excited) < 0.01 * abs(expected_chi_excited)
        
        # Validate anharmonicity effect magnitude
        chi_difference = chi_excited - chi_ground
        
        # More negative anharmonicity should increase the chi difference
        # (excited state frequency is lower, detuning is larger, chi is smaller)
        assert chi_excited != chi_ground, f"Anharmonicity should change chi shift: anharm={anharm}"
        
        print(f"✅ Anharmonicity {anharm} MHz: χ_ground={chi_ground:.3f}, χ_excited={chi_excited:.3f}, Δχ={chi_difference:.3f}")
    
    print("✅ Anharmonicity effects on chi shifts validated")