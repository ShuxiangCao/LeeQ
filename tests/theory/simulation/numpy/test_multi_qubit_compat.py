import pytest
import numpy as np
from leeq.theory.simulation.numpy.dispersive_readout.simulator import (
    DispersiveReadoutSimulatorSyntheticData
)
from leeq.theory.simulation.numpy.dispersive_readout.multi_qubit_simulator import (
    MultiQubitDispersiveReadoutSimulator
)


def test_single_qubit_backward_compatibility():
    """Verify N=1 case works and produces reasonable traces"""
    # Create single-qubit multi-qubit simulator
    multi_sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000],
        qubit_anharmonicities=[-250],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={('Q0', 'R0'): 100},
        amp=1.0,
        width=10.0,
        baseline=0.1
    )
    
    # Test basic functionality
    assert multi_sim.n_qubits == 1
    assert multi_sim.n_resonators == 1
    
    # Test state conversion
    assert multi_sim._get_state_tuple(0) == (0,)
    assert multi_sim._get_state_tuple(1) == (1,)
    assert multi_sim._get_state_tuple((0,)) == (0,)
    assert multi_sim._get_state_tuple((1,)) == (1,)
    
    # Test chi shift calculation
    chi_0 = multi_sim._calculate_chi_shifts((0,))
    chi_1 = multi_sim._calculate_chi_shifts((1,))
    
    # Chi shifts should be different for ground vs excited state
    assert not np.allclose(chi_0, chi_1)
    assert len(chi_0) == 1
    assert len(chi_1) == 1
    
    # Test trace generation
    trace_0 = multi_sim.simulate_trace(0, 0, 7000, noise_std=0)
    trace_1 = multi_sim.simulate_trace(1, 0, 7000, noise_std=0)
    
    # Traces should be complex arrays
    assert isinstance(trace_0, np.ndarray)
    assert isinstance(trace_1, np.ndarray)
    assert trace_0.dtype == np.complex128
    assert trace_1.dtype == np.complex128
    
    # Traces should be different for different states
    assert not np.allclose(trace_0, trace_1)
    
    # Should have same length
    assert len(trace_0) == len(trace_1)


def test_import_integration():
    """Test module can be imported alongside existing code"""
    # Test that we can import all the relevant classes
    from leeq.theory.simulation.numpy.dispersive_readout import (
        DispersiveReadoutSimulator,
        ChiShiftCalculator
    )
    from leeq.theory.simulation.numpy.dispersive_readout.multi_qubit_simulator import (
        MultiQubitDispersiveReadoutSimulator
    )
    
    # Verify inheritance chain
    assert issubclass(MultiQubitDispersiveReadoutSimulator, DispersiveReadoutSimulator)
    
    # Test that both simulators can coexist
    single_sim = DispersiveReadoutSimulatorSyntheticData(
        f_r=7000,
        kappa=1.0,
        chis=[0, -0.25],
        use_physics_model=False,
        amp=1.0,
        width=10.0
    )
    
    multi_sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000],
        qubit_anharmonicities=[-250],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={('Q0', 'R0'): 100},
        amp=1.0,
        width=10.0
    )
    
    # Both should be instances of their respective classes
    assert isinstance(single_sim, DispersiveReadoutSimulatorSyntheticData)
    assert isinstance(multi_sim, MultiQubitDispersiveReadoutSimulator)
    
    # Multi-qubit should also be instance of base class
    assert isinstance(multi_sim, DispersiveReadoutSimulator)


def test_state_validation():
    """Test state validation for single qubit case"""
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000],
        qubit_anharmonicities=[-250],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={('Q0', 'R0'): 100}
    )
    
    # Valid states
    assert sim._validate_state(0)
    assert sim._validate_state(1)
    assert sim._validate_state((0,))
    assert sim._validate_state((1,))
    
    # Invalid states
    assert not sim._validate_state(2)  # Out of range for 1 qubit
    assert not sim._validate_state(-1)
    assert not sim._validate_state((0, 1))  # Wrong length
    assert not sim._validate_state((2,))  # Invalid qubit state


def test_error_handling():
    """Test proper error handling in single qubit case"""
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000],
        qubit_anharmonicities=[-250],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={('Q0', 'R0'): 100}
    )
    
    # Test invalid joint state
    with pytest.raises(ValueError, match="Invalid joint state"):
        sim.simulate_trace(2, 0, 7000)  # State 2 doesn't exist for 1 qubit
    
    # Test invalid resonator ID
    with pytest.raises(ValueError, match="Resonator ID"):
        sim.simulate_trace(0, 1, 7000)  # Only resonator 0 exists
    
    with pytest.raises(ValueError, match="Resonator ID"):
        sim.simulate_trace(0, -1, 7000)


def test_multiplexed_readout_single_resonator():
    """Test multiplexed readout with single resonator"""
    sim = MultiQubitDispersiveReadoutSimulator(
        qubit_frequencies=[5000],
        qubit_anharmonicities=[-250],
        resonator_frequencies=[7000],
        resonator_kappas=[1.0],
        coupling_matrix={('Q0', 'R0'): 100}
    )
    
    # Test single resonator multiplexed readout
    traces = sim.simulate_multiplexed_readout(
        joint_state=0,
        probe_frequencies=[7000],
        noise_std=0
    )
    
    assert len(traces) == 1
    assert isinstance(traces[0], np.ndarray)
    assert traces[0].dtype == np.complex128
    
    # Test with wrong number of frequencies
    with pytest.raises(ValueError, match="Expected 1 probe frequencies"):
        sim.simulate_multiplexed_readout(
            joint_state=0,
            probe_frequencies=[7000, 7500],  # Too many frequencies
            noise_std=0
        )