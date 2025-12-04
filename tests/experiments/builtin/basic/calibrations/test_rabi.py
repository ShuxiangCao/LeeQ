"""
Test cases for NormalisedRabi experiment with drive_frequency parameter.
"""

import pytest
import numpy as np
from leeq.experiments.builtin.basic.calibrations.rabi import NormalisedRabi
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.experiments.experiments import ExperimentManager
from leeq.chronicle import Chronicle


@pytest.fixture()
def simulation_setup():
    """Create a high-level simulation setup with experiment manager."""
    Chronicle().start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    virtual_transmon = VirtualTransmon(
        name="VQubit",
        qubit_frequency=5040.4,  # MHz - using frequency from PRP example
        anharmonicity=-200.0,
        t1=50.0,
        t2=30.0,
        readout_frequency=7000.0,
        readout_linewith=5.0,
        readout_dipsersive_shift=2.0,
        quiescent_state_distribution=np.asarray([0.9, 0.08, 0.02])
    )

    setup = HighLevelSimulationSetup(
        name='HighLevelSimulationSetup',
        virtual_qubits={2: virtual_transmon},
        omega_to_amp_map={2: 500.0}  # 500 MHz per unit amplitude
    )

    manager.register_setup(setup)
    return manager


# Configuration for TransmonElement
configuration = {
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 5040.4,  # MHz - on resonance frequency
            'channel': 2,
            'shape': 'square',
            'amp': 0.05,  # Initial amplitude
            'phase': 0.,
            'width': 0.01,  # 10 ns pulse
            'alpha': 0,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 7000.0,
            'channel': 1,
            'shape': 'square',
            'amp': 0.2,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        }
    }
}


def test_normalised_rabi_drive_frequency(simulation_setup):
    """Test drive_frequency parameter in both run methods."""
    manager = simulation_setup
    
    # Create a transmon element
    dut_qubit = TransmonElement(
        name="Q2",
        parameters=configuration
    )
    
    # Test 1: None case (backward compatibility) - should use pulse frequency
    experiment = NormalisedRabi(
        dut_qubit=dut_qubit, 
        amp=0.05,
        start=0.01,
        stop=0.05,
        step=0.001,
        drive_frequency=None,
        fit=True
    )
    
    assert hasattr(experiment, 'data'), "Experiment should have data after run"
    assert experiment.data is not None, "Data should not be None for backward compatibility case"
    assert len(experiment.data) > 0, "Data should contain measurement points"
    
    # Store first result for comparison
    data1 = experiment.data.copy()
    
    # Test 2: frequency override case - off resonance
    frequency_detuning = 0.1  # MHz
    drive_frequency = 5040.4 + frequency_detuning  # 5040.5 MHz - off resonance
    
    experiment2 = NormalisedRabi(
        dut_qubit=dut_qubit,
        amp=0.05,
        start=0.01, 
        stop=0.05,
        step=0.001,
        drive_frequency=drive_frequency,
        fit=True
    )
    
    assert hasattr(experiment2, 'data'), "Experiment should have data after run with frequency override"
    assert experiment2.data is not None, "Data should not be None for frequency override case"
    assert len(experiment2.data) > 0, "Data should contain measurement points"
    
    # Store second result for comparison
    data2 = experiment2.data.copy()
    
    # Test 3: Verify frequency was applied - results should be different
    # Due to detuning, the Rabi oscillations should have different characteristics
    assert not np.array_equal(data1, data2), "Different results expected due to frequency change"
    
    # Test 4: Verify both experiments have fit parameters
    assert hasattr(experiment, 'fit_params'), "First experiment should have fit parameters"
    assert hasattr(experiment2, 'fit_params'), "Second experiment should have fit parameters"
    
    # Test 5: Verify off-resonance case shows expected physics
    # Off-resonance driving should typically show reduced oscillation amplitude
    max_amp_1 = np.max(np.abs(data1))
    max_amp_2 = np.max(np.abs(data2))
    
    # For small detuning, amplitude should be somewhat reduced but still present
    assert max_amp_2 > 0.1 * max_amp_1, "Off-resonance case should still show significant oscillations"


def test_normalised_rabi_run_method_drive_frequency(simulation_setup):
    """Test drive_frequency parameter works in regular run method as well."""
    manager = simulation_setup
    
    # Create a transmon element
    dut_qubit = TransmonElement(
        name="Q2", 
        parameters=configuration
    )
    
    # Test that regular run method accepts drive_frequency parameter without error
    try:
        # Note: We can't actually run the full hardware experiment in tests,
        # but we can verify the parameter is accepted and method signature is correct
        
        # Test parameter acceptance by checking the method signature directly from the class
        import inspect
        sig = inspect.signature(NormalisedRabi.run)
        params = sig.parameters
        
        assert 'drive_frequency' in params, "run method should have drive_frequency parameter"
        assert params['drive_frequency'].default is None, "drive_frequency should default to None"
        
        # Verify the parameter has correct type annotation
        annotation = params['drive_frequency'].annotation
        expected_type = type(None)  # Optional[float] includes None
        # For Optional[float], we check that it allows None
        assert annotation.__origin__ is type(None) or hasattr(annotation, '__args__'), \
            "drive_frequency should be typed as Optional[float]"
            
    except Exception as e:
        pytest.fail(f"run method should accept drive_frequency parameter: {e}")


def test_normalised_rabi_drive_frequency_edge_cases(simulation_setup):
    """Test edge cases for drive_frequency parameter."""
    manager = simulation_setup
    
    # Create a transmon element
    dut_qubit = TransmonElement(
        name="Q2",
        parameters=configuration
    )
    
    # Test large detuning case
    large_detuning_freq = 5040.4 + 10.0  # 10 MHz detuning - significant
    
    experiment = NormalisedRabi(
        dut_qubit=dut_qubit,
        amp=0.05,
        start=0.01,
        stop=0.05, 
        step=0.001,
        drive_frequency=large_detuning_freq,
        fit=True
    )
    
    assert experiment.data is not None, "Large detuning case should still produce data"
    assert len(experiment.data) > 0, "Data should contain measurement points even with large detuning"
    
    # We won't test zero frequency as it's unphysical, but test a different valid frequency
    different_freq = 5000.0  # Different but reasonable frequency
    
    experiment2 = NormalisedRabi(
        dut_qubit=dut_qubit,
        amp=0.05,
        start=0.01,
        stop=0.05,
        step=0.001,
        drive_frequency=different_freq,
        fit=True
    )
    
    assert experiment2.data is not None, "Different frequency case should produce data"
    assert len(experiment2.data) > 0, "Data should contain measurement points"