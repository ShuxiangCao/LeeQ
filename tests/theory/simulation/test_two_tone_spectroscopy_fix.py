import pytest
import numpy as np
from leeq.experiments.builtin.basic.calibrations.two_tone_spectroscopy import TwoToneQubitSpectroscopy
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
    """Setup with single qubit for testing."""
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


def test_two_tone_same_channel_symmetry(single_qubit_setup):
    """Test two-tone spectroscopy symmetry without workaround."""
    # Setup single qubit
    sim = CWSpectroscopySimulator(single_qubit_setup)
    
    # Use channel 1 directly (from our setup virtual_qubits={1: vq})
    channel = 1
    
    # Get readout parameters
    readout_params = {
        channel: {
            'frequency': 6000.0, 
            'amplitude': 10.0
        }
    }
    
    # Test direct simulator symmetry (bypassing experiment workaround)
    drives_ab = [(channel, 5000.0, 30.0), (channel, 5010.0, 50.0)]
    drives_ba = [(channel, 5010.0, 50.0), (channel, 5000.0, 30.0)]
    
    iq_ab = sim.simulate_spectroscopy_iq(drives_ab, readout_params)
    iq_ba = sim.simulate_spectroscopy_iq(drives_ba, readout_params)
    
    # Should show perfect symmetry
    assert abs(iq_ab[channel] - iq_ba[channel]) < 1e-10, "Direct simulator not symmetric"


def test_two_tone_multiple_drives_same_channel(single_qubit_setup):
    """Test multiple drives on same channel combine correctly."""
    sim = CWSpectroscopySimulator(single_qubit_setup)
    
    channel = 1
    readout_params = {
        channel: {
            'frequency': 6000.0, 
            'amplitude': 10.0
        }
    }
    
    # Test that multiple drives on same channel work correctly
    drives = [(channel, 5000.0, 25.0), (channel, 5010.0, 25.0)]
    
    # This should not raise an error and should produce a valid result
    iq_response = sim.simulate_spectroscopy_iq(drives, readout_params)
    
    assert channel in iq_response, "Response missing for driven channel"
    assert isinstance(iq_response[channel], complex), "IQ response should be complex"


def test_drive_combination_physics_consistency(single_qubit_setup):
    """Test that drive combination follows physics expectations."""
    sim = CWSpectroscopySimulator(single_qubit_setup)
    
    channel = 1
    readout_params = {
        channel: {
            'frequency': 6000.0, 
            'amplitude': 10.0
        }
    }
    
    # Test single drive
    single_drive = [(channel, 5000.0, 50.0)]
    iq_single = sim.simulate_spectroscopy_iq(single_drive, readout_params)
    
    # Test equivalent combined drive  
    combined_drives = [(channel, 5000.0, 25.0), (channel, 5000.0, 25.0)]
    iq_combined = sim.simulate_spectroscopy_iq(combined_drives, readout_params)
    
    # Should be identical since they represent the same total drive
    assert abs(iq_single[channel] - iq_combined[channel]) < 1e-10, "Equivalent drives should give identical results"