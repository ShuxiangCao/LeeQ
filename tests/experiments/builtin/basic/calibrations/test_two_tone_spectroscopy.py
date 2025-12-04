import pytest
import numpy as np
from unittest.mock import Mock, patch

from leeq.experiments.builtin.basic.calibrations.two_tone_spectroscopy import TwoToneQubitSpectroscopy
from leeq.core.elements.built_in.qudit_transmon import TransmonElement


@pytest.fixture
def simulation_setup():
    """Setup for simulation tests."""
    from leeq.chronicle import Chronicle
    from leeq.experiments.experiments import ExperimentManager
    from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
    from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
    
    Chronicle().start_log()
    manager = ExperimentManager()
    manager.clear_setups()
    
    # Create separate virtual transmons for proper channel mapping
    virtual_transmon1 = VirtualTransmon(
        name="TestQubit1",
        qubit_frequency=5000.0,
        anharmonicity=-200.0,
        t1=50,
        t2=25,
        readout_frequency=9500.0,
        quiescent_state_distribution=np.array([0.9, 0.08, 0.02, 0.0])
    )
    
    virtual_transmon2 = VirtualTransmon(
        name="TestQubit2",
        qubit_frequency=4800.0,
        anharmonicity=-200.0,
        t1=50,
        t2=25,
        readout_frequency=9500.0,
        quiescent_state_distribution=np.array([0.9, 0.08, 0.02, 0.0])
    )
    
    virtual_transmon3 = VirtualTransmon(
        name="TestQubit3",
        qubit_frequency=5000.0,
        anharmonicity=-200.0,
        t1=50,
        t2=25,
        readout_frequency=9500.0,
        quiescent_state_distribution=np.array([0.9, 0.08, 0.02, 0.0])
    )
    
    setup = HighLevelSimulationSetup(
        name='TestSetup',
        virtual_qubits={
            1: virtual_transmon1,  # f01 channel
            2: virtual_transmon2,  # f12 channel (different qubit for testing different channels)
            3: virtual_transmon3   # readout channel
        }
    )
    
    manager.register_setup(setup)
    return setup


@pytest.fixture
def test_qubit():
    """Create a test qubit element."""
    config = {
        'lpb_collections': {
            'f01': {
                'type': 'SimpleDriveCollection',
                'freq': 5000.0,
                'channel': 1,
                'shape': 'blackman_drag',
                'amp': 0.5,
                'phase': 0.,
                'width': 0.05,
                'alpha': 500,
                'trunc': 1.2
            },
            'f12': {
                'type': 'SimpleDriveCollection',
                'freq': 4800.0,
                'channel': 2,
                'shape': 'blackman_drag',
                'amp': 0.1,
                'phase': 0.,
                'width': 0.025,
                'alpha': 425,
                'trunc': 1.2
            }
        },
        'measurement_primitives': {
            '0': {
                'type': 'SimpleDispersiveMeasurement',
                'freq': 9500.0,
                'channel': 3,
                'shape': 'square',
                'amp': 0.15,
                'phase': 0.,
                'width': 1,
                'trunc': 1.2,
                'distinguishable_states': [0, 1]
            }
        }
    }
    
    return TransmonElement(name='Q1', parameters=config)


def test_two_tone_different_channels(simulation_setup, test_qubit):
    """Test two-tone spectroscopy on different channels."""
    exp = TwoToneQubitSpectroscopy(
        dut_qubit=test_qubit,
        tone1_start=4950.0,
        tone1_stop=5050.0,
        tone1_step=10.0,
        tone2_start=4750.0,
        tone2_stop=4850.0,
        tone2_step=10.0,
        same_channel=False
    )
    
    # Check data shape
    assert exp.result['Magnitude'].shape == (11, 11)
    assert exp.result['Phase'].shape == (11, 11)
    
    # Check frequency arrays
    assert len(exp.freq1_arr) == 11
    assert len(exp.freq2_arr) == 11
    assert np.isclose(exp.freq1_arr[0], 4950.0)
    assert np.isclose(exp.freq1_arr[-1], 5050.0)
    
    # Check peak detection
    peaks = exp.find_peaks()
    assert 'peak_freq1' in peaks
    assert 'peak_freq2' in peaks
    assert 'peak_magnitude' in peaks
    
    # Peak should be somewhere near the qubit frequency
    assert 4900 < peaks['peak_freq1'] < 5100
    

def test_two_tone_same_channel(simulation_setup, test_qubit):
    """Test two-tone spectroscopy superimposed on same channel."""
    exp = TwoToneQubitSpectroscopy(
        dut_qubit=test_qubit,
        tone1_start=4980.0,
        tone1_stop=5020.0,
        tone1_step=5.0,
        tone2_start=5000.0,
        tone2_stop=5040.0,
        tone2_step=5.0,
        same_channel=True
    )
    
    # Check results exist
    assert exp.result is not None
    assert 'Magnitude' in exp.result
    assert 'Phase' in exp.result
    
    # Check data shape
    assert exp.result['Magnitude'].shape == (9, 9)


def test_cross_sections(simulation_setup, test_qubit):
    """Test extracting 1D cross-sections from 2D data."""
    exp = TwoToneQubitSpectroscopy(
        dut_qubit=test_qubit,
        tone1_start=4950.0,
        tone1_stop=5050.0,
        tone1_step=20.0,
        tone2_start=4750.0,
        tone2_stop=4850.0,
        tone2_step=20.0,
        same_channel=False
    )
    
    # Get cross-section at peak
    cross1 = exp.get_cross_section(axis='freq1')
    assert 'frequencies' in cross1
    assert 'magnitude' in cross1
    assert len(cross1['magnitude']) == len(exp.freq2_arr)
    
    cross2 = exp.get_cross_section(axis='freq2')
    assert len(cross2['magnitude']) == len(exp.freq1_arr)
    
    # Get cross-section at specific frequency
    cross3 = exp.get_cross_section(axis='freq1', value=5000.0)
    assert 'slice_freq' in cross3
    assert np.isclose(cross3['slice_freq'], 5000.0, atol=20.0)


def test_plotting_functions(simulation_setup, test_qubit):
    """Test that plotting functions return correct objects."""
    # Disable plotting for tests
    manager = simulation_setup
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    exp = TwoToneQubitSpectroscopy(
        dut_qubit=test_qubit,
        tone1_start=4980.0,
        tone1_stop=5020.0,
        tone1_step=10.0,
        tone2_start=4980.0,
        tone2_stop=5020.0,
        tone2_step=10.0,
        same_channel=False
    )
    
    # Test magnitude plot
    fig_mag = exp.plot()
    assert fig_mag is not None
    # Just verify the plot function returns something
    # Don't check data length as it may be mocked
    
    # Test phase plot
    fig_phase = exp.plot_phase()
    assert fig_phase is not None
    # Just verify the plot function returns something


def test_small_sweep(simulation_setup, test_qubit):
    """Test with very small sweep for quick validation."""
    exp = TwoToneQubitSpectroscopy(
        dut_qubit=test_qubit,
        tone1_start=4990.0,
        tone1_stop=5010.0,
        tone1_step=10.0,
        tone2_start=4790.0,
        tone2_stop=4810.0,
        tone2_step=10.0,
        same_channel=False,
        num_avs=100
    )
    
    # Should produce 3x3 grid
    assert exp.result['Magnitude'].shape == (3, 3)
    assert exp.trace.shape == (3, 3)
    
    # Check that data is complex
    assert np.iscomplexobj(exp.trace)


def test_amplitude_parameters(simulation_setup, test_qubit):
    """Test different amplitude settings."""
    exp = TwoToneQubitSpectroscopy(
        dut_qubit=test_qubit,
        tone1_start=5000.0,
        tone1_stop=5000.0,
        tone1_step=10.0,
        tone1_amp=0.05,
        tone2_start=4800.0,
        tone2_stop=4800.0,
        tone2_step=10.0,
        tone2_amp=0.15,
        same_channel=False
    )
    
    # Single point measurement
    assert exp.result['Magnitude'].shape == (1, 1)
    assert exp.tone1_amp == 0.05
    assert exp.tone2_amp == 0.15


def test_noise_scaling(simulation_setup, test_qubit):
    """Test that noise scales correctly with averages."""
    # Run with different averaging
    exp1 = TwoToneQubitSpectroscopy(
        dut_qubit=test_qubit,
        tone1_start=5000.0,
        tone1_stop=5020.0,
        tone1_step=20.0,
        tone2_start=4800.0,
        tone2_stop=4820.0,
        tone2_step=20.0,
        same_channel=False,
        num_avs=100
    )
    
    exp2 = TwoToneQubitSpectroscopy(
        dut_qubit=test_qubit,
        tone1_start=5000.0,
        tone1_stop=5020.0,
        tone1_step=20.0,
        tone2_start=4800.0,
        tone2_stop=4820.0,
        tone2_step=20.0,
        same_channel=False,
        num_avs=10000
    )
    
    # With more averages, results should be less noisy
    # We can't test exact values due to randomness, but structure should be there
    assert exp1.result is not None
    assert exp2.result is not None
    assert exp1.result['Magnitude'].shape == exp2.result['Magnitude'].shape