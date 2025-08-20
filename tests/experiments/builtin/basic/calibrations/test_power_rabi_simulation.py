"""
Test cases for PowerRabi high-level simulation implementation.
"""

import pytest
import numpy as np
from leeq.experiments.builtin.basic.calibrations.rabi import PowerRabi
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
        qubit_frequency=5000.0,
        anharmonicity=-200.0,
        t1=50.0,
        t2=30.0,
        readout_frequency=7000.0,
        readout_linewith=5.0,  # Note: typo in original class
        readout_dipsersive_shift=2.0,  # Note: typo in original class
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
            'freq': 5000.0,
            'channel': 2,
            'shape': 'square',
            'amp': 0.5,  # Initial amplitude
            'phase': 0.,
            'width': 0.02,  # 20 ns pi pulse
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


@pytest.fixture
def dut_qubit():
    """Create a TransmonElement for testing."""
    dut = TransmonElement(
        name='test_qubit',
        parameters=configuration
    )
    return dut


def test_power_rabi_basic(simulation_setup, dut_qubit):
    """Test basic PowerRabi functionality with simulation."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    # Run PowerRabi - the experiment runs in __init__
    rabi = PowerRabi(
        dut_qubit=dut_qubit,
        amp_start=0.01,
        amp_stop=0.2,
        amp_step=0.01,
        fit=True,
        update=False
    )
    
    # Check that data was collected
    assert hasattr(rabi, 'data')
    assert len(rabi.data) == len(np.arange(0.01, 0.2, 0.01))
    
    # Check fit results
    assert hasattr(rabi, 'fit_params')
    assert 'Frequency' in rabi.fit_params
    assert 'Amplitude' in rabi.fit_params
    assert 'Phase' in rabi.fit_params
    assert 'Offset' in rabi.fit_params
    
    # Check that we see oscillations
    assert max(rabi.data) > 0.8  # Should reach high population
    assert min(rabi.data) < 0.2  # Should have low population at some points


def test_power_rabi_finds_optimal_amplitude(simulation_setup, dut_qubit):
    """Test that PowerRabi finds reasonable pi amplitude."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    # Run with update=True to find optimal amplitude
    rabi = PowerRabi(
        dut_qubit=dut_qubit,
        amp_start=0.01,
        amp_stop=0.2,
        amp_step=0.005,
        fit=True,
        update=True
    )
    
    # Check optimal amplitude was found
    assert hasattr(rabi, 'optimal_amp')
    # For 20ns pulse with omega_per_amp=500MHz: optimal_amp = 1/(500*0.02) = 0.1
    assert 0.08 <= rabi.optimal_amp <= 0.12
    
    # Check that the amplitude was updated
    updated_amp = dut_qubit.get_c1('f01')['X'].amp
    assert abs(updated_amp - rabi.optimal_amp) < 1e-10


def test_power_rabi_with_custom_width(simulation_setup, dut_qubit):
    """Test PowerRabi with custom pulse width."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    # Run with custom width
    custom_width = 0.04  # 40 ns
    rabi = PowerRabi(
        dut_qubit=dut_qubit,
        width=custom_width,
        amp_start=0.01,
        amp_stop=0.1,
        amp_step=0.005,
        fit=True,
        update=True
    )
    
    # With double the pulse width, optimal amplitude should be half
    # Expected: 1/(500*0.04) = 0.05
    assert hasattr(rabi, 'optimal_amp')
    assert 0.045 <= rabi.optimal_amp <= 0.055


def test_power_rabi_no_fit(simulation_setup, dut_qubit):
    """Test PowerRabi without fitting."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    # Run without fitting
    rabi = PowerRabi(
        dut_qubit=dut_qubit,
        amp_start=0.01,
        amp_stop=0.2,
        amp_step=0.01,
        fit=False,
        update=False
    )
    
    # Data should still be collected
    assert hasattr(rabi, 'data')
    assert len(rabi.data) > 0
    
    # But no fit params or optimal amp
    assert not hasattr(rabi, 'fit_params')
    assert not hasattr(rabi, 'optimal_amp')


def test_power_rabi_noise_effects(simulation_setup, dut_qubit):
    """Test PowerRabi with different noise settings."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    # Run with noise disabled
    manager.status.set_parameter('Sampling_Noise', False)
    rabi_no_noise = PowerRabi(
        dut_qubit=dut_qubit,
        amp_start=0.0,
        amp_stop=0.2,
        amp_step=0.005,
        fit=False
    )
    
    # First point should be exactly 0 (no rotation)
    assert rabi_no_noise.data[0] == 0.0
    
    # Enable noise
    manager.status.set_parameter('Sampling_Noise', True)
    rabi_with_noise = PowerRabi(
        dut_qubit=dut_qubit,
        amp_start=0.01,
        amp_stop=0.2,
        amp_step=0.01,
        fit=False
    )
    
    # With noise, most values shouldn't be exactly 0 or 1
    # But some might be by chance, so check that there's variation
    assert len(set(rabi_with_noise.data)) > 10  # Should have many different values
    assert np.std(rabi_with_noise.data) > 0.01  # Should have some variation


def test_power_rabi_plot_exists(simulation_setup, dut_qubit):
    """Test that PowerRabi has a plot method."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    rabi = PowerRabi(
        dut_qubit=dut_qubit,
        amp_start=0.01,
        amp_stop=0.2,
        amp_step=0.01,
        fit=True
    )
    
    # Check plot method exists and returns a figure
    assert hasattr(rabi, 'plot')
    fig = rabi.plot()
    assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])