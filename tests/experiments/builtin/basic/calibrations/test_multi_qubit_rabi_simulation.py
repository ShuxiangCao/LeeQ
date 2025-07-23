"""
Test cases for MultiQubitRabi high-level simulation implementation.
"""

import pytest
import numpy as np
from leeq.experiments.builtin.basic.calibrations.rabi import MultiQubitRabi
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.experiments.experiments import ExperimentManager
from labchronicle import Chronicle


@pytest.fixture()
def simulation_setup():
    """Create a high-level simulation setup with multiple qubits."""
    Chronicle().start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    # Create two virtual transmons with different parameters
    virtual_transmon_1 = VirtualTransmon(
        name="VQubit1",
        qubit_frequency=5000.0,
        anharmonicity=-200.0,
        t1=50.0,  # 50 μs T1
        t2=30.0,  # 30 μs T2
        readout_frequency=7000.0,
        readout_linewith=5.0,
        readout_dipsersive_shift=2.0,
        quiescent_state_distribution=np.asarray([0.9, 0.08, 0.02])
    )
    
    virtual_transmon_2 = VirtualTransmon(
        name="VQubit2",
        qubit_frequency=5100.0,
        anharmonicity=-210.0,
        t1=60.0,  # 60 μs T1
        t2=35.0,  # 35 μs T2
        readout_frequency=7100.0,
        readout_linewith=5.0,
        readout_dipsersive_shift=2.0,
        quiescent_state_distribution=np.asarray([0.85, 0.12, 0.03])
    )

    setup = HighLevelSimulationSetup(
        name='HighLevelSimulationSetup',
        virtual_qubits={2: virtual_transmon_1, 4: virtual_transmon_2}
    )
    
    manager.register_setup(setup)
    return manager


def create_transmon_config(channel, frequency):
    """Create configuration for a TransmonElement."""
    return {
        'lpb_collections': {
            'f01': {
                'type': 'SimpleDriveCollection',
                'freq': frequency,
                'channel': channel,
                'shape': 'square',
                'amp': 0.5,
                'phase': 0.,
                'width': 0.02,
                'alpha': 0,
                'trunc': 1.2
            }
        },
        'measurement_primitives': {
            '0': {
                'type': 'SimpleDispersiveMeasurement',
                'freq': frequency + 2000,
                'channel': channel + 1,
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
def dut_qubits():
    """Create two TransmonElements for testing."""
    qubit1 = TransmonElement(
        name='test_qubit_1',
        parameters=create_transmon_config(2, 5000.0)
    )
    qubit2 = TransmonElement(
        name='test_qubit_2',
        parameters=create_transmon_config(4, 5100.0)
    )
    return [qubit1, qubit2]


def test_multi_qubit_rabi_basic(simulation_setup, dut_qubits):
    """Test basic MultiQubitRabi functionality with simulation."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    manager.status.set_parameter('Sampling_Noise', False)
    
    # Run MultiQubitRabi
    rabi_exp = MultiQubitRabi(
        duts=dut_qubits,
        amps=0.1,
        start=0.01,
        stop=0.08,
        step=0.001,
        fit=False
    )
    
    # Check that data was collected for both qubits
    assert hasattr(rabi_exp, 'data')
    assert len(rabi_exp.data) == 2  # Two qubits
    
    # Check data length
    expected_points = int((0.08 - 0.01) / 0.001)
    assert len(rabi_exp.data[0]) == expected_points
    assert len(rabi_exp.data[1]) == expected_points
    
    # Check that data oscillates (should not be constant)
    for data in rabi_exp.data:
        assert np.std(data) > 0.1  # Should have variation
        assert -1 <= np.min(data) <= 1
        assert -1 <= np.max(data) <= 1


def test_multi_qubit_rabi_different_frequencies(simulation_setup, dut_qubits):
    """Test Rabi oscillations with different frequencies per qubit."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    manager.status.set_parameter('Sampling_Noise', False)
    
    # Run with different amplitudes to get different Rabi frequencies
    rabi_exp = MultiQubitRabi(
        duts=dut_qubits,
        amps=[0.1, 0.2],  # Different amplitudes
        start=0.01,
        stop=0.15,
        step=0.001,
        fit=True
    )
    
    # Check that fitting was done
    assert hasattr(rabi_exp, 'fit_params')
    assert len(rabi_exp.fit_params) == 2
    
    # Check that the two qubits have different frequencies
    freq1 = rabi_exp.fit_params[0]['Frequency']
    freq2 = rabi_exp.fit_params[1]['Frequency']
    
    # The frequency should be proportional to amplitude
    # freq2 should be approximately 2 * freq1
    assert abs(freq2 / freq1 - 2.0) < 0.1


def test_multi_qubit_rabi_with_decoherence(simulation_setup, dut_qubits):
    """Test that T1/T2 effects are included."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    manager.status.set_parameter('Sampling_Noise', False)
    
    # Run with long time to see decoherence
    rabi_exp = MultiQubitRabi(
        duts=dut_qubits,
        amps=0.05,
        start=0.01,
        stop=0.5,  # Long time compared to T2
        step=0.005,
        fit=False
    )
    
    # Check that oscillations decay over time
    for data in rabi_exp.data:
        # Get envelope by looking at extrema
        first_quarter = data[:len(data)//4]
        last_quarter = data[3*len(data)//4:]
        
        # The amplitude should decrease
        first_amplitude = np.max(np.abs(first_quarter))
        last_amplitude = np.max(np.abs(last_quarter))
        
        # For very long times, the oscillations should decay
        # But with T2=30-35us and time up to 500us, the decay should be visible
        # exp(-500/30) ≈ 0.000001, so we should see significant decay
        assert last_amplitude < first_amplitude  # Should decay at least somewhat


def test_multi_qubit_rabi_time_resolution(simulation_setup, dut_qubits):
    """Test effect of time resolution on results."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    manager.status.set_parameter('Sampling_Noise', False)
    
    # Run with coarse resolution
    rabi_coarse = MultiQubitRabi(
        duts=dut_qubits,
        amps=0.2,
        start=0.01,
        stop=0.05,
        step=0.005,  # Coarse
        fit=True
    )
    
    # Run with fine resolution
    rabi_fine = MultiQubitRabi(
        duts=dut_qubits,
        amps=0.2,
        start=0.01,
        stop=0.05,
        step=0.0005,  # Fine
        fit=True
    )
    
    # Both should capture the oscillation frequency
    freq_coarse = rabi_coarse.fit_params[0]['Frequency']
    freq_fine = rabi_fine.fit_params[0]['Frequency']
    
    # Frequencies should be similar but fine should be more accurate
    assert abs(freq_coarse - freq_fine) / freq_fine < 0.1


def test_multi_qubit_rabi_with_noise(simulation_setup, dut_qubits):
    """Test MultiQubitRabi with sampling noise enabled."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    manager.status.set_parameter('Sampling_Noise', True)
    manager.status.set_parameter('Shot_Number', 1000)
    
    # Run experiment
    rabi_exp = MultiQubitRabi(
        duts=dut_qubits,
        amps=0.1,
        start=0.01,
        stop=0.08,
        step=0.001,
        fit=True
    )
    
    # With noise, fitted parameters should still be reasonable
    for fit_param in rabi_exp.fit_params:
        assert fit_param['Frequency'] > 0
        assert 0 < fit_param['Amplitude'] < 2
        assert abs(fit_param['Offset']) < 0.5


def test_multi_qubit_rabi_single_qubit(simulation_setup):
    """Test MultiQubitRabi with just one qubit."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    # Create single qubit
    qubit = TransmonElement(
        name='single_qubit',
        parameters=create_transmon_config(2, 5000.0)
    )
    
    # Run with single qubit
    rabi_exp = MultiQubitRabi(
        duts=[qubit],
        amps=0.1,
        start=0.01,
        stop=0.08,
        step=0.001,
        fit=True
    )
    
    assert len(rabi_exp.data) == 1
    assert len(rabi_exp.fit_params) == 1
    
    # Should still show oscillations
    assert np.std(rabi_exp.data[0]) > 0.1


def test_multi_qubit_rabi_plot_methods(simulation_setup, dut_qubits):
    """Test that MultiQubitRabi has plotting methods."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    rabi_exp = MultiQubitRabi(
        duts=dut_qubits,
        amps=0.1,
        start=0.01,
        stop=0.05,
        step=0.001,
        fit=True
    )
    
    # Check plot methods exist
    assert hasattr(rabi_exp, 'plot_all')
    assert hasattr(rabi_exp, 'plot')
    
    # Test plotting individual qubit
    fig = rabi_exp.plot(0)
    assert fig is not None
    
    fig2 = rabi_exp.plot(1)
    assert fig2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])