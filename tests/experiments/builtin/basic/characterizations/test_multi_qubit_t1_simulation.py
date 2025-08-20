"""
Test cases for MultiQubitT1 high-level simulation implementation.
"""

import pytest
import numpy as np
from leeq.experiments.builtin.basic.characterizations.t1 import MultiQubitT1
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.experiments.experiments import ExperimentManager
from leeq.chronicle import Chronicle


@pytest.fixture()
def simulation_setup():
    """Create a high-level simulation setup with multiple qubits."""
    Chronicle().start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    # Create two virtual transmons with different T1 values
    virtual_transmon_1 = VirtualTransmon(
        name="VQubit1",
        qubit_frequency=5000.0,
        anharmonicity=-200.0,
        t1=50.0,  # 50 μs T1
        t2=30.0,
        readout_frequency=7000.0,
        readout_linewith=5.0,
        readout_dipsersive_shift=2.0,
        quiescent_state_distribution=np.asarray([0.9, 0.08, 0.02])
    )
    
    virtual_transmon_2 = VirtualTransmon(
        name="VQubit2",
        qubit_frequency=5100.0,
        anharmonicity=-210.0,
        t1=70.0,  # 70 μs T1 - different from qubit 1
        t2=40.0,
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


def test_multi_qubit_t1_basic(simulation_setup, dut_qubits):
    """Test basic MultiQubitT1 functionality with simulation."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    # Run MultiQubitT1
    t1_exp = MultiQubitT1(
        duts=dut_qubits,
        time_length=150.0,  # Go to ~3*T1 of first qubit
        time_resolution=2.0
    )
    
    # Check that traces were collected
    assert hasattr(t1_exp, 'traces')
    assert len(t1_exp.traces) == 2  # Two qubits
    
    # Check data length
    expected_points = int(150.0 / 2.0)
    assert len(t1_exp.traces[0]) == expected_points
    assert len(t1_exp.traces[1]) == expected_points
    
    # Check that data starts near 1 (excited state)
    assert 0.9 <= t1_exp.traces[0][0] <= 1.0
    assert 0.9 <= t1_exp.traces[1][0] <= 1.0
    
    # Check that data decays
    assert t1_exp.traces[0][-1] < 0.5  # Should decay significantly
    assert t1_exp.traces[1][-1] < 0.5


def test_multi_qubit_t1_independent_decay(simulation_setup, dut_qubits):
    """Test that each qubit decays independently with its own T1."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    manager.status.set_parameter('Sampling_Noise', False)  # Disable noise for clean test
    
    # Run experiment
    t1_exp = MultiQubitT1(
        duts=dut_qubits,
        time_length=200.0,
        time_resolution=5.0
    )
    
    # Extract time array
    times = np.arange(0.0, 200.0, 5.0)
    
    # Check that qubit 1 follows T1=50μs decay
    expected_q1 = np.exp(-times / 50.0)
    np.testing.assert_allclose(t1_exp.traces[0], expected_q1, rtol=1e-10)
    
    # Check that qubit 2 follows T1=70μs decay
    expected_q2 = np.exp(-times / 70.0)
    np.testing.assert_allclose(t1_exp.traces[1], expected_q2, rtol=1e-10)
    
    # Verify they are different (since T1 values are different)
    assert not np.allclose(t1_exp.traces[0], t1_exp.traces[1])


def test_multi_qubit_t1_with_noise(simulation_setup, dut_qubits):
    """Test MultiQubitT1 with sampling noise enabled."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    manager.status.set_parameter('Sampling_Noise', True)
    manager.status.set_parameter('Shot_Number', 1000)
    
    # Run experiment
    t1_exp = MultiQubitT1(
        duts=dut_qubits,
        time_length=100.0,
        time_resolution=2.0
    )
    
    # With noise, values should vary
    # Check that traces have variation (not smooth exponential)
    for trace in t1_exp.traces:
        differences = np.diff(trace)
        # Some differences should be positive (noise causing jumps up)
        assert np.any(differences > 0)
        # But overall trend should be downward
        assert trace[0] > trace[-1]


def test_multi_qubit_t1_collection_names(simulation_setup, dut_qubits):
    """Test MultiQubitT1 with different collection names."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    # Run with single collection name (should apply to all qubits)
    t1_exp = MultiQubitT1(
        duts=dut_qubits,
        collection_names='f01',
        time_length=50.0,
        time_resolution=1.0
    )
    
    assert t1_exp.collection_names == ['f01', 'f01']
    
    # Run with different collection names
    t1_exp2 = MultiQubitT1(
        duts=dut_qubits,
        collection_names=['f01', 'f01'],
        time_length=50.0,
        time_resolution=1.0
    )
    
    assert t1_exp2.collection_names == ['f01', 'f01']


def test_multi_qubit_t1_time_parameters(simulation_setup, dut_qubits):
    """Test MultiQubitT1 with different time parameters."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    # Test with fine time resolution
    t1_fine = MultiQubitT1(
        duts=dut_qubits,
        time_length=50.0,
        time_resolution=0.5
    )
    
    assert len(t1_fine.traces[0]) == 100  # 50.0 / 0.5
    
    # Test with coarse time resolution
    t1_coarse = MultiQubitT1(
        duts=dut_qubits,
        time_length=50.0,
        time_resolution=5.0
    )
    
    assert len(t1_coarse.traces[0]) == 10  # 50.0 / 5.0
    
    # Both should show decay
    assert t1_fine.traces[0][-1] < 0.5
    assert t1_coarse.traces[0][-1] < 0.5


def test_multi_qubit_t1_single_qubit(simulation_setup):
    """Test MultiQubitT1 with just one qubit."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    # Create single qubit
    qubit = TransmonElement(
        name='single_qubit',
        parameters=create_transmon_config(2, 5000.0)
    )
    
    # Run with single qubit
    t1_exp = MultiQubitT1(
        duts=[qubit],
        time_length=100.0,
        time_resolution=2.0
    )
    
    assert len(t1_exp.traces) == 1
    assert len(t1_exp.traces[0]) == 50
    
    # Should still show T1 decay
    assert t1_exp.traces[0][0] > 0.9
    assert t1_exp.traces[0][-1] < 0.5


def test_multi_qubit_t1_plot_methods(simulation_setup, dut_qubits):
    """Test that MultiQubitT1 has plotting methods."""
    manager = ExperimentManager().get_default_setup()
    manager.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    t1_exp = MultiQubitT1(
        duts=dut_qubits,
        time_length=100.0,
        time_resolution=2.0
    )
    
    # Check plot methods exist
    assert hasattr(t1_exp, 'plot_all')
    assert hasattr(t1_exp, 'plot_t1')
    
    # Test plotting individual qubit
    fig = t1_exp.plot_t1(0, fit=True)
    assert fig is not None
    
    fig2 = t1_exp.plot_t1(1, fit=False)
    assert fig2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])