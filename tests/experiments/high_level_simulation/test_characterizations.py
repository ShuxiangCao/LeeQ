import numpy as np
import pytest

from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon


@pytest.fixture()
def simulation_setup():
    from labchronicle import Chronicle
    Chronicle().start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    virtual_transmon = VirtualTransmon(
        name="VQubit",
        qubit_frequency=5040.4,
        anharmonicity=-198,
        t1=70,
        t2=35,
        readout_frequency=9645.5,
        quiescent_state_distribution=np.asarray(
            [
                0.8,
                0.15,
                0.04,
                0.01]))

    setup = HighLevelSimulationSetup(
        name='HighLevelSimulationSetup',
        virtual_qubits={2: virtual_transmon}
    )
    manager.register_setup(setup)
    return manager


configuration = {
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 4144.417053428905,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 0.025,
            'alpha': 425.1365229849309,
            'trunc': 1.2
        },
        'f12': {
            'type': 'SimpleDriveCollection',
            'freq': 4144.417053428905,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 0.025,
            'alpha': 425.1365229849309,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9144.41,
            'channel': 1,
            'shape': 'square',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        },
        '1': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9144.41,
            'channel': 1,
            'shape': 'square',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1, 2]
        }
    }
}


@pytest.fixture
def qubit():
    dut = TransmonElement(
        name='test_qubit',
        parameters=configuration
    )

    return dut


def test_t1(simulation_setup, qubit):
    from leeq.experiments.builtin import SimpleT1
    manager = ExperimentManager().get_default_setup(
    ).status.set_parameter("Plot_Result_In_Jupyter", False)
    sweep = SimpleT1(
        qubit=qubit,
    )

def test_t2_echo(simulation_setup, qubit):
    from leeq.experiments.builtin import SpinEchoMultiLevel
    manager = ExperimentManager().get_default_setup(
    ).status.set_parameter("Plot_Result_In_Jupyter", False)
    sweep = SpinEchoMultiLevel(
        qubit=qubit,
    )
