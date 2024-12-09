from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
import numpy as np
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq import ExperimentManager

manager = ExperimentManager()
manager.clear_setups()

def get_virtual_qubit_pair():
    virtual_transmon_1 = VirtualTransmon(
        name="VQubit_1",
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

    virtual_transmon_2 = VirtualTransmon(
        name="VQubit_2",
        qubit_frequency=5040. - 123,
        anharmonicity=-197,
        t1=70,
        t2=35,
        readout_frequency=9645.5 + 100,
        quiescent_state_distribution=np.asarray(
            [
                0.8,
                0.15,
                0.04,
                0.01]))

    setup = HighLevelSimulationSetup(
        name='HighLevelSimulationSetup',
        virtual_qubits={2: virtual_transmon_1,
                        4: virtual_transmon_2
                        }
    )
    setup.set_coupling_strength_by_qubit(
        virtual_transmon_1, virtual_transmon_2, coupling_strength=1.5)

    manager.register_setup(setup)

    configuration_q1 = {
        'lpb_collections': {
            'f01': {
                'type': 'SimpleDriveCollection',
                'freq': 5040.4,
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
                'freq': 5040.4 - 198,
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
                'freq': 9645.5,
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

    configuration_q2 = {
        'lpb_collections': {
            'f01': {
                'type': 'SimpleDriveCollection',
                'freq': 5040. - 123,
                'channel': 4,
                'shape': 'blackman_drag',
                'amp': 0.21323904814245054 / 5 * 4,
                'phase': 0.,
                'width': 0.025,
                'alpha': 425.1365229849309,
                'trunc': 1.2
            },
            'f12': {
                'type': 'SimpleDriveCollection',
                'freq': 5040.4 - 198 - 123,
                'channel': 4,
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
                'freq': 9645.5 + 100,
                'channel': 3,
                'shape': 'square',
                'amp': 0.21323904814245054 / 5 * 4,
                'phase': 0.,
                'width': 1,
                'trunc': 1.2,
                'distinguishable_states': [0, 1]
            },
            '1': {
                'type': 'SimpleDispersiveMeasurement',
                'freq': 9645.5 + 100,
                'channel': 3,
                'shape': 'square',
                'amp': 0.21323904814245054 / 5 * 4,
                'phase': 0.,
                'width': 1,
                'trunc': 1.2,
                'distinguishable_states': [0, 1, 2]
            }
        }
    }

    qubit_1 = TransmonElement(
            name='test-qubit-1',
            parameters=configuration_q1
        )

    qubit_2 = TransmonElement(
            name='test-qubit-2',
            parameters=configuration_q2
        )
    return qubit_1, qubit_2