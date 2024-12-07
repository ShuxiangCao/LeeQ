# This file setup the high-level simulation and provides a 2Q virtual device.
import numpy as np

from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon

def simulation_setup(qubit_frequency=5040.4,readout_frequency=9645.4,quiescent_state_distribution=None):
    from labchronicle import Chronicle
    Chronicle(config={'handler':'memory'}).start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    # print('readout_frequency',readout_frequency)

    if quiescent_state_distribution is None:
        quiescent_state_distribution=np.asarray(
                [
                    0.8,
                    0.15,
                    0.04,
                    0.01])
        
    virtual_transmon_a = VirtualTransmon(
        name="VQubitA",
        qubit_frequency=qubit_frequency,
        anharmonicity=-198,
        t1=70,
        t2=35,
        readout_frequency=readout_frequency,quiescent_state_distribution=quiescent_state_distribution)
    

    virtual_transmon_b = VirtualTransmon(
        name="VQubitB",
        qubit_frequency=4855.3,
        anharmonicity=-197,
        t1=60,
        t2=30,
        readout_frequency=readout_frequency,
        quiescent_state_distribution=quiescent_state_distribution)

    setup = HighLevelSimulationSetup(
        name='HighLevelSimulationSetup',
        virtual_qubits={2: virtual_transmon_a,
                        4: virtual_transmon_b}
    )
    manager.register_setup(setup)
    return manager


configuration_a = {
    'hrid':'QA',
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 5040.4,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.1 ,
            'phase': 0.,
            'width': 0.05,
            'alpha': 500,
            'trunc': 1.2
        },
        'f12': {
            'type': 'SimpleDriveCollection',
            'freq': 5040.4-198,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.1 / np.sqrt(2),
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
            'amp': 0.15,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        }
    }
}

configuration_b = {
    'hrid':'QB',
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 4855.3,
            'channel': 4,
            'shape': 'blackman_drag',
            'amp': 0.1 ,
            'phase': 0.,
            'width': 0.05,
            'alpha': 500,
            'trunc': 1.2
        },
        'f12': {
            'type': 'SimpleDriveCollection',
            'freq': 5040.4-197,
            'channel': 4,
            'shape': 'blackman_drag',
            'amp': 0.1 / np.sqrt(2),
            'phase': 0.,
            'width': 0.025,
            'alpha': 425.1365229849309,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9025.5,
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

from experiments import *
