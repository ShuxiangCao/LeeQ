"""
Standardized simulation configurations for LeeQ notebooks.

This module provides common setup patterns used across all tutorial, example, 
and workflow notebooks, based on the established patterns from simulated_setup.py.
"""

import numpy as np
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.chronicle import Chronicle


def get_standard_setup():
    """
    Initialize standard two-qubit simulation setup.
    
    Returns:
        ExperimentManager: Configured manager with virtual transmons
    """
    Chronicle().start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    # Virtual transmon A - higher frequency qubit
    virtual_transmon_a = VirtualTransmon(
        name="VQubitA",
        qubit_frequency=5040.4,
        anharmonicity=-198,
        t1=70,
        t2=35,
        readout_frequency=9645.4,
        quiescent_state_distribution=np.asarray([0.8, 0.15, 0.04, 0.01])
    )

    # Virtual transmon B - lower frequency qubit  
    virtual_transmon_b = VirtualTransmon(
        name="VQubitB",
        qubit_frequency=4855.3,
        anharmonicity=-197,
        t1=60,
        t2=30,
        readout_frequency=9025.1,
        quiescent_state_distribution=np.asarray([0.75, 0.18, 0.05, 0.02])
    )

    setup = HighLevelSimulationSetup(
        name='HighLevelSimulationSetup',
        virtual_qubits={2: virtual_transmon_a, 4: virtual_transmon_b}
    )

    # Set coupling strength for two-qubit operations
    setup.set_coupling_strength_by_qubit(
        virtual_transmon_a, virtual_transmon_b, coupling_strength=1.5
    )

    manager.register_setup(setup)
    return manager


def get_single_qubit_setup():
    """
    Initialize single-qubit simulation setup for basic tutorials.
    
    Returns:
        ExperimentManager: Configured manager with single virtual transmon
    """
    Chronicle().start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    virtual_transmon = VirtualTransmon(
        name="VQubit",
        qubit_frequency=5040.4,
        anharmonicity=-198,
        t1=70,
        t2=35,
        readout_frequency=9645.4,
        quiescent_state_distribution=np.asarray([0.8, 0.15, 0.04, 0.01])
    )

    setup = HighLevelSimulationSetup(
        name='SingleQubitSimulation',
        virtual_qubits={2: virtual_transmon}
    )

    manager.register_setup(setup)
    return manager


def get_standard_qubit_config():
    """
    Get standard qubit configuration dictionary for Qubit A.
    
    Returns:
        dict: Standard configuration parameters
    """
    return {
        'hrid': 'QA',
        'lpb_collections': {
            'f01': {
                'type': 'SimpleDriveCollection',
                'freq': 5040.4,
                'channel': 2,
                'shape': 'blackman_drag',
                'amp': 0.5487,
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


def get_second_qubit_config():
    """
    Get standard qubit configuration dictionary for Qubit B.
    
    Returns:
        dict: Standard configuration parameters
    """
    return {
        'hrid': 'QB',
        'lpb_collections': {
            'f01': {
                'type': 'SimpleDriveCollection',
                'freq': 4855.3,
                'channel': 4,
                'shape': 'blackman_drag',
                'amp': 0.5399696605966315,
                'phase': 0.,
                'width': 0.05,
                'alpha': 500,
                'trunc': 1.2
            },
            'f12': {
                'type': 'SimpleDriveCollection',
                'freq': 4855.3-197,
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


def setup_duts(single_qubit=False):
    """
    Setup standard device-under-test (DUT) configurations.
    
    Args:
        single_qubit (bool): If True, create only single qubit setup
        
    Returns:
        dict: Dictionary of configured TransmonElement objects
    """
    # Configure standard measurement shots
    from leeq.experiments.experiments import setup
    setup().status().set_param("Shot_Number", 500)
    setup().status().set_param("Shot_Period", 500)
    
    if single_qubit:
        dut_dict = {
            'Q1': {'Active': True, 'Tuneup': False, 'FromLog': False, 
                   'Params': get_standard_qubit_config()}
        }
    else:
        dut_dict = {
            'Q1': {'Active': True, 'Tuneup': False, 'FromLog': False, 
                   'Params': get_standard_qubit_config()},
            'Q2': {'Active': True, 'Tuneup': False, 'FromLog': False, 
                   'Params': get_second_qubit_config()}
        }
    
    duts_dict = {}
    for hrid, dd in dut_dict.items():
        if dd['Active']:
            if dd['FromLog']:
                dut = TransmonElement.load_from_calibration_log(dd['Params']['hrid'])
            else:
                dut = TransmonElement(name=dd['Params']['hrid'], parameters=dd['Params'])
            
            if not dd['Tuneup']:
                # Setup basic measurement calibration
                lpb_scan = (dut.get_c1('f01')['I'], dut.get_c1('f01')['X'])
                from leeq.experiments.builtin.basic.calibrations import MeasurementCalibrationMultilevelGMM
                calib = MeasurementCalibrationMultilevelGMM(
                    dut, mprim_index=0, sweep_lpb_list=lpb_scan
                )
            
            dut.print_config_info()
            duts_dict[hrid] = dut
    
    return duts_dict


def initialize_notebook_environment(single_qubit=False):
    """
    Complete initialization for notebook environment.
    
    Args:
        single_qubit (bool): If True, setup single qubit only
        
    Returns:
        tuple: (manager, duts_dict) - experiment manager and configured DUTs
    """
    # Setup simulation environment
    if single_qubit:
        manager = get_single_qubit_setup()
    else:
        manager = get_standard_setup()
    
    # Setup DUT configurations
    duts_dict = setup_duts(single_qubit=single_qubit)
    
    return manager, duts_dict