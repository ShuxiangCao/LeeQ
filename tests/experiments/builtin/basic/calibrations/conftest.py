"""
Pytest fixtures for basic calibration experiments testing.

Provides common test fixtures for calibration experiment testing including:
- Standard simulation setup with VirtualTransmon
- ExperimentManager configuration
- Chronicle logging initialization
"""

import pytest
import numpy as np

from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.experiments.experiments import ExperimentManager
from leeq.chronicle import Chronicle


@pytest.fixture
def simulation_setup():
    """Standard simulation setup for experiment tests."""
    # Start chronicle logging
    Chronicle().start_log()
    
    # Clear any existing setups
    manager = ExperimentManager()
    manager.clear_setups()
    
    # Create virtual transmon with standard parameters
    virtual_transmon = VirtualTransmon(
        name="test_qubit",
        qubit_frequency=5000.0,
        anharmonicity=-200.0,
        t1=50.0,  # 50 μs T1
        t2=30.0,  # 30 μs T2
        readout_frequency=6000.0,
        readout_linewith=5.0,
        readout_dipsersive_shift=2.0,
        quiescent_state_distribution=np.asarray([0.9, 0.08, 0.02])
    )
    
    # Create and register setup
    setup = HighLevelSimulationSetup(
        name='HighLevelSimulationSetup',
        virtual_qubits={2: virtual_transmon}
    )
    manager.register_setup(setup)
    
    # Disable plotting for tests
    default_setup = manager.get_default_setup()
    default_setup.status.set_parameter("Plot_Result_In_Jupyter", False)
    
    yield manager
    
    # Cleanup
    manager.clear_setups()