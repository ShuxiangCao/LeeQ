"""
Test fixtures for simplified EPII parameter management.
"""

from unittest.mock import Mock
import numpy as np


def get_test_setup():
    """Create a mock LeeQ setup with realistic parameter structure."""
    setup = Mock()

    # Mock status parameters with _internal_dict (following LeeQ pattern)
    status = Mock()
    status._internal_dict = {
        "shot_number": 2000,
        "shot_period": 500.0,
        "acquisition_type": "IQ",
        "debug_plotter": False,
        "measurement_basis": "<z>",
        "test_float": 3.14159,
        "test_bool": True,
        "test_string": "test_value"
    }

    # Mock get_parameters method to return dict copy
    status.get_parameters = lambda: status._internal_dict.copy()

    # Mock set_parameter method to update dict
    def set_param(key, value):
        status._internal_dict[key] = value
        return True
    status.set_parameter = Mock(side_effect=set_param)

    setup.status = status
    setup._name = "test_setup"
    setup._active = True

    # Mock elements with nested _parameters dict
    q0 = Mock()
    q0._parameters = {
        "f01": 5.0e9,
        "anharmonicity": -0.33e9,
        "characterizations.SimpleT1": 20e-6,
        "t2": 15e-6,
        "pi_amp": 0.5,
        "pi_len": 40,
        "nested": {
            "level1": {
                "level2": "deep_value"
            }
        }
    }

    q1 = Mock()
    q1._parameters = {
        "f01": 5.1e9,
        "anharmonicity": -0.32e9,
        "characterizations.SimpleT1": 25e-6,
        "t2": 18e-6,
        "pi_amp": 0.45,
        "pi_len": 38
    }

    # Add other element types
    res0 = Mock()
    res0._parameters = {
        "frequency": 7.0e9,
        "kappa": 1e6,
        "readout_amp": 0.1,
        "readout_len": 2000
    }

    setup._elements = {"q0": q0, "q1": q1, "res0": res0}

    # Mock qubits list (for indexed access)
    setup.qubits = [q0, q1]

    return setup


def get_test_parameters_flat():
    """Get expected flattened parameter dictionary."""
    return {
        # Status parameters
        "status.shot_number": 2000,
        "status.shot_period": 500.0,
        "status.acquisition_type": "IQ",
        "status.debug_plotter": False,
        "status.measurement_basis": "<z>",
        "status.test_float": 3.14159,
        "status.test_bool": True,
        "status.test_string": "test_value",

        # Q0 parameters
        "q0.f01": 5.0e9,
        "q0.anharmonicity": -0.33e9,
        "q0.t1": 20e-6,
        "q0.t2": 15e-6,
        "q0.pi_amp": 0.5,
        "q0.pi_len": 40,
        "q0.nested.level1.level2": "deep_value",

        # Q1 parameters
        "q1.f01": 5.1e9,
        "q1.anharmonicity": -0.32e9,
        "q1.t1": 25e-6,
        "q1.t2": 18e-6,
        "q1.pi_amp": 0.45,
        "q1.pi_len": 38,

        # Resonator parameters
        "res0.frequency": 7.0e9,
        "res0.kappa": 1e6,
        "res0.readout_amp": 0.1,
        "res0.readout_len": 2000
    }


def get_test_types():
    """Get test values for different supported types."""
    return {
        "int_value": 42,
        "float_value": 3.14159,
        "bool_true": True,
        "bool_false": False,
        "string_value": "test_string",
        "none_value": None,
        "numpy_array": np.array([1.0, 2.0, 3.0]),
        "list_value": [1, 2, 3],
        "dict_value": {"key": "value"}
    }


def get_empty_setup():
    """Create a minimal setup with no elements."""
    setup = Mock()

    # Minimal status
    status = Mock()
    status._internal_dict = {"shot_number": 1000}
    status.get_parameters = Mock(return_value=status._internal_dict.copy())
    status.set_parameter = Mock(side_effect=lambda k, v: status._internal_dict.update({k: v}))

    setup.status = status
    setup._elements = {}
    setup.qubits = []

    return setup


# Export for easy testing
__all__ = [
    'get_test_setup',
    'get_test_parameters_flat',
    'get_test_types',
    'get_empty_setup'
]
