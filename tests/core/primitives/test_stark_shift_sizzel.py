import pytest
from pytest import fixture
import numpy as np
from leeq.core.primitives.built_in.sizzel_gate import SiZZelTwoQubitGateCollection

from leeq.core.elements.built_in.qudit_transmon import TransmonElement

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
        }
    }
}


@fixture
def qubit_control():
    dut = TransmonElement(
        name='test_qubit_control',
        parameters=configuration
    )

    return dut


@fixture
def qubit_target():
    dut = TransmonElement(
        name='test_qubit_target',
        parameters=configuration
    )

    return dut


@fixture
def sizzel_gate_collection(qubit_control, qubit_target):
    # Creating an instance of SiZZelTwoQubitGateCollection with mocked
    # TransmonElements
    parameters = {
        'freq': 5123.0,
        'amp_control': 0.5,
        'amp_target': 0.6,
        'iz_control': 0.1,
        'iz_target': 0.2,
        'phase_diff': np.pi / 4,
        'echo': True,
    }
    return SiZZelTwoQubitGateCollection(
        "test_collection",
        parameters,
        qubit_control,
        qubit_target)


def test_initialization(sizzel_gate_collection):
    # Testing that the instance is initialized correctly with merged parameters
    assert sizzel_gate_collection._parameters['freq'] == 5123.0
    assert sizzel_gate_collection._parameters['echo'] is True


def test_validate_parameters(sizzel_gate_collection):
    # Testing that validate_parameters does not raise an assertion error for
    # valid parameters
    try:
        sizzel_gate_collection.validate_parameters()
    except AssertionError:
        pytest.fail(
            "validate_parameters() raised an AssertionError unexpectedly!")


def test_validate_parameters_missing_key(sizzel_gate_collection):
    # Testing that validate_parameters raises an assertion error for missing
    # required parameters
    del sizzel_gate_collection._parameters['freq']
    with pytest.raises(AssertionError):
        sizzel_gate_collection.validate_parameters()


def test_get_lpbs(sizzel_gate_collection):
    sizzel_gate_collection['stark_drive_control']
    sizzel_gate_collection['stark_drive_target']

    sizzel_gate_collection.get_z_cancellation_pulse()
    sizzel_gate_collection.get_cz()
    sizzel_gate_collection.get_zzm()
    sizzel_gate_collection.get_zzp()
    sizzel_gate_collection.get_zxm()
    sizzel_gate_collection.get_zxp()
    sizzel_gate_collection.get_cnot_like()
    sizzel_gate_collection.get_cnot()
    sizzel_gate_collection.get_swap_like()
    sizzel_gate_collection.get_iswap_like()
