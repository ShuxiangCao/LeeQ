import numpy as np
from pytest import fixture
from leeq.core.elements.built_in.qudit_transmon import TransmonElement


@fixture
def transmon_element():
    drive_configure_f01 = {
        'type': 'SimpleDriveCollection',
        'freq': 4144.12,
        'channel': 2,
        'shape': 'BlackmanDRAG',
        'amp': 0.223,
        'phase': 0.,
        'width': 0.025,
        'alpha': 425.13,
        'trunc': 1.2
    }
    drive_configure_f12 = {
        'type': 'SimpleDriveCollection',
        'freq': 3980.12,
        'channel': 2,
        'shape': 'BlackmanDRAG',
        'amp': 0.2,
        'phase': 0.,
        'width': 0.025,
        'alpha': 425.13,
        'trunc': 1.2
    }

    qubit = TransmonElement(
        name='test_element',
        parameters={
            'lpb_collections': {
                'f01': drive_configure_f01,
                'f12': drive_configure_f12},
            'measurement_primitives': {}})

    return qubit


def test_element_creation(transmon_element):
    qubit = transmon_element

    assert isinstance(qubit, TransmonElement)
    assert qubit._name == 'test_element'
    c1 = qubit.get_c1('f01')
    assert c1['X'].freq == 4144.12

    lpb = (c1['X'] + c1['Y']) * c1['Xp']

    assert lpb is not None


def test_get_qudit_gates(transmon_element):
    qubit = transmon_element

    X_gate = qubit.get_gate('X', 'f01')
    assert X_gate.freq == 4144.12
    assert X_gate.amp == 0.223
    assert X_gate.phase == 0.
    assert X_gate.width == 0.025
    assert X_gate.alpha == 425.13
    assert X_gate.trunc == 1.2
    assert X_gate.channel == 2

    Y_gate = qubit.get_gate('Y', 'f01')
    assert Y_gate.freq == 4144.12
    assert Y_gate.amp == 0.223
    assert Y_gate.phase == np.pi / 2
    assert Y_gate.width == 0.025
    assert Y_gate.alpha == 425.13
    assert Y_gate.trunc == 1.2
    assert Y_gate.channel == 2

    Xp_gate = qubit.get_gate('Xp', 'f01')
    assert Xp_gate.freq == 4144.12
    assert Xp_gate.amp == 0.223 / 2
    assert Xp_gate.phase == 0.
    assert Xp_gate.width == 0.025
    assert Xp_gate.alpha == 425.13
    assert Xp_gate.trunc == 1.2
    assert Xp_gate.channel == 2

    Yp_gate = qubit.get_gate('Yp', 'f01')
    assert Yp_gate.freq == 4144.12
    assert Yp_gate.amp == 0.223 / 2
    assert Yp_gate.phase == np.pi / 2
    assert Yp_gate.width == 0.025
    assert Yp_gate.alpha == 425.13
    assert Yp_gate.trunc == 1.2
    assert Yp_gate.channel == 2


def test_create_qutrit_qudit_gates(transmon_element):
    # qutrit_clifford_green
    lpb = transmon_element.get_gate('qutrit_clifford_green', angle=(1, 2))
    assert lpb is not None

    # qutrit_clifford_red
    lpb = transmon_element.get_gate('qutrit_clifford_red', angle=(2, 1))
    assert lpb is not None

    # qutrit_hadamard
    lpb = transmon_element.get_gate('qutrit_hadamard')
    assert lpb is not None

    # qutrit_hadamard_dag
    lpb = transmon_element.get_gate('qutrit_hadamard_dag')
    assert lpb is not None
