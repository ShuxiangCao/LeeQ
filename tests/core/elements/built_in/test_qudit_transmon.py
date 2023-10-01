import pytest
from leeq.core.elements.built_in.qudit_transmon import TransmonElement


def test_element_creation():
    drive_configure = {
        'type': 'SimpleDriveCollection',
        'freq': 4144.417053428905,
        'channel': 2,
        'shape': 'BlackmanDRAG',
        'amp': 0.21323904814245054,
        'phase': 0.,
        'width': 0.025,
        'alpha': 425.1365229849309,
        'trunc': 1.2
    }

    qubit = TransmonElement(name='test_element', parameters={'lpb_collections': {
        'f01': drive_configure
    }, 'measurement_primitives': {}})
    assert isinstance(qubit, TransmonElement)
    assert qubit._name == 'test_element'
    # assert element._parameters == {'lpb_collections': [], 'measurement_primitives': []}

    # assert element._lpb_collections == {}
    c1 = qubit.get_c1('f01')
    assert c1['X'].freq == 4144.417053428905

    lpb = (c1['X'] + c1['Y']) * c1['Xp']

    assert lpb is not None