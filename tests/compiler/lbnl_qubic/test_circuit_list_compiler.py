import pytest
from pytest import fixture
from leeq.compiler.lbnl_qubic.circuit_list_compiler import QubiCCircuitListLPBCompiler
from leeq.core.context import ExperimentContext
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.core.primitives.built_in.common import Delay

configuration_1 = {
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 4144.417053428905,
            'channel': 0,
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
            'channel': 0,
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

configuration_2 = {
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
            'channel': 3,
            'shape': 'square',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        }
    }
}

leeq_channel_to_qubic_channel = {0: 'Q0', 1: 'Q0', 2: 'Q1', 3: 'Q1'}


@fixture
def qubit_1():
    dut = TransmonElement(
        name='test_qubit1',
        parameters=configuration_1
    )

    return dut


@fixture
def qubit_2():
    dut = TransmonElement(
        name='test_qubit2',
        parameters=configuration_2
    )
    return dut


def test_initialization():
    mapping = {"LeeQ1": "QubiC1"}
    compiler = QubiCCircuitListLPBCompiler(leeq_channel_to_qubic_channel=mapping)
    assert compiler._leeq_channel_to_qubic_channel == mapping


def compile_lpb(lpb):
    # Mocking the necessary classes and methods
    context = ExperimentContext(name='test_context_name')
    compiler = QubiCCircuitListLPBCompiler(leeq_channel_to_qubic_channel=leeq_channel_to_qubic_channel)

    # You might have more specific behavior to test here, like specific method calls or return values
    result_context = compiler.compile_lpb(context, lpb)

    return result_context.instructions


def test_atomic_lpb_compilation(qubit_1):
    atomic_lpbs = [
        qubit_1.get_lpb_collection('f01')['Xp'],
        qubit_1.get_lpb_collection('f01')['X'],
        qubit_1.get_lpb_collection('f01').x(0.2),
        qubit_1.get_lpb_collection('f01').y(0.6),
    ]

    for lpb in atomic_lpbs:
        instructions = compile_lpb(lpb)
        circuits = instructions['circuits']
        assert len(circuits) == 1
        assert circuits[0]['dest'] == leeq_channel_to_qubic_channel[lpb.channel] + '.qdrv'

    atomic_lpbs = [
        qubit_1.get_lpb_collection('f12').z(0.1),
    ]

    for lpb in atomic_lpbs:
        instructions = compile_lpb(lpb)
        circuits = instructions['circuits']
        assert len(circuits) == 0

    atomic_lpbs = [
        qubit_1.get_measurement_primitive('0'),
    ]

    for lpb in atomic_lpbs:
        instructions = compile_lpb(lpb)
        circuits = instructions['circuits']
        assert len(circuits) == 3
        assert circuits[0]['name'] == 'pulse'
        assert circuits[0]['dest'] == leeq_channel_to_qubic_channel[lpb.channel] + '.rdrv'
        assert circuits[1]['name'] == 'delay'
        assert circuits[2]['name'] == 'pulse'
        assert circuits[2]['dest'] == leeq_channel_to_qubic_channel[lpb.channel] + '.rdlo'
        assert len(instructions['qubic_channel_to_lpb_uuid']) == 1


def test_block_lpbs(qubit_1, qubit_2):
    serial_lpbs = [
        qubit_1.get_gate('qutrit_hadamard'),
        qubit_2.get_lpb_collection('f01')['Xp'] + qubit_2.get_lpb_collection('f12')['Xp'],
        qubit_2.get_lpb_collection('f01')['Xp'] + qubit_2.get_lpb_collection('f12')[
            'Xp'] + qubit_2.get_measurement_primitive('0'),
    ]

    parallel_lpbs = [
        qubit_1.get_gate('qutrit_hadamard') * qubit_2.get_gate('qutrit_hadamard'),
        (qubit_1.get_gate('qutrit_hadamard') * qubit_2.get_gate('qutrit_hadamard')) + qubit_2.get_measurement_primitive(
            '0'),
        (qubit_1.get_gate('qutrit_hadamard') * qubit_2.get_gate('qutrit_hadamard')) + qubit_2.get_measurement_primitive(
            '0') + (qubit_1.get_gate('qutrit_hadamard') * qubit_2.get_gate('qutrit_hadamard')),
    ]

    for lpb in serial_lpbs:
        instructions = compile_lpb(lpb)

    for lpb in parallel_lpbs:
        instructions = compile_lpb(lpb)

    with pytest.raises(AssertionError):
        lpb = qubit_1.get_gate('qutrit_hadamard') * qubit_1.get_gate('qutrit_hadamard')
        instructions = compile_lpb(lpb)


def test_measurement_like_pulse_experiment_circuit(qubit_1, qubit_2):
    rotate_X_qubit = qubit_1.get_lpb_collection('f01')['Xp']
    mlp = qubit_2.get_measurement_primitive('0')
    delay = Delay(0.5)
    phi_rotate_X_qubit = rotate_X_qubit.clone()
    measurement_primitive_qubit = qubit_1.get_measurement_prim_intlist('0')

    lpb = rotate_X_qubit + mlp + delay + phi_rotate_X_qubit + measurement_primitive_qubit

    instructions = compile_lpb(lpb)

    for i in [1, 5, 6, 8]:
        assert set(instructions['circuits'][i]['scope']) == {'Q1', 'Q0'}

    assert set(instructions['circuits'][3]['scope']) == {'Q1'}
    assert set(instructions['circuits'][10]['scope']) == {'Q0'}