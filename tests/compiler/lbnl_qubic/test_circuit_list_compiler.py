import pytest
from pytest import fixture
from leeq.compiler.lbnl_qubic.circuit_list_compiler import QubiCCircuitListLPBCompiler, segment_array
from leeq.core.context import ExperimentContext
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.core.primitives.built_in.common import Delay
from leeq.experiments.experiments import ExperimentManager
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
import numpy as np

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
        },
        'f23': {
            'type': 'SimpleDriveCollection',
            'freq': 4104.417053428905,
            'channel': 0,
            'shape': 'soft_square',
            'amp': 0.9,
            'phase': 0.,
            'rise': 0.1,
            'width': 1,
            'trunc': 1
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
def test_setup():
    """Create a test setup using the high level simulation setup."""
    from leeq.chronicle import Chronicle
    Chronicle().start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    # Create virtual transmons for the channels used in tests
    virtual_transmon_0 = VirtualTransmon(
        name="VQubit0",
        qubit_frequency=4144.417053428905,
        anharmonicity=-425,
        t1=70,
        t2=35,
        readout_frequency=9144.41,
        quiescent_state_distribution=np.asarray([0.8, 0.15, 0.04, 0.01]))

    virtual_transmon_2 = VirtualTransmon(
        name="VQubit2",
        qubit_frequency=4144.417053428905,
        anharmonicity=-425,
        t1=70,
        t2=35,
        readout_frequency=9144.41,
        quiescent_state_distribution=np.asarray([0.8, 0.15, 0.04, 0.01]))

    setup = HighLevelSimulationSetup(
        name='TestSimulationSetup',
        virtual_qubits={0: virtual_transmon_0, 2: virtual_transmon_2}
    )

    # Add all channels used in test configurations: 0, 1, 2, 3
    # Channels 0 and 2 are drive channels (handled by virtual_qubits)
    # Channels 1 and 3 are measurement channels (need to be added manually)
    setup.status.add_channel(0)  # Drive channel for qubit 1
    setup.status.add_channel(1)  # Measurement channel for qubit 1
    setup.status.add_channel(2)  # Drive channel for qubit 2
    setup.status.add_channel(3)  # Measurement channel for qubit 2

    manager.register_setup(setup, set_as_default=True)

    yield setup

    # Cleanup
    manager.clear_setups()


@fixture
def qubit_1(test_setup):
    dut = TransmonElement(
        name='test_qubit1',
        parameters=configuration_1
    )

    return dut


@fixture
def qubit_2(test_setup):
    dut = TransmonElement(
        name='test_qubit2',
        parameters=configuration_2
    )
    return dut


def test_initialization():
    mapping = {"LeeQ1": "QubiC1"}
    compiler = QubiCCircuitListLPBCompiler(
        leeq_channel_to_qubic_channel=mapping)
    assert compiler._leeq_channel_to_qubic_channel == mapping


def compile_lpb(lpb):
    # Mocking the necessary classes and methods
    context = ExperimentContext(name='test_context_name')
    compiler = QubiCCircuitListLPBCompiler(
        leeq_channel_to_qubic_channel=leeq_channel_to_qubic_channel)

    # You might have more specific behavior to test here, like specific method
    # calls or return values
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
        qubit_2.get_lpb_collection('f01')['Xp'] +
        qubit_2.get_lpb_collection('f12')['Xp'],
        qubit_2.get_lpb_collection('f01')['Xp'] +
        qubit_2.get_lpb_collection('f12')['Xp'] +
        qubit_2.get_measurement_primitive('0'),
    ]

    parallel_lpbs = [
        qubit_1.get_gate('qutrit_hadamard') *
        qubit_2.get_gate('qutrit_hadamard'),
        (qubit_1.get_gate('qutrit_hadamard') *
         qubit_2.get_gate('qutrit_hadamard')) +
        qubit_2.get_measurement_primitive('0'),
        (qubit_1.get_gate('qutrit_hadamard') *
         qubit_2.get_gate('qutrit_hadamard')) +
        qubit_2.get_measurement_primitive('0') +
        (
                qubit_1.get_gate('qutrit_hadamard') *
                qubit_2.get_gate('qutrit_hadamard')),
    ]

    for lpb in serial_lpbs:
        compile_lpb(lpb)

    for lpb in parallel_lpbs:
        compile_lpb(lpb)

    with pytest.raises(AssertionError):
        lpb = qubit_1.get_gate('qutrit_hadamard') * \
              qubit_1.get_gate('qutrit_hadamard')
        compile_lpb(lpb)


def test_measurement_like_pulse_experiment_circuit(qubit_1, qubit_2):
    rotate_X_qubit = qubit_1.get_lpb_collection('f01')['Xp']
    mlp = qubit_2.get_measurement_primitive('0')
    delay = Delay(0.5)
    phi_rotate_X_qubit = rotate_X_qubit.clone()
    measurement_primitive_qubit = qubit_1.get_measurement_prim_intlist('0')

    lpb = rotate_X_qubit + mlp + delay + \
          phi_rotate_X_qubit + measurement_primitive_qubit

    instructions = compile_lpb(lpb)

    for i in [1, 5, 6, 8]:
        assert set(instructions['circuits'][i]['scope']) == {'Q1', 'Q0'}

    assert set(instructions['circuits'][3]['scope']) == {'Q1'}
    assert set(instructions['circuits'][10]['scope']) == {'Q0'}


def test_automated_segmentation_of_very_long_pulses(qubit_1):
    rotate_X_qubit = qubit_1.get_lpb_collection('f23')['X']
    mlp = qubit_1.get_measurement_primitive('0')

    rotate_X_qubit.update_parameters(width=1.2)

    lpb = rotate_X_qubit + mlp

    instructions = compile_lpb(lpb)

    total_width = 0
    for i, circuit in enumerate(instructions['circuits']):
        assert circuit['dest'] == 'Q0.qdrv'
        total_width += circuit['twidth']
        if i == 0:
            assert circuit['env']['env_func'] == 'leeq_segment_by_index'
        else:
            if circuit['env'] == 'cw':
                continue
            if circuit['env']['env_func'] == 'leeq_segment_by_index':
                break

    assert np.abs(total_width * 1e6 - 1.4) < 1e-10  # 1.2 + 0.2, include the rise time


def generate_flat_sequence(length, value):
    """Generate a sequence of flat complex values."""
    return np.full(length, value, dtype=complex)


def generate_changing_sequence(length, start_value, end_value):
    """Generate a sequence of linearly changing complex values."""
    real_part = np.linspace(start_value.real, end_value.real, length)
    imag_part = np.linspace(start_value.imag, end_value.imag, length)
    return real_part + 1j * imag_part


# Segmentation tests

def _test_segmentation(sequences, expected_flat, expected_change):
    # Threshold for flat regions
    threshold = 0.001
    min_flat_length = 5  # Minimum length for flat regions

    test_data = np.concatenate(sequences)

    # Segment the data
    flat_regions, change_regions = segment_array(test_data, threshold, min_flat_length)

    # Assert results
    assert flat_regions == expected_flat, f"Expected flat regions {expected_flat}, got {flat_regions}"
    assert change_regions == expected_change, f"Expected change regions {expected_change}, got {change_regions}"


def test_segmentation():
    # Simple case
    _test_segmentation(
        sequences=[
            generate_flat_sequence(6, 1 + 1j),
            generate_changing_sequence(6, 1.5 + 1.5j, 2 + 2j),
            generate_flat_sequence(6, 2.5 + 2.5j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
        ],
        expected_flat=[(0, 6), (12, 18)],
        expected_change=[(6, 12), (18, 24)]
    )

    # Start with a change region
    _test_segmentation(
        sequences=[
            generate_changing_sequence(6, 1.5 + 1.5j, 2 + 2j),
            generate_flat_sequence(6, 1 + 1j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
            generate_flat_sequence(6, 2.5 + 2.5j),
        ],
        expected_flat=[(6, 12), (18, 24)],
        expected_change=[(0, 6), (12, 18)]
    )

    # Start and end with a change region
    _test_segmentation(
        sequences=[
            generate_changing_sequence(6, 1.5 + 1.5j, 2 + 2j),
            generate_flat_sequence(6, 1 + 1j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
            generate_flat_sequence(6, 2.5 + 2.5j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
        ],
        expected_flat=[(6, 12), (18, 24)],
        expected_change=[(0, 6), (12, 18), (24, 30)]
    )

    # Start and end with a flat region
    _test_segmentation(
        sequences=[
            generate_flat_sequence(6, 1 + 1j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
            generate_flat_sequence(6, 2.5 + 2.5j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
            generate_flat_sequence(6, 1 + 1j),
        ],
        expected_flat=[(0, 6), (12, 18), (24, 30)],
        expected_change=[(6, 12), (18, 24)],
    )

    # Consecutive flat regions
    _test_segmentation(
        sequences=[
            generate_flat_sequence(6, 1 + 1j),
            generate_flat_sequence(6, 2.5 + 2.5j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
            generate_flat_sequence(6, 1 + 1j),
        ],
        expected_flat=[(0, 6), (6, 12), (18, 24)],
        expected_change=[(12, 18)],
    )

    # Short flat regions beginning
    _test_segmentation(
        sequences=[
            generate_flat_sequence(3, 1 + 1j),
            generate_flat_sequence(6, 2.5 + 2.5j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
            generate_flat_sequence(6, 1 + 1j),
        ],
        expected_flat=[(0, 3), (3, 9), (15, 21)],
        expected_change=[(9, 15)],
    )

    # Short flat regions beginning
    _test_segmentation(
        sequences=[
            generate_flat_sequence(3, 1 + 1j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
            generate_flat_sequence(3, 2.5 + 2.5j),
            generate_flat_sequence(6, 1 + 1j),
        ],
        expected_change=[(0, 12)],
        expected_flat=[(12, 18)],
    )

    # Short flat regions end
    _test_segmentation(
        sequences=[
            generate_flat_sequence(3, 1 + 1j),
            generate_flat_sequence(6, 2.5 + 2.5j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
            generate_flat_sequence(6, 1 + 1j),
        ],
        expected_flat=[(0, 3), (3, 9), (15, 21)],
        expected_change=[(9, 15)],
    )

    # change regions beginning and end: Most of the cases
    _test_segmentation(
        sequences=[
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
            generate_flat_sequence(6, 1 + 1j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
        ],
        expected_flat=[(6, 12)],
        expected_change=[(0, 6), (12, 18)],
    )

    # flat regions beginning and end
    _test_segmentation(
        sequences=[
            generate_flat_sequence(6, 1 + 1j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
            generate_flat_sequence(6, 1 + 1j),
        ],
        expected_flat=[(0, 6), (12, 18)],
        expected_change=[(6, 12)],
    )

    # Short flat regions beginning and end
    _test_segmentation(
        sequences=[
            generate_flat_sequence(3, 1 + 1j),
            generate_changing_sequence(6, 4 + 4j, 5 + 5j),
            generate_flat_sequence(3, 1 + 1j),
        ],
        expected_change=[(0, 12)],
        expected_flat=[]
    )
