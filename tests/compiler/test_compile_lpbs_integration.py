from pytest import fixture

from leeq.compiler.full_sequecing.full_sequencing import FullSequencingCompiler
from leeq.core.context import ExperimentContext
from leeq.core.elements.built_in.qudit_transmon import TransmonElement


@fixture
def qubit_q1():
    configuration_q1 = {
        'lpb_collections': {
            'f01': {
                'type': 'SimpleDriveCollection',
                'freq': 5144.1,
                'channel': 4,
                'shape': 'square',
                'amp': 0.21323904814245054 / 5 * 4,
                'phase': 0.,
                'width': 0.025,
            },
            'f12': {
                'type': 'SimpleDriveCollection',
                'freq': 3144.0,
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
                'freq': 8144.41,
                'channel': 3,
                'shape': 'square',
                'amp': 0.2323904814245054 / 5 * 4,
                'phase': 0.,
                'width': 0.5,
                'trunc': 1.2,
                'distinguishable_states': [0, 1]
            }
        }
    }

    return TransmonElement(name='q1', parameters=configuration_q1)


@fixture
def qubit_q2():
    configuration_q2 = {
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
                'shape': 'gaussian_drag',
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
                'shape': 'Square',
                'amp': 0.21323904814245054 / 5 * 4,
                'phase': 0.,
                'width': 1,
                'trunc': 1.2,
                'distinguishable_states': [0, 1]
            }
        }
    }

    return TransmonElement(name='q2', parameters=configuration_q2)


def test_compile_primitive_to_sequence_integration(qubit_q1):
    hq1 = qubit_q1.get_lpb_collection('f01')['Yp']
    mprim_1 = qubit_q1.get_measurement_primitive('0')

    compiler = FullSequencingCompiler(
        sampling_rate={
            1: 1e6,
            2: 1e6,
            3: 1e6,
            4: 1e6
        }
    )

    context = ExperimentContext(name='test_compile_primitive_to_sequence_integration')
    result = compiler.compile_lpb(context=context, lpb=hq1)

    assert result.instructions['pulse_sequence'][(4, 5144.1)].shape == (25000,)
    assert result.instructions['pulse_sequence'][(4, 5144.1)].dtype == 'complex64'


def test_compile_lpb_to_full_sequence_integration(qubit_q1, qubit_q2):
    hq1 = qubit_q1.get_gate('qutrit_hadamard')

    hq2 = qubit_q2.get_lpb_collection('f01')['Yp'] + qubit_q2.get_lpb_collection('f12')['Xp']

    mprim_1 = qubit_q1.get_measurement_primitive('0')
    mprim_2 = qubit_q1.get_measurement_primitive('0')

    lpb = hq1 + hq2 + mprim_1 * mprim_2

    compiler = FullSequencingCompiler(
        sampling_rate={
            1: 1e2,
            2: 1e2,
            3: 1e2,
            4: 1e2
        }
    )

    context = ExperimentContext(name='test_compile_primitive_to_sequence_integration')
    result = compiler.compile_lpb(context=context, lpb=lpb)

    pulses = result.instructions['pulse_sequence']
    assert len(pulses) == 4

    from matplotlib import pyplot as plt
    for key, val in pulses.items():
        plt.plot(val.real, label=str(key) + ' real')
        plt.plot(val.imag, label=str(key) + ' imag')
    plt.legend()
    plt.show()

    # Assert all the pulses has the same length
    length = pulses[(4, 5144.1)].shape
    for pulse in pulses.values():
        assert pulse.shape == length
