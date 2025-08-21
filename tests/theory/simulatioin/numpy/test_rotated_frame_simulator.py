import pytest
from pytest import fixture
import numpy as np

from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon


def test_initialization():
    vt = VirtualTransmon(name="Qubit1",
                         qubit_frequency=5,
                         anharmonicity=0.1,
                         t1=30,
                         t2=30,
                         readout_frequency=7)
    assert vt.name == "Qubit1"
    assert vt.qubit_frequency == 5
    assert vt.anharmonicity == 0.1
    assert vt.t1 == 30
    assert vt.t2 == 30
    assert vt.readout_frequency == 7


def test_apply_readout_iq():
    vt = VirtualTransmon(name="Qubit2",
                         qubit_frequency=5,
                         anharmonicity=0.1,
                         t1=30,
                         t2=30,
                         readout_frequency=7)
    iq_data = vt._apply_readout_iq(readout_frequency=7,
                                   sampling_number=100,
                                   iq_noise_std=0.01,
                                   readout_width=2)
    assert isinstance(iq_data, np.ndarray)
    assert len(iq_data) == 100


def test_apply_averaged_iq():
    vt = VirtualTransmon(name="Qubit3",
                         qubit_frequency=5,
                         anharmonicity=0.1,
                         t1=30,
                         t2=30,
                         readout_frequency=7)
    averaged_iq = vt._apply_averaged_iq(readout_frequency=7,
                                        iq_noise_std=0.01,
                                        readout_width=2)
    assert isinstance(averaged_iq, (np.complex64, np.complex128))


def test_apply_drive():
    vt = VirtualTransmon(name="Qubit4",
                         qubit_frequency=5000,
                         anharmonicity=-200,
                         t1=30,
                         t2=30,
                         readout_frequency=7)
    pulse_shape = np.ones(100)
    sampling_rate = 1
    try:
        vt.apply_drive(frequency=5,
                       pulse_shape=pulse_shape,
                       sampling_rate=sampling_rate)
    except Exception as e:
        pytest.fail(f"apply_drive raised exception: {e}")


@fixture
def mock_qubit():
    vt = VirtualTransmon(name="QubitMock",
                         qubit_frequency=5000,
                         anharmonicity=-200,
                         t1=30,
                         t2=30,
                         readout_frequency=8900,
                         quiescent_state_distribution=[0.85, 0.1, 0.05, 0],
                         readout_linewith=0.25
                         )

    return vt


def test_integration_mock_population(mock_qubit):
    fs = np.linspace(8880, 8920, 100)

    response = []
    for f in fs:
        result = mock_qubit.apply_readout(
            return_type='population_distribution',
            sampling_number=100,
            iq_noise_std=0.01,
            readout_width=5,
            readout_frequency=f
        )
        response.append(result)

    # Make sure all the values in response are equal and between 0 and 1
    assert np.allclose(response, response[0])
    assert np.all(np.array(response) >= 0)
    assert np.all(np.array(response) <= 1)


def test_integration_mock_resonator_spectroscopy(mock_qubit):
    fs = np.linspace(8880, 8920, 100)

    response = []

    for f in fs:
        result = mock_qubit.apply_readout(
            return_type='IQ_average',
            sampling_number=100,
            iq_noise_std=0.01,
            readout_width=2,
            readout_frequency=f
        )
        response.append(result)

    # Make sure the response is a complex number
    assert isinstance(response[0], (np.complex64, np.complex128))

    # plot the response
    plot = False
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fs, np.abs(response))
        plt.show()

        plt.figure()
        plt.plot(fs, np.angle(response))
        plt.show()


def test_integration_mock_IQ_single_shot(mock_qubit):
    f = 8900

    result = mock_qubit.apply_readout(
        return_type='IQ',
        sampling_number=1000,
        iq_noise_std=0.01,
        readout_width=2,
        readout_frequency=f
    )

    # plot the response
    plot = False
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(result.real, result.imag)
        plt.show()


def test_integration_mock_drive(mock_qubit):
    mock_qubit.apply_drive(
        frequency=5000, pulse_shape=np.ones(10) * 0.01, sampling_rate=1
    )

    result = mock_qubit.apply_readout(
        return_type='population_distribution'
    )

    assert not np.allclose(result, [0.85, 0.1, 0.05, 0])
    assert sum(result) == 1
