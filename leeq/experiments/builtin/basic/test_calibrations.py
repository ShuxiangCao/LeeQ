import pytest
from pytest import fixture

from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.experiments.builtin.basic.calibrations import *
from leeq.setups.built_in.setup_numpy_2q_virtual_device import Numpy2QVirtualDeviceSetup


@fixture
def virtual_setup():
    return Numpy2QVirtualDeviceSetup(sampling_rate=1e6)


configuration_1 = {
    "lpb_collections": {
        "f01": {
            "type": "SimpleDriveCollection",
            "freq": 4144.417053428905,
            "channel": 0,
            "shape": "blackman_drag",
            "amp": 0.21323904814245054 / 5 * 4,
            "phase": 0.0,
            "width": 0.025,
            "alpha": 425.1365229849309,
            "trunc": 1.2,
        },
        "f12": {
            "type": "SimpleDriveCollection",
            "freq": 4144.417053428905,
            "channel": 0,
            "shape": "blackman_drag",
            "amp": 0.21323904814245054 / 5 * 4,
            "phase": 0.0,
            "width": 0.025,
            "alpha": 425.1365229849309,
            "trunc": 1.2,
        },
    },
    "measurement_primitives": {
        "0": {
            "type": "SimpleDispersiveMeasurement",
            "freq": 9144.41,
            "channel": 1,
            "shape": "square",
            "amp": 0.21323904814245054 / 5 * 4,
            "phase": 0.0,
            "width": 1,
            "trunc": 1.2,
            "distinguishable_states": [0, 1],
        }
    },
}

configuration_2 = {
    "lpb_collections": {
        "f01": {
            "type": "SimpleDriveCollection",
            "freq": 4144.417053428905,
            "channel": 2,
            "shape": "blackman_drag",
            "amp": 0.21323904814245054 / 5 * 4,
            "phase": 0.0,
            "width": 0.025,
            "alpha": 425.1365229849309,
            "trunc": 1.2,
        },
        "f12": {
            "type": "SimpleDriveCollection",
            "freq": 4144.417053428905,
            "channel": 2,
            "shape": "blackman_drag",
            "amp": 0.21323904814245054 / 5 * 4,
            "phase": 0.0,
            "width": 0.025,
            "alpha": 425.1365229849309,
            "trunc": 1.2,
        },
    },
    "measurement_primitives": {
        "0": {
            "type": "SimpleDispersiveMeasurement",
            "freq": 9144.41,
            "channel": 3,
            "shape": "square",
            "amp": 0.21323904814245054 / 5 * 4,
            "phase": 0.0,
            "width": 1,
            "trunc": 1.2,
            "distinguishable_states": [0, 1],
        }
    },
}


@fixture
def qubit_1():
    dut = TransmonElement(name="test_qubit1", parameters=configuration_1)

    return dut


def qubit_2():
    dut = TransmonElement(name="test_qubit2", parameters=configuration_2)

    return dut


@pytest.mark.skip(reason="Too slow, needs update.")
def test_ResonatorSweepTransmissionWithExtraInitialLPB(virtual_setup, qubit_1):
    setup().register_setup(virtual_setup)

    ResonatorSweepTransmissionWithExtraInitialLPB(
        qubit_1,
        start=8000,
        stop=9000,
        step=1.0,
        res_power=-10.0,
        num_avs=1000,
        rep_rate=10.0,
        mp_width=None,
        initial_lpb=None,
        update=True,
        amp=50e-3,
    )

    setup().clear_setups()
