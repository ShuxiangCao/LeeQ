import numpy as np
import pytest

from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon

configuration_q1 = {
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 5040.4,
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
            'freq': 5040.4 - 198,
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
            'freq': 9645.5,
            'channel': 1,
            'shape': 'square',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        },
        '1': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9144.41,
            'channel': 1,
            'shape': 'square',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1, 2]
        }
    }
}

configuration_q2 = {
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 5040. - 123,
            'channel': 4,
            'shape': 'blackman_drag',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 0.025,
            'alpha': 425.1365229849309,
            'trunc': 1.2
        },
        'f12': {
            'type': 'SimpleDriveCollection',
            'freq': 5040.4 - 198 - 123,
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
            'freq': 9645.5 + 100,
            'channel': 3,
            'shape': 'square',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        },
        '1': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9645.5 + 100,
            'channel': 3,
            'shape': 'square',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1, 2]
        }
    }
}


@pytest.fixture()
def simulation_setup():
    from leeq.chronicle import Chronicle
    Chronicle().start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    virtual_transmon_1 = VirtualTransmon(
        name="VQubit_1",
        qubit_frequency=5040.4,
        anharmonicity=-198,
        t1=70,
        t2=35,
        readout_frequency=9645.5,
        quiescent_state_distribution=np.asarray(
            [
                0.8,
                0.15,
                0.04,
                0.01]))

    virtual_transmon_2 = VirtualTransmon(
        name="VQubit_2",
        qubit_frequency=5040. - 123,
        anharmonicity=-197,
        t1=70,
        t2=35,
        readout_frequency=9645.5 + 100,
        quiescent_state_distribution=np.asarray(
            [
                0.8,
                0.15,
                0.04,
                0.01]))

    setup = HighLevelSimulationSetup(
        name='HighLevelSimulationSetup',
        virtual_qubits={2: virtual_transmon_1,
                        4: virtual_transmon_2
                        }
    )
    setup.set_coupling_strength_by_qubit(
        virtual_transmon_1, virtual_transmon_2, coupling_strength=1.5)

    manager.register_setup(setup)
    return manager


@pytest.fixture
def qubit_1():
    dut = TransmonElement(
        name='test_qubit_1',
        parameters=configuration_q1
    )

    return dut


@pytest.fixture
def qubit_2():
    dut = TransmonElement(
        name='test_qubit_2',
        parameters=configuration_q2
    )

    return dut


def test_get_coupling_strengths(simulation_setup):
    setup = simulation_setup.get_setup('HighLevelSimulationSetup')
    coupling_strengths = setup.get_coupling_strength_by_name('VQubit_1', 'VQubit_2')
    assert coupling_strengths == 1.5


def test_run_stark_shift_gate_continuous(simulation_setup, qubit_1, qubit_2):
    manager = ExperimentManager().get_default_setup(
    ).status.set_parameter("Plot_Result_In_Jupyter", False)
    from leeq.experiments.builtin import ConditionalStarkShiftContinuous

    continious_exp = ConditionalStarkShiftContinuous(
        duts=[qubit_1, qubit_2],
        amp_control=0.1,
        amp_target=0.1,
        frequency=5040 - 60,
        rise=0.015,
        start=0,
        stop=15,
        sweep_points=30,
        axis='Y',
        echo=True,
        iz_rate_cancel=0,
        phase_diff=0,
        iz_rise_drop=0
    )


def test_run_stark_shift_gate_repeated_gate(simulation_setup, qubit_1, qubit_2):
    manager = ExperimentManager().get_default_setup(
    ).status.set_parameter("Plot_Result_In_Jupyter", False)
    from leeq.experiments.builtin import ConditionalStarkShiftRepeatedGate
    repeated_gate_exp = ConditionalStarkShiftRepeatedGate(
        duts=[qubit_1, qubit_2],
        amp_control=0.1, amp_target=0.1, frequency=5040 - 60, phase_diff=0, rise=0.03,
        echo=True, iz_control=0, iz_target=0, width=0.2, start_gate_number=0, gate_count=40)
