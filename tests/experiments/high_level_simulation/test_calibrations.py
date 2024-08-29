import numpy as np
import pytest

from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.experiments.experiments import ExperimentManager
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon


@pytest.fixture()
def simulation_setup():
    from labchronicle import Chronicle
    Chronicle().start_log()
    manager = ExperimentManager()
    manager.clear_setups()

    virtual_transmon = VirtualTransmon(
        name="VQubit",
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

    setup = HighLevelSimulationSetup(
        name='HighLevelSimulationSetup',
        virtual_qubits={2: virtual_transmon}
    )
    manager.register_setup(setup)
    return manager


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


@pytest.fixture
def qubit():
    dut = TransmonElement(
        name='test_qubit',
        parameters=configuration
    )

    return dut


def test_normalised_rabi(simulation_setup, qubit):
    from leeq.experiments.builtin.basic.calibrations.rabi import NormalisedRabi
    manager = ExperimentManager().get_default_setup(
    ).status.set_parameter("Plot_Result_In_Jupyter", False)
    rabi = NormalisedRabi(
        dut_qubit=qubit,
        amp=0.51,
        start=0.00,
        stop=0.5,
        step=0.001
    )


def test_ramsey(simulation_setup, qubit):
    from leeq.experiments.builtin.basic.calibrations.ramsey import SimpleRamseyMultilevel
    manager = ExperimentManager().get_default_setup(
    ).status.set_parameter("Plot_Result_In_Jupyter", False)
    ramsey = SimpleRamseyMultilevel(
        qubit=qubit,
    )


def test_gmm_measurements(simulation_setup, qubit):
    from leeq.experiments.builtin import MeasurementCalibrationMultilevelGMM
    manager = ExperimentManager().get_default_setup(
    ).status.set_parameter("Plot_Result_In_Jupyter", False)
    cali = MeasurementCalibrationMultilevelGMM(
        dut=qubit,
        sweep_lpb_list=['0', '1'],
        mprim_index=0
    )


def test_resonator_spectroscopy(simulation_setup, qubit):
    from leeq.experiments.builtin import ResonatorSweepTransmissionWithExtraInitialLPB
    manager = ExperimentManager().get_default_setup(
    ).status.set_parameter("Plot_Result_In_Jupyter", False)
    sweep = ResonatorSweepTransmissionWithExtraInitialLPB(
        qubit,
        start=9100,
        stop=9200,
        step=0.002,
        num_avs=1e3
    )


def test_qubit_spectroscopy(simulation_setup, qubit):
    from leeq.experiments.builtin import QubitSpectroscopyFrequency
    manager = ExperimentManager().get_default_setup(
    ).status.set_parameter("Plot_Result_In_Jupyter", False)
    sweep = QubitSpectroscopyFrequency(
        dut_qubit=qubit,
        res_freq=9141.21, start=3.e3, stop=5.e3,
        step=1., num_avs=1000,
        rep_rate=0., mp_width=0.5, amp=0.01
    )


def test_drag_clibration(simulation_setup, qubit):
    from leeq.experiments.builtin import DragCalibrationSingleQubitMultilevel
    manager = ExperimentManager().get_default_setup(
    ).status.set_parameter("Plot_Result_In_Jupyter", False)
    sweep = DragCalibrationSingleQubitMultilevel(
        dut=qubit,
    )


def test_pingpong_clibration(simulation_setup, qubit):
    from leeq.experiments.builtin import AmpPingpongCalibrationSingleQubitMultilevel
    manager = ExperimentManager().get_default_setup(
    ).status.set_parameter("Plot_Result_In_Jupyter", False)
    sweep = AmpPingpongCalibrationSingleQubitMultilevel(
        dut=qubit,
    )
