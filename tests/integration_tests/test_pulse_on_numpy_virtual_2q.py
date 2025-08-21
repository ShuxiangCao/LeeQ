from pytest import fixture
from leeq.chronicle import log_and_record, register_browser_function

from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.experiments.experiments import Experiment
from leeq.experiments.experiments import setup
from leeq.experiments.experiments import basic_run as basic
from leeq.setups.built_in.setup_numpy_2q_virtual_device import Numpy2QVirtualDeviceSetup


class DummyObject(object):
    def __init__(self):
        self.result = None
        self.result_raw = None


dummy_obj = DummyObject()
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
def qubit():
    dut = TransmonElement(
        name='test_qubit',
        parameters=configuration
    )

    return dut


class SimpleSampleExperiment(Experiment):
    # Define the quantum element

    # Generate circuit
    @log_and_record
    def run(self, dut, lpb, mprim_index=0):
        mprim = dut.get_measurement_prim_intlist(mprim_index)

        lpb = lpb + mprim

        assert mprim.uuid in lpb.nodes

        basic(lpb, None, None)

        self.results = mprim.result()
        self.raw_values = mprim.result_raw()

    @register_browser_function()
    def plot(self):
        dummy_obj.result = self.results
        dummy_obj.result_raw = self.raw_values


def test_pulse_on_virtual_2q(qubit):
    # Prepare some lpb to prepare a non-trivial distribution
    setup().clear_setups()

    exp_setup = Numpy2QVirtualDeviceSetup()
    setup().register_setup(exp_setup)

    lpb = qubit.get_lpb_collection('f01')['Xp']
    mprim = qubit.get_measurement_primitive('0')
    lpb += mprim

    assert mprim.uuid in lpb.nodes

    SimpleSampleExperiment(qubit, lpb=lpb)

    assert dummy_obj.result is not None
    assert dummy_obj.result_raw is not None
