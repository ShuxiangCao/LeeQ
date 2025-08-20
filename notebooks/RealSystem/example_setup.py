from leeq.setups.qubic_lbnl_setups import QubiCSingleBoardRemoteRPCSetup
from leeq.chronicle import Chronicle
from leeq.experiments import setup

# Start the logging
Chronicle().start_log()


class QubiCSetupTest(QubiCSingleBoardRemoteRPCSetup):

    @staticmethod
    def _readout_frequency_mixing_callback(parameters: dict):
        """
        This function changes the frequency of the lpb parameters of the readout channel,
        considering we have a mixing of 15GHz signal.
        """

        if 'freq' not in parameters:
            return parameters

        modified_parameters = parameters.copy()
        modified_parameters['freq'] = 15000-parameters['freq']  # 4-8 GHz for IF Readout
        return modified_parameters

    def __init__(self):
        super().__init__(
            name='qubic_setup',
            rpc_uri='http://192.168.137.81:9095' # For QubiC ZCU216 board
        )

        # Register call function for all readout channels. Delete when not needed.

        for i in range(8):
            readout_channel = 2 * i + 1
            self._status.register_compile_lpb_callback(
                channel=readout_channel,
                callback=self._readout_frequency_mixing_callback
            )

# Examples of the qubit parameters:
q1_params = {
    'hrid':'Q1',
    'lpb_collections':{
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 4888.205889560021,
            'channel': 0,
            'shape': 'blackman_drag',
            'amp': 0.21,
            'phase': 0.,
            'width': 0.05,
            'alpha': 1e9,
            'trunc': 1.2
        },
        'f12': {
            'type': 'SimpleDriveCollection',
            'freq': 4888.205889560021-200,
            'channel': 0,
            'shape': 'blackman_drag',
            'amp': 0.21 / 5 * 4,
            'phase': 0.,
            'width': 0.025,
            'alpha': 1e9,
            'trunc': 1.2
        },
        'f23': {
            'type': 'SimpleDriveCollection',
            'freq': 4888.205889560021-200 - 200*1.11,
            'channel': 0,
            'shape': 'blackman_drag',
            'amp': 0.21 / 5 * 4,
            'phase': 0.,
            'width': 0.025,
            'alpha': 1e9,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9997.6,
            'channel': 1,
            'shape': 'square',
            'amp': 0.06,
            'phase': 0.,
            'width': 8,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        },
        '1': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9997.6,
            'channel': 1,
            'shape': 'square',
            'amp': 0.06,
            'phase': 0.,
            'width': 8,
            'trunc': 1.2,
            'distinguishable_states': [0, 1, 2]
        },
        '2': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9997.6,
            'channel': 1,
            'shape': 'square',
            'amp': 0.06,
            'phase': 0.,
            'width': 8,
            'trunc': 1.2,
            'distinguishable_states': [0, 1, 2, 3]
        }
    }
}


q2_params = {
    'hrid':'Q2',
    'lpb_collections':{
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 4795.569634070339,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.21,
            'phase': 0.,
            'width': 0.05,
            'alpha': 1e9,
            'trunc': 1.2
        },
        'f12': {
            'type': 'SimpleDriveCollection',
            'freq': 4795.569634070339-200,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.21 / 5 * 4,
            'phase': 0.,
            'width': 0.025,
            'alpha': 1e9,
            'trunc': 1.2
        },
        'f23': {
            'type': 'SimpleDriveCollection',
            'freq': 4795.569634070339-200 - 200*1.11,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.21 / 5 * 4,
            'phase': 0.,
            'width': 0.025,
            'alpha': 1e9,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq':  9386.8,
            'channel': 3,
            'shape': 'square',
            'amp': 0.05,
            'phase': 0.,
            'width': 8,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        },
        '1': {
            'type': 'SimpleDispersiveMeasurement',
            'freq':  9386.8,
            'channel': 3,
            'shape': 'square',
            'amp': 0.05,
            'phase': 0.,
            'width': 8,
            'trunc': 1.2,
            'distinguishable_states': [0, 1, 2]
        },
        '2': {
            'type': 'SimpleDispersiveMeasurement',
            'freq':  9386.8,
            'channel': 3,
            'shape': 'square',
            'amp': 0.05,
            'phase': 0.,
            'width': 8,
            'trunc': 1.2,
            'distinguishable_states': [0, 1, 2, 3]
        }
    }
}

setup_ = QubiCSetupTest()
setup().register_setup(setup_)