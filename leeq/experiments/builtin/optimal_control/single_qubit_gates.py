
import matplotlib.pyplot as plt
import numpy as np

import leeq
from leeq import Experiment
from leeq.chronicle import log_and_record, register_browser_function
from leeq.core.primitives import LogicalPrimitiveCollectionFactory

__all__ = [
    'GRAPESingleQubitGate',
]

logger = leeq.utils.setup_logging(__name__)


class GRAPESingleQubitGate(Experiment):
    EPII_INFO = {
        "name": "GRAPESingleQubitGate",
        "description": "GRAPE: Gradient Ascent Pulse Engineering for single qubit gates",
        "purpose": "Generates optimized pulse shapes for single qubit gates using the GRAPE algorithm. Creates a new collection with the optimized pulse shape and calibrates the amplitude using Rabi experiments.",
        "attributes": {
            "grape_result": {
                "type": "object",
                "description": "Result object from the GRAPE optimization containing the optimized control fields",
                "fields": {
                    "u": "np.ndarray[complex] - Control field evolution array with shape (iterations, 2, time_points)"
                }
            },
            "rabi": {
                "type": "NormalisedRabi",
                "description": "Rabi experiment instance used for amplitude calibration"
            },
            "rabi_rate_per_amp": {
                "type": "float",
                "description": "Calibrated Rabi rate per unit amplitude (MHz/amp)"
            },
            "collection": {
                "type": "LogicalPrimitiveCollection",
                "description": "The generated collection containing the GRAPE pulse"
            }
        },
        "notes": [
            "The GRAPE algorithm optimizes pulse shapes for quantum gates using gradient ascent",
            "Creates a new collection with the optimized pulse shape",
            "Uses Rabi experiments to calibrate the pulse amplitude",
            "The anharmonicity is hardcoded to -200 MHz in the current implementation",
            "Pulse shape is normalized and stored as an arbitrary waveform"
        ]
    }
    @log_and_record
    def run(self,
            dut,
            reference_collection_name: str = 'f01',
            grape_collection_name: str = 'f01_grape',
            mprim_index: int = 0,
            initial_lpb=None,
            rabi_amp=0.1,
            rabi_start=0.01,
            rabi_stop=0.3,
            rabi_step=0.01,
            ) -> None:
        """
        Execute the GRAPE optimization experiment on hardware.

        Generates a GRAPE (Gradient Ascent Pulse Engineering) pulse for a single qubit gate.
        Creates a new collection with the optimized pulse shape. The experiment first calls
        the GRAPE method to obtain a pulse shape, then uses a Rabi experiment to extract
        the Rabi rate-amplitude relation for calibration.

        Parameters
        ----------
        dut : TransmonElement
            The device under test (qubit object).
        reference_collection_name : str, optional
            The name of the reference collection. Default: 'f01'
        grape_collection_name : str, optional
            The name of the collection to store the GRAPE pulse. Default: 'f01_grape'
        mprim_index : int, optional
            The index of the measurement primitive. Default: 0
        initial_lpb : LogicalPrimitiveBlock, optional
            The initial logical primitive block for state preparation. Default: None
        rabi_amp : float, optional
            The amplitude for the Rabi calibration experiment. Default: 0.1
        rabi_start : float, optional
            The start amplitude for the Rabi sweep. Default: 0.01
        rabi_stop : float, optional
            The stop amplitude for the Rabi sweep. Default: 0.3
        rabi_step : float, optional
            The step amplitude for the Rabi sweep. Default: 0.01

        Returns
        -------
        None
            Results are stored in instance attributes (grape_result, rabi, collection).
        """

        # Run the GRAPE experiment
        from leeq.theory.optimal_control.grape import get_single_qubit_pulse_grape
        grape = get_single_qubit_pulse_grape(qubit_frequency=dut.get_c1('f01').freq, anharmonicity=-200,
                                             width=dut.get_c1('f01').width, sampling_rate=2e3, initial_guess=None,
                                             max_amplitude=0.015, )
        self.grape_result = grape

        # Run the Rabi experiment
        from leeq.experiments.builtin import NormalisedRabi
        self.rabi = NormalisedRabi(dut_qubit=dut,
                                   amp=rabi_amp,
                                   start=rabi_start,
                                   stop=rabi_stop,
                                   step=rabi_step,
                                   fit=True,
                                   collection_name=reference_collection_name,
                                   mprim_index=mprim_index,
                                   update=False,
                                   initial_lpb=initial_lpb)

        self.rabi_rate_per_amp = self.rabi.fit_params['Frequency'] / rabi_amp

        amp_per_rabi_rate = 1 / self.rabi_rate_per_amp * 1000

        pulse_shape = self.grape_result.u[-1, :, :]  # *amp_per_rabi_rate*2
        pulse_shape = pulse_shape / np.max(np.abs(pulse_shape))

        amp = amp_per_rabi_rate * np.max(np.abs(pulse_shape))
        # Create a new collection

        reference_collection = dut.get_c1(reference_collection_name)

        grape_collection_parameters = {
            'type': 'SimpleDriveCollection',
            "transition_name": reference_collection.transition_name,
            "channel": reference_collection.channel,
            "freq": reference_collection.freq,
            "amp": amp,
            "phase": reference_collection.phase,
            "width": self.grape_result.u.shape[-1] / 2e3,
            # "trunc":1.2,
            "shape": 'arbitrary_pulse',
            # "shape": 'gaussian',
            "pulse_window_array": pulse_shape.tolist(),
        }

        factory = LogicalPrimitiveCollectionFactory()
        collection = factory(grape_collection_parameters['type'], grape_collection_name, grape_collection_parameters)

        self.collection = collection
        dut.add_collection(grape_collection_name, collection)

    @log_and_record
    def run_simulated(self,
                      dut,
                      reference_collection_name: str = 'f01',
                      grape_collection_name: str = 'f01_grape',
                      mprim_index: int = 0,
                      initial_lpb=None,
                      rabi_amp=0.1,
                      rabi_start=0.01,
                      rabi_stop=0.3,
                      rabi_step=0.01,
                      ) -> None:
        """
        Execute the GRAPE optimization experiment in simulation mode.

        Generates a GRAPE (Gradient Ascent Pulse Engineering) pulse for a single qubit gate.
        Creates a new collection with the optimized pulse shape. The experiment first calls
        the GRAPE method to obtain a pulse shape, then uses a Rabi experiment to extract
        the Rabi rate-amplitude relation for calibration.

        Parameters
        ----------
        dut : TransmonElement
            The device under test (qubit object).
        reference_collection_name : str, optional
            The name of the reference collection. Default: 'f01'
        grape_collection_name : str, optional
            The name of the collection to store the GRAPE pulse. Default: 'f01_grape'
        mprim_index : int, optional
            The index of the measurement primitive. Default: 0
        initial_lpb : LogicalPrimitiveBlock, optional
            The initial logical primitive block for state preparation. Default: None
        rabi_amp : float, optional
            The amplitude for the Rabi calibration experiment. Default: 0.1
        rabi_start : float, optional
            The start amplitude for the Rabi sweep. Default: 0.01
        rabi_stop : float, optional
            The stop amplitude for the Rabi sweep. Default: 0.3
        rabi_step : float, optional
            The step amplitude for the Rabi sweep. Default: 0.01

        Returns
        -------
        None
            Simulated results are stored in instance attributes (grape_result, rabi, collection).
        """
        # In simulation mode, we run the same GRAPE optimization
        # The GRAPE algorithm itself is already a simulation/optimization
        from leeq.theory.optimal_control.grape import get_single_qubit_pulse_grape
        grape = get_single_qubit_pulse_grape(qubit_frequency=dut.get_c1('f01').freq, anharmonicity=-200,
                                             width=dut.get_c1('f01').width, sampling_rate=2e3, initial_guess=None,
                                             max_amplitude=0.015, )
        self.grape_result = grape

        # Run the Rabi experiment in simulation
        from leeq.experiments.builtin import NormalisedRabi
        self.rabi = NormalisedRabi(dut_qubit=dut,
                                   amp=rabi_amp,
                                   start=rabi_start,
                                   stop=rabi_stop,
                                   step=rabi_step,
                                   fit=True,
                                   collection_name=reference_collection_name,
                                   mprim_index=mprim_index,
                                   update=False,
                                   initial_lpb=initial_lpb)

        self.rabi_rate_per_amp = self.rabi.fit_params['Frequency'] / rabi_amp

        amp_per_rabi_rate = 1 / self.rabi_rate_per_amp * 1000

        pulse_shape = self.grape_result.u[-1, :, :]  # *amp_per_rabi_rate*2
        pulse_shape = pulse_shape / np.max(np.abs(pulse_shape))

        amp = amp_per_rabi_rate * np.max(np.abs(pulse_shape))
        # Create a new collection

        reference_collection = dut.get_c1(reference_collection_name)

        grape_collection_parameters = {
            'type': 'SimpleDriveCollection',
            "transition_name": reference_collection.transition_name,
            "channel": reference_collection.channel,
            "freq": reference_collection.freq,
            "amp": amp,
            "phase": reference_collection.phase,
            "width": self.grape_result.u.shape[-1] / 2e3,
            # "trunc":1.2,
            "shape": 'arbitrary_pulse',
            # "shape": 'gaussian',
            "pulse_window_array": pulse_shape.tolist(),
        }

        factory = LogicalPrimitiveCollectionFactory()
        collection = factory(grape_collection_parameters['type'], grape_collection_name, grape_collection_parameters)

        self.collection = collection
        dut.add_collection(grape_collection_name, collection)

    @register_browser_function()
    def visual_analyze(self):
        """
        Visualize the pulse shape.

        Parameters
        """
        result = self.grape_result

        times = np.arange(result.u.shape[-1])

        from qutip.control import plot_grape_control_fields
        plot_grape_control_fields(times, result.u, ['x', 'y'])
        # Print results
        plt.grid()
        return plt.gcf()
