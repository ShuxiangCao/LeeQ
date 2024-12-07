from labchronicle import log_and_record, register_browser_function
import leeq
from leeq import Experiment, Sweeper, SweepParametersSideEffectFactory, basic_run, setup
from leeq.core.primitives import LogicalPrimitiveCollectionFactory
from leeq.core.primitives.built_in.simple_drive import SimpleDriveCollection
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.utils.compatibility import prims

from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'GRAPESingleQubitGate',
]

logger = leeq.utils.setup_logging(__name__)


class GRAPESingleQubitGate(Experiment):
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
        This experiment is used to generate a GRAPE pulse for a single qubit gate. It will create a new collection.

        It first calls the grape method to obtain a pulse shape, then use Rabi experiment to extract the rabi rate
        amplitude relation of the pulse to estimate the calibration. Then it will store the pulse shape into the
        collection.

        Parameters:
            dut (TransmonElement): The device under test.
            reference_collection_name (str): The name of the reference collection.
            grape_collection_name (str): The name of the collection to store the GRAPE pulse.
            mprim_index (int): The index of the measurement primitive.
            initial_lpb (Optional[LogicalPrimitiveBlock]): The initial logical primitive block.
            rabi_amp (float): The amplitude of the Rabi experiment.
            rabi_start (float): The start amplitude of the Rabi experiment.
            rabi_stop (float): The stop amplitude of the Rabi experiment.
            rabi_step (float): The step amplitude of the Rabi experiment.
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

        pulse_shape = self.grape_result.u[-1, :, :]#*amp_per_rabi_rate*2
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
