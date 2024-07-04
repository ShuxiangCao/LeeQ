from typing import List

import numpy as np

from leeq.core.context import ExperimentContext
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.core.engine.measurement_result import MeasurementResult
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import Sweeper
from leeq.setups.setup_base import ExperimentalSetup
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.theory.simulation.qutip.pulsed_simulator import QutipPulsedSimulator


class HighLevelSimulationSetup(ExperimentalSetup):
    """
    The HighLevelSimulationSetup class defines a high level simulation that directly simulates the experiment by
    running the theoretical simulation of each experiment, instead of pulse level physical simulations.
    """

    def __init__(self, name: str, virtual_qubits: dict[int,VirtualTransmon],
                 omega_to_amp_map: dict[int, float] = None, *args, **kwargs):
        """
        Initialize the HighLevelSimulationSetup class.

        Parameters
        ----------
        name: str
            The name of the setup.
        virtual_qubits: dict[int,VirtualTransmon]
            The virtual qubits. The key is the channel index id for the driving port.
            Here we assume all the readout channels are the same / multiplexed.
        omega_to_amp_map: dict[int,float]
            The mapping from the omega to the amp.
        """

        self._compiler = "dummy compiler"
        self._engine = "dummy engine"
        self._virtual_qubits = virtual_qubits
        if omega_to_amp_map is None:
            omega_to_amp_map = {}
        self._omega_per_amp_dict = omega_to_amp_map

        super(HighLevelSimulationSetup, self).__init__(name)
        self._status.add_param("Sampling_Noise", True)
        self._status.set_param("High_Level_Simulation_Mode", True)

    def get_virtual_qubit(self, dut_qubit: TransmonElement) -> VirtualTransmon:
        """
        Get the virtual qubit for the given dut qubit.

        Parameters
        ----------
        dut_qubit: TransmonElement The dut qubit element. The channel configuration will be mapped to the virtual qubit.

        Returns
        -------
        VirtualTransmon
            The virtual qubit.
        """

        channel = dut_qubit.get_default_c1().channel

        if channel not in self._virtual_qubits:
            raise ValueError(f"Channel {channel} not found in the virtual qubits.")

        return self._virtual_qubits[channel]

    def get_omega_per_amp(self, channel: int) -> float:
        """
        Get the omega per amp for the given channel.

        Parameters
        ----------
        channel: int: The channel index id.

        Returns
        -------
        float
            The omega per amp.
        """

        # TODO: Implement mapping

        return self._omega_per_amp_dict.get(channel, 200)
