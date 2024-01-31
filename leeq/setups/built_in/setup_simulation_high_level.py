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

    def __init__(self, name: str, virtual_qubit: VirtualTransmon, omega_to_amp_map: dict[int, float] = None, *args,
                 **kwargs):
        """
        Initialize the HighLevelSimulationSetup class.

        Parameters
        ----------
        name: str
            The name of the setup.
        virtual_qubit: VirtualTransmon
            The virtual qubit.
        omega_to_amp_map: dict[int,float]
            The mapping from the omega to the amp.
        """

        # TODO: Implement multiple qubit support.

        self._compiler = "dummy compiler"
        self._engine = "dummy engine"
        self._virtual_qubit = virtual_qubit
        if omega_to_amp_map is None:
            omega_to_amp_map = {}
        self._omega_per_amp_dict = omega_to_amp_map

        super(HighLevelSimulationSetup, self).__init__(name)
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

        # TODO: Implement the mapping from the dut qubit to the virtual qubit.
        return self._virtual_qubit

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

        return self._omega_per_amp_dict.get(channel, 20)
