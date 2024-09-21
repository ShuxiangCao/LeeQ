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

    def __init__(self, name: str, virtual_qubits: dict[int, VirtualTransmon],
                 omega_to_amp_map: dict[int, float] = None,
                 coupling_strength_map: dict[frozenset[int], float] = None, *args, **kwargs):
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
        coupling_strength_map: dict[frozenset[int],float]
            The coupling strength (J) map. Unit is MHz.

        """

        self._compiler = "dummy compiler"
        self._engine = "dummy engine"

        for v_qubit in virtual_qubits.values():
            if not isinstance(v_qubit, VirtualTransmon):
                raise ValueError(f"Virtual qubit {v_qubit} is not a VirtualTransmon.")
            # Check the names are unique
            if len([v for v in virtual_qubits.values() if v.name == v_qubit.name]) > 1:
                raise ValueError(f"Virtual qubit name {v_qubit.name} is not unique.")

        self._virtual_qubits = virtual_qubits
        if omega_to_amp_map is None:
            omega_to_amp_map = {}
        self._omega_per_amp_dict = omega_to_amp_map
        self._coupling_strength_map = coupling_strength_map

        if self._coupling_strength_map is None:
            self._coupling_strength_map = {}

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

    def get_coupling_strength_by_name(self, name1: str, name2: str) -> float:
        """
        Get the coupling strength between the two channels.

        Parameters
        ----------
        name1: str
            The name of the first channel.
        name2: str
        """

        virtual_qubit_names = [x.name for x in self._virtual_qubits.values()]

        if name1 not in virtual_qubit_names or name2 not in virtual_qubit_names:
            raise ValueError(f"Virtual qubit {name1} or {name2} not found in the virtual qubits.")

        if name1 == name2:
            raise ValueError(f"Virtual qubit {name1} and {name2} are the same.")

        key = frozenset([name1, name2])

        return self._coupling_strength_map.get(key, 0)

    def get_coupling_strength_by_qubit(self, qubit1: VirtualTransmon, qubit2: VirtualTransmon) -> float:
        """
        Get the coupling strength between the two qubits.

        Parameters
        ----------
        qubit1: VirtualTransmon
            The first qubit element.
        qubit2: VirtualTransmon
            The second qubit element.

        Returns
        ------
        float
            The coupling strength in MHz.
        """

        return self.get_coupling_strength_by_name(qubit1.name, qubit2.name)

    def set_coupling_strength_by_name(self, name_1: str, name_2: str, coupling_strength: float):
        """
        Set the coupling strength between the two virtual qubits.

        Parameters
        ----------
        name_1: str
            The name of the first virtual qubit.
        name_2: str
            The name of the second virtual qubit.
        coupling_strength:
            The coupling strength in MHz.
        Returns
        -------
        """

        self._coupling_strength_map[frozenset([name_1, name_2])] = coupling_strength
        pass

    def set_coupling_strength_by_qubit(self, qubit1: VirtualTransmon, qubit2: VirtualTransmon,
                                       coupling_strength: float):
        """
        Set the coupling strength between the two qubits.

        Parameters
        ----------
        qubit1: VirtualTransmon
            The first qubit element.
        qubit2: VirtualTransmon
            The second qubit element.
        coupling_strength: float
            The coupling strength in MHz.
        """

        self.set_coupling_strength_by_name(qubit1.name, qubit2.name, coupling_strength)
