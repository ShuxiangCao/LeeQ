import uuid

import numpy as np

from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory
from leeq.core.primitives.built_in.common import PhaseShift
from leeq.core.primitives.built_in.compatibility import PulseArgsUpdatable
from leeq.core.primitives.logical_primitives import (
    LogicalPrimitive,
    LogicalPrimitiveFactory,
    MeasurementPrimitive, LogicalPrimitiveClone, MeasurementPrimitiveClone, LogicalPrimitiveBlockSerial,
)
from leeq.core.primitives.collections import LogicalPrimitiveCollection


class SimpleDrive(LogicalPrimitive, PulseArgsUpdatable):
    _parameter_names = ["freq", "channel", "shape", "amp", "phase", "width"]

    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)

    def calculate_envelope_area(self):
        """
        Calculate the area of the envelope.
        """
        factory = PulseShapeFactory()
        return factory.calculate_integrated_area(
            pulse_shape_name=self.shape,
            sampling_rate=2e3,  # In Msps unit. Just an estimation
            **self._parameters,
        )

    @staticmethod
    def _validate_parameters(parameters: dict):
        """
        Validate the parameters of the logical primitive.

        Parameters:
            parameters (dict): The parameters of the logical primitive.

        Raises:
            AssertionError: If the parameters are invalid.


        Example parameters:
        ```
         {
        'type': 'SimpleDriveCollection',
        'freq': 4144.417053428905,
        'channel': 2,
        'shape': 'BlackmanDRAG',
        'amp': 0.21323904814245054 / 5 * 4,
        'phase': 0.,
        'width': 0.025,
        'alpha': 425.1365229849309,
        'trunc': 1.2
        }
        ```
        """

        for parameter_name in SimpleDrive._parameter_names:
            assert (
                    parameter_name in parameters
            ), f"The parameter {parameter_name} is not found."

    def clone_with_parameters(self, parameters: dict, name_postfix=None):
        """
        Copy the logical primitive with the parameters updated.

        Parameters:
            parameters (dict): The parameters to be updated.
            name_postfix (str): The postfix to be added to the name of the logical primitive.

        Returns:
            LogicalPrimitive: The copied logical primitive.
        """

        if name_postfix is None:
            name_postfix = f"_clone_{uuid.uuid4()}"

        cloned_primitive = SimpleDriveClone(
            name=self._name + name_postfix,
            parameters=parameters,
            original=self)
        return cloned_primitive


class SimpleDriveClone(LogicalPrimitiveClone, PulseArgsUpdatable):
    pass


class SimpleDriveCollection(LogicalPrimitiveCollection):
    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)

        primitive_params = {"drive": "SimpleDrive"}

        # If the transition name is not specified, we will use the name of the
        # collection
        if "transition_name" not in "parameters":
            self.transition_name = name.split(".")[-1]

        factory = LogicalPrimitiveFactory()
        factory.register_collection_template(SimpleDrive)

        self._build_primitives(primitives_params=primitive_params)

    def __getitem__(self, item):
        """
        Get the logical primitive by name.

        Parameters:
            item (str): The name of the logical primitive.

        Returns:
            LogicalPrimitive: The logical primitive.

        Raises
            KeyError: If the logical primitive is not found.
        """

        if item == "I":
            return self._primitives["drive"].clone_with_parameters(
                {"amp": 0}, name_postfix="_I"
            )
        elif item == "X":
            return self._primitives["drive"]
        elif item == "Y":
            return self._primitives["drive"].clone_with_parameters(
                {"phase": np.pi / 2}, name_postfix="_Y"
            )
        elif item == "Xp":
            return self._primitives["drive"].clone_with_parameters(
                {"amp": self._parameters["amp"] / 2}, name_postfix="_Xp"
            )
        elif item == "Yp":
            return self._primitives["drive"].clone_with_parameters(
                {"amp": self._parameters["amp"] / 2, "phase": np.pi / 2},
                name_postfix="_Yp",
            )
        elif item == "Xm":
            return self._primitives["drive"].clone_with_parameters(
                {"amp": self._parameters["amp"] / 2, "phase": np.pi}, name_postfix="_Xm"
            )
        elif item == "Ym":
            return self._primitives["drive"].clone_with_parameters(
                {"amp": -self._parameters["amp"] / 2, "phase": np.pi / 2},
                name_postfix="_Ym",
            )

        return super(SimpleDriveCollection).__getitem__(item)

    def _get_amp_modified_primitive(self, gate_pi, angle):
        """
        Get the parameter modified primitive.

        Parameters:
            gate_pi (LogicalPrimitive): The gate to be modified.
            angle (float): The angle of the gate.
        """
        start_level = int(self.transition_name[1])
        end_level = int(self.transition_name[2])
        level_diff = end_level - start_level

        # Multi-photon transition will have different amplitude scaling
        new_amp = gate_pi.amp * (angle / np.pi) ** (1 / level_diff)
        return gate_pi.clone_with_parameters(
            {"amp": new_amp}, name_postfix=f"_{angle}")

    def hadamard(self):
        """
        Get the hadamard gate.
        """

        return self["Ym"] + self.z(angle=np.pi)

    def x(self, angle):
        """
        Get a X(angle) gate.

        Parameters:
            angle (float): The angle of the gate.

        Returns:
            LogicalPrimitive: The X(angle) gate.
        """
        return self._get_amp_modified_primitive(self["X"], angle)

    def y(self, angle):
        """
        Get a Y(angle) gate.

        Parameters:
            angle (float): The angle of the gate.

        Returns:
            LogicalPrimitive: The Y(angle) gate.
        """
        return self._get_amp_modified_primitive(self["Y"], angle)

    def z(self, angle):
        """
        Get a Z(angle) gate. Note this gate is not compatible to qutrits or qudits vz gates.

        Parameters:
            angle (float): The angle of the gate.
        """

        return PhaseShift(
            name=self._name + ".Z",
            parameters={
                "type": "PhaseShift",
                "channel": self.channel,  # Which channel to apply the phase shift
                # The phase shift value, note that adding a phase is equivalent to
                # displacing
                "phase_shift": -angle,
                # a negative phase to the reference frame, so we have minus sign
                # here.
                "transition_multiplier": {
                    # There is a trick to determine the sign of the phase shift for different
                    # transitions. If the level you want to change is the earlier letter, that's
                    # positive.
                    "f01": -1,
                },
            },
        )

    def z_omega(self, omega):
        """
        Get a virtual Z gate with a associated omega value, when setting the virtual width the shifted
        phase will be automatically updated.
        """
        vz = PhaseShift(
            name=self._name + ".Z",
            parameters={
                "type": "PhaseShift",
                "channel": self.channel,  # Which channel to apply the phase shift
                # The phase shift value, note that adding a phase is equivalent to
                # displacing
                "phase_shift": 0,
                # a negative phase to the reference frame, so we have minus sign
                # here.
                "transition_multiplier": {
                    # There is a trick to determine the sign of the phase shift for different
                    # transitions. If the level you want to change is the earlier letter, that's
                    # positive.
                    "f01": -1,
                },
            },
        )
        vz.set_omega(omega)
        return vz

    def get_clifford(self, clifford_id: int, ignore_identity=False):
        """
        Get a Clifford gate.

        Parameters:
            clifford_id (int): The id of the Clifford gate.
            ignore_identity (bool): Whether to ignore the identity gate.

        Returns:
            LogicalPrimitiveBlockSerial: The Clifford gate.
        """
        from leeq.theory.cliffords import get_clifford_from_id

        if ignore_identity and clifford_id == 0:
            return LogicalPrimitiveBlockSerial([])

        return LogicalPrimitiveBlockSerial(
            [self[x] for x in get_clifford_from_id(clifford_id)])


class QuditVirtualZCollection(LogicalPrimitiveCollection):
    @classmethod
    def _validate_parameters(cls, parameters: dict):
        """
        Validate the parameters of the logical primitive.
        """
        assert "channel" in parameters, "The channel is not specified."

    def z1(self, angle):
        """
        Get a Z1(angle) gate, adding a phase to the |1> state.
        """

        return PhaseShift(
            name=self._name + ".Z1",
            parameters={
                "type": "PhaseShift",
                "channel": self.channel,  # Which channel to apply the phase shift
                # The phase shift value, note that adding a phase is equivalent to
                # displacing
                "phase_shift": -angle,
                # a negative phase to the reference frame, so we have minus sign
                # here.
                "transition_multiplier": {
                    # There is a trick to determine the sign of the phase shift for different
                    # transitions. If the level you want to change is the earlier letter, that's
                    # positive.
                    "f01": -1,
                    "f12": 1,
                    "f13": 1,
                },
            },
        )

    def z2(self, angle):
        """
        Get a Z2(angle) gate, adding a phase to the |2> state.
        """
        return PhaseShift(
            name=self._name + ".Z2",
            parameters={
                "type": "PhaseShift",
                "channel": self.channel,
                "phase_shift": -angle,
                "transition_multiplier": {"f02": -1, "f12": -1, "f23": 1},
            },
        )

    def z3(self, angle):
        """
        Get a Z3(angle) gate, adding a phase to the |3> state.
        """

        return PhaseShift(
            name=self._name + ".Z3",
            parameters={
                "type": "PhaseShift",
                "channel": self.channel,
                "phase_shift": -angle,
                "transition_multiplier": {"f03": -1, "f13": -1, "f23": -1},
            },
        )


class SimpleDispersiveMeasurement(MeasurementPrimitive, PulseArgsUpdatable):
    """
    Describes a simple dispersive measurement which send a single frequency pulse and look at the reflected signal.
    """

    @classmethod
    def _validate_parameters(cls, parameters: dict):
        assert (
                "distinguishable_states" in parameters
        ), "The distinguishable states are not specified."

    def clone_with_parameters(self, parameters: dict, name_postfix=None):
        """
        Copy the logical primitive with the parameters updated.

        Parameters:
            parameters (dict): The parameters to be updated.
            name_postfix (str): The postfix to be added to the name of the logical primitive.

        Returns:
            LogicalPrimitive: The copied logical primitive.
        """

        if name_postfix is None:
            name_postfix = f"_clone_{uuid.uuid4()}"

        cloned_primitive = SimpleDispersiveMeasurementClone(
            name=self._name + name_postfix, parameters=parameters, original=self)
        return cloned_primitive


class SimpleDispersiveMeasurementClone(
    MeasurementPrimitiveClone,
    PulseArgsUpdatable):
    pass
