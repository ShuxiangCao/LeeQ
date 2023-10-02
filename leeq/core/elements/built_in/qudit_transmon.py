import numpy as np

from leeq.core.elements import Element
from leeq.core.primitives import LogicalPrimitiveCollectionFactory


class TransmonElement(Element):

    def __init__(self, name: str, parameters: dict = None):
        """
        Initialize the transmon element.

        Parameters:
            name (str): The name of the element.
            parameters (dict, Optional): The parameters of the element.
        """

        # Register necessary factory classes
        from leeq.core.primitives.built_in.simple_drive import SimpleDriveCollection
        from leeq.core.primitives.built_in.simple_drive import QuditVirtualZCollection

        factory = LogicalPrimitiveCollectionFactory()
        factory.register_collection_template(SimpleDriveCollection)

        # Build the element
        super().__init__(name, parameters)

        # Manually build the virtual z logical primitive collection
        # assert all the channels in lpb configurations are the same
        lpb_configurations = self._parameters['lpb_configurations']
        assert len(set([lpb_configuration['channel'] for lpb_configuration in lpb_configurations.values()])) == 1

        # Build the virtual z logical primitive collection
        self._lpb_collections['qudit_vz'] = QuditVirtualZCollection(name=self._name + '.' + 'qudit_vz',
                                                                    parameters={
                                                                        'channel': lpb_configurations.values[0][
                                                                            'channel']
                                                                    })

    def _validate_parameters(self, parameters: dict):
        """
        Validate the parameters of the element.

        Parameters:
            parameters (dict): The parameters of the element.
        """

        from leeq.core.primitives.built_in.simple_drive import SimpleDriveCollection

        for name, lpb_parameter in parameters['lpb_collections'].items():
            assert 'type' in lpb_parameter, 'The type of the lpb collection is not specified.'
            assert lpb_parameter['type'] in [SimpleDriveCollection.__qualname__], \
                f"The lpb collection {lpb_parameter['name']} is not supported."

        for name, measurement_parameter in parameters['measurement_primitives'].items():
            assert 'name' in measurement_parameter, 'The name of the measurement parameter is not specified.'
            assert measurement_parameter['name'] in ['simple_drive_measurement']

    def get_gate(self, gate_name, transition_name='f01', angle=None):
        """
        Get a specific gate For compatibility and convenience.

        Available gates:
            - All single qubit operation gates
            - qutrit_clifford_green: Qutrit green spider in qutrit ZX calculus
            - qutrit_clifford_red: Qutrit red spider in qutrit ZX calculus
            - qutrit_hadamard: Qutrit hadamard gate
            - qutrit_hadamard_dag: Qutrit hadamard gate

        Parameters:
            gate_name (str): The name of the gate.
            transition_name (str): The name of the transition. Default to 'f01'.
            angle (float, Optional): The angle of the gate.
        """

        if gate_name == 'I':
            return self.get_c1('f01')['I']

        if gate_name == 'qutrit_clifford_green':
            vz1, vz2 = angle

            vz1 *= np.pi * 2 / 3
            vz2 *= np.pi * 2 / 3

            c1_vz = self.get_c1('qudit_vz')
            return c1_vz.z1(vz1) + c1_vz.z2(vz2)

        if gate_name == 'qutrit_clifford_red':
            hadamard = self.get_gate('qutrit_hadamard')
            hadamard_dag = self.get_gate('qutrit_hadamard_dag')
            return hadamard_dag + self.get_gate('qutrit_clifford_green', angle=angle) + hadamard

        if gate_name == 'qutrit_hadamard':
            c1_01 = self.get_c1('f01')
            c1_12 = self.get_c1('f12')
            c1_vz = self.get_c1('qudit_vz')

            magic_angle = np.arccos(1 / (np.sqrt(3))) * 2
            magic_y = c1_01.y(magic_angle)

            hadamard_f12 = c1_12['Ym'] + c1_vz.z2(np.pi)
            qth = hadamard_f12 + c1_vz.z2(np.pi / 2) + c1_vz.z1(np.pi) + magic_y + hadamard_f12

            return qth

        if gate_name == 'qutrit_hadamard_dag':
            c1_01 = self.get_c1('f01')
            c1_12 = self.get_c1('f12')
            c1_vz = self.get_c1('qudit_vz')

            magic_angle = np.arccos(1 / (np.sqrt(3))) * 2
            magic_y = c1_01.y(-magic_angle)

            hadamard_f12 = c1_12['Ym'] + c1_vz.z2(np.pi)
            qth = hadamard_f12 + magic_y + c1_vz.z2(-np.pi / 2) + c1_vz.z1(np.pi) + hadamard_f12

            return qth

        c1 = self.get_c1(transition_name)
        if angle is not None:
            if gate_name == 'Z':
                return c1.z(angle)
            if gate_name == 'X':
                return c1.x(angle)
            if gate_name == 'Y':
                return c1.y(angle)

        if gate_name in ['Xp', 'Xm', 'Yp', 'Ym'] or (gate_name in ['X', 'Y'] and angle is None):
            return c1[gate_name]
        elif gate_name == 'hadamard':
            return c1.hadamard()

        raise ValueError()
