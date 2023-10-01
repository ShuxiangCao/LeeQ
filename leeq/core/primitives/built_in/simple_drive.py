import numpy as np

from leeq.core.primitives.logical_primitives import LogicalPrimitive, LogicalPrimitiveFactory
from leeq.core.primitives.collections import LogicalPrimitiveCollection


class SimpleDrive(LogicalPrimitive):
    _parameter_names = ['freq', 'channel', 'shape', 'amp', 'phase', 'width', 'alpha', 'trunc']

    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)

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
            assert parameter_name in parameters, f"The parameter {parameter_name} is not found."


class SimpleDriveCollection(LogicalPrimitiveCollection):

    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)

        primitive_params = {
            'drive': 'SimpleDrive'
        }

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

        if item == 'X':
            return self._primitives['drive']
        elif item == 'Y':
            return self._primitives['drive'].copy_with_parameters({'phase': np.pi / 2}, name_postfix='_Y')
        elif item == 'Xp':
            return self._primitives['drive'].copy_with_parameters({'amp': self._parameters['amp'] / 2},
                                                                  name_postfix='_Xp')
        elif item == 'Yp':
            return self._primitives['drive'].copy_with_parameters(
                {'amp': self._parameters['amp'] / 2, 'phase': np.pi / 2}, name_postfix='_Yp')
        elif item == 'Xm':
            return self._primitives['drive'].copy_with_parameters({'amp': -self._parameters['amp'] / 2},
                                                                  name_postfix='_Xm')
        elif item == 'Ym':
            return self._primitives['drive'].copy_with_parameters(
                {'amp': -self._parameters['amp'] / 2, 'phase': np.pi / 2}, name_postfix='_Ym')

        return super(SimpleDriveCollection).__getitem__(item)
