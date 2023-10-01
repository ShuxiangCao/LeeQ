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

        factory = LogicalPrimitiveCollectionFactory()
        factory.register_collection_template(SimpleDriveCollection)

        # Build the element
        super().__init__(name, parameters)

    def _validate_parameters(self, parameters: dict):
        """
        Validate the parameters of the element.

        Parameters:
            parameters (dict): The parameters of the element.
        """

        from leeq.core.primitives.built_in.simple_drive import SimpleDriveCollection

        for name,lpb_parameter in parameters['lpb_collections'].items():
            assert 'type' in lpb_parameter, 'The type of the lpb collection is not specified.'
            assert lpb_parameter['type'] in [SimpleDriveCollection.__qualname__], \
                f"The lpb collection {lpb_parameter['name']} is not supported."

        for name,measurement_parameter in parameters['measurement_primitives'].items():
            assert 'name' in measurement_parameter, 'The name of the measurement parameter is not specified.'
            assert measurement_parameter['name'] in ['simple_drive_measurement']
