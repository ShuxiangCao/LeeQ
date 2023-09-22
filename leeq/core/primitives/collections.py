from leeq.core import LeeQObject
from leeq.utils import ObjectFactory, setup_logging
from leeq.core.primitives.logical_primitives import LogicalPrimitiveFactory
from leeq.core.primitives.base import SharedParameterObject

logger = setup_logging(__name__)


class LogicalPrimitiveCollection(SharedParameterObject):
    """
    Base class for all logical primitive collections.

    Gate collection is a concept for grouping a set of logical primitives that shares some parameters together.
    For example, a lpb collection can be both the X and Y gates of a transmon qubit, which share the same
    frequency and amplitude. Sometimes it also includes the gates that are not directly related to the qubit,
    for example the X,Y,Z together, for convenience.
    """

    def __init__(self, name: str, parameters: dict):
        """
        Initialize the gate collection.

        Parameters:
            name (str): The name of the lpb collection.
            parameters (dict): The parameters of the lpb collection.
        """

        super().__init__(name, parameters)
        self._primitives = {}

    def _build_primitives(self):
        """
        Build the logical primitives of the collection.
        """
        factory = LogicalPrimitiveFactory()
        for primitive_name, primitive_type in self._parameters['primitives'].items():
            self._primitives[primitive_name] = factory(primitive_type, self._parametersp)

    def __getitem__(self, item):
        """
        Get the logical primitive by name.

        Parameters:
            item (str): The name of the logical primitive.

        Returns:
            LogicalPrimitive: The logical primitive.

        Raises:
            KeyError: If the logical primitive is not found.
        """
        if item not in self._primitives:
            msg = f"The logical primitive {item} is not found."
            logger.error(msg)
            raise KeyError(msg)

        return self._primitives[item]


class LogicalPrimitiveCollectionFactory(ObjectFactory):
    """
    The factory class for logical primitive collections.
    """

    def __init__(self):
        super().__init__([LogicalPrimitiveCollection])
