import copy
import uuid

from leeq.core import LeeQObject
from leeq.utils import ObjectFactory, setup_logging
from leeq.utils import elementwise_update_dict
from leeq.core.primitives.base import SharedParameterObject

logger = setup_logging(__name__)


# TODO: Add conditional blocks for fast feedback

class LogicalPrimitiveCombinable(object):

    def __add__(self, other):
        """
        Syntax sugar for combining two logical primitives in series.
        """
        assert issubclass(type(other),
                          LogicalPrimitiveCombinable), f"The other object is not a logical primitive, got {type(other)}."

        if isinstance(other, LogicalPrimitiveBlockSerial):
            other._children.insert(0, self)
            return other

        return LogicalPrimitiveBlockSerial([self, other])

    def __mul__(self, other):
        """
        Syntax sugar for combining two logical primitives in parallel.
        """
        assert issubclass(type(other),
                          LogicalPrimitiveCombinable), f"The other object is not a logical primitive, got {type(other)}."

        if isinstance(other, LogicalPrimitiveBlockParallel):
            other._children.insert(0, self)

        return LogicalPrimitiveBlockParallel([self, other])


class LogicalPrimitive(SharedParameterObject, LogicalPrimitiveCombinable):
    """
    A logical primitive is the basic building block of the experiment sequence.
    """

    def __init__(self, name: str, parameters: dict):
        """
        Initialize the logical primitive by simply saving the name and parameters.

        Parameters:
            name (str): The name of the logical primitive.
            parameters (dict): The parameters of the logical primitive.
        """
        self._validate_parameters(parameters)
        super().__init__(name, parameters)

    def clone(self):
        """
        Clone the object. The returned object is safe to modify.

        Returns:
            SharedParameterObject: The cloned object.
        """
        clone_name = self._name + f'_clone_{uuid.uuid4()}'
        return self.__class__(clone_name, copy.deepcopy(self._parameters))

    @staticmethod
    def _validate_parameters(parameters: dict):
        """
        Validate the parameters of the logical primitive.
        """
        raise NotImplementedError()


class LogicalPrimitiveBlock(LeeQObject, LogicalPrimitiveCombinable):
    """
    A logical primitive block is a tree structure set of logical primitives, composed in series or in parallel.
    Each logical primitive block can be composed of other logical primitive blocks, or logical primitives.
    It acts as the non-leaf node of the tree structure, and the logical primitive acts as the leaf node.
    """

    def __init__(self, name, children=None):
        """
        Initialize the logical primitive block.

        Parameters:
            name (str): The name of the logical primitive block.
            children (list, Optional): The children of the logical primitive block.
        """
        super().__init__(name)
        if children is not None:
            self._children = children
        else:
            self._children = []

    def clone(self):
        """
        Clone the object. The returned object is safe to modify. For a block, the children are also cloned.

        Returns:
            SharedParameterObject: The cloned object.
        """
        clone_name = self._name + f'_clone_{uuid.uuid4()}'

        # Clone the children
        cloned_children = []
        for child in self._children:
            cloned_children.append(child.clone())

        return self.__class__(name=clone_name, children=cloned_children)


class LogicalPrimitiveBlockParallel(LogicalPrimitiveBlock):
    """
    A logical primitive block that is composed in parallel.
    """

    def __init__(self, children=None, name=None):
        """
        Initialize the logical primitive block.
        """
        if name is None:
            name = f"Parallel LPB: {len(children)} : {str(uuid.uuid4())}"
        super().__init__(name, children)

    def __mul__(self, other):
        """
        Syntax sugar for combining two logical primitive blocks in parallel.
        """
        assert isinstance(other,
                          LogicalPrimitiveBlock), f"The other object is not a logical primitive block, got {type(other)}."

        if isinstance(other, LogicalPrimitiveBlockParallel):
            return LogicalPrimitiveBlockParallel(children=self._children + other._children)

        self._children.append(other)
        return self


class LogicalPrimitiveBlockSerial(LogicalPrimitiveBlock):
    """
    A logical primitive block that is composed in serial.
    """

    def __init__(self, children, name=None):
        """
        Initialize the logical primitive block.
        """
        if name is None:
            name = f"SerialLPB: {len(children)} : {str(uuid.uuid4())}"
        super().__init__(name, children)

    def __add__(self, other):
        """
        Syntax sugar for combining two logical primitive blocks in serial.
        """
        assert isinstance(other,
                          LogicalPrimitiveBlock), f"The other object is not a logical primitive block, got {type(other)}."

        if isinstance(other, LogicalPrimitiveBlockSerial):
            return LogicalPrimitiveBlockSerial(self._children + other._children)

        self._children.append(other)
        return self


class LogicalPrimitiveFactory(ObjectFactory):
    """
    The factory class for logical primitive collections.
    """

    def __init__(self):
        super().__init__([LogicalPrimitive])
