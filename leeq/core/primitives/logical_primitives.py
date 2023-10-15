import copy
import uuid

import numpy
from labchronicle import log_event

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
        assert issubclass(
            type(other), LogicalPrimitiveCombinable
        ), f"The other object is not a logical primitive, got {type(other)}."

        if isinstance(other, LogicalPrimitiveBlockSerial):
            other._children.insert(0, self)
            other._uuid_to_nodes.update(self.nodes)
            return other

        return LogicalPrimitiveBlockSerial([self, other])

    def __mul__(self, other):
        """
        Syntax sugar for combining two logical primitives in parallel.
        """
        assert issubclass(
            type(other), LogicalPrimitiveCombinable
        ), f"The other object is not a logical primitive, got {type(other)}."

        if isinstance(other, LogicalPrimitiveBlockParallel):
            other._children.insert(0, self)
            other._uuid_to_nodes.update(self.nodes)
            return other

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
        self._tags = {}

    def __getattribute__(self, item):
        """
        Get the value of the parameter.

        Parameters:
            item (str): The name of the parameter.

        Returns:
            The value of the parameter.

        Raises:
            AttributeError: If the parameter is not found.
        """

        reserved_names = ["_parameters", "__dict__"]

        get_attribute = super(LogicalPrimitive, self).__getattribute__

        if item not in reserved_names:
            if "_parameters" in get_attribute("__dict__"):
                if item in get_attribute("_parameters"):
                    return get_attribute("_parameters")[item]

        return get_attribute(item)

    def clone(self):
        """
        Clone the object. The returned object is safe to modify.

        Returns:
            SharedParameterObject: The cloned object.
        """
        clone_name = self._name + f"_clone_{uuid.uuid4()}"
        return self.__class__(clone_name, copy.deepcopy(self._parameters))

    def shallow_copy(self):
        """
        Copy the logical primitive without copying the parameters. Updating
        the parameters of the copied logical primitive will also update the
        parameters of the original logical primitive, and the full element
        configuration. The tags will not be copied.

        Returns:
            LogicalPrimitive: The copied logical primitive.
        """
        return self.__class__(self._name, self._parameters)

    def copy_with_parameters(self, parameters: dict, name_postfix=None):
        """
        Copy the logical primitive with new parameters.

        Parameters:
            parameters (dict): The new parameters.
            name_postfix (str, Optional): The postfix of the name of the copied logical primitive.

        Returns:
            LogicalPrimitive: The copied logical primitive.
        """

        new_parameters = copy.deepcopy(self._parameters)
        elementwise_update_dict(new_parameters, parameters)
        if name_postfix is None:
            name_postfix = f"_modified_{uuid.uuid4()}"
        return self.__class__(self._name + name_postfix, new_parameters)

    @staticmethod
    def _validate_parameters(parameters: dict):
        """
        Validate the parameters of the logical primitive.
        """
        raise NotImplementedError()

    @log_event
    def tag(self, **kwargs):
        """
        Add a tag to the logical primitive. This is for the user to add additional information to the logical primitive.
        Especially for the compiler to use.

        Parameters:
            kwargs: The tags to be added to the logical primitive.
        """
        self._tags.update(kwargs)
        return self

    @property
    def tags(self):
        """
        Get the tags of the logical primitive.

        Returns:
            dict: The tags of the logical primitive.
        """
        return self._tags.copy()

    @property
    def children(self):
        """
        Get the children of the logical primitive.

        Returns:
            list: The children of the logical primitive.
        """
        return None

    def get_parameters(self):
        """
        Get the parameters of the logical primitive.

        Returns:
            dict: The parameters of the logical primitive.
        """
        params = copy.deepcopy(self._parameters)
        return params

    @property
    def nodes(self):
        """
        Get the nodes of the logical primitive.
        """
        return {self.uuid: self}


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

        self._uuid_to_nodes = {}
        for child in children:
            self._uuid_to_nodes.update(child.nodes)

    def clone(self):
        """
        Clone the object. The returned object is safe to modify. For a block, the children are also cloned.

        Returns:
            SharedParameterObject: The cloned object.
        """
        clone_name = self._name + f"_clone"

        # Clone the children
        cloned_children = []
        for child in self._children:
            cloned_children.append(child.clone())

        return self.__class__(name=clone_name, children=cloned_children)

    @property
    def children(self):
        """
        Get the children of the logical primitive block.

        Returns:
            list: The children of the logical primitive block.
        """
        return self._children

    @property
    def nodes(self):
        """
        Get the nodes of the logical primitive block.
        """
        return self._uuid_to_nodes


class LogicalPrimitiveBlockSweep(LogicalPrimitiveBlock):
    """
    A logical primitive block that is composed in serial.
    """

    def __init__(self, children=None, name=None):
        """
        Initialize the logical primitive block.
        """
        if name is None:
            name = f"Sweep LPB: {len(children)} : {str(uuid.uuid4())}"
        super().__init__(name, children)
        self._selected = 0

    def set_selected(self, selected):
        """
        Set the selected child of the logical primitive block.
        The selected child is the child that will be executed.

        Parameters:
            selected (int): The index of the selected child.
        """
        self._selected = selected

    @property
    def selected(self):
        """
        Get the selected child of the logical primitive block.
        The selected child is the child that will be executed.

        Returns:
            int: The index of the selected child.
        """
        return self._selected

    @property
    def current_lpb(self):
        """
        Get the current logical primitive block.

        Returns:
            LogicalPrimitiveBlock: The current logical primitive block.
        """
        return self._children[self._selected]

    def __len__(self):
        """
        Get the number of children.
        """
        return len(self._children)

    def children(self):
        """
        Get the children of the logical primitive block. For sweep block are the selected block.

        Returns:
            list: The children of the logical primitive block.
        """
        return [self._children[self._selected]]

    @property
    def nodes(self):
        """
        Get the nodes of the logical primitive block. For sweep block are the selected block.
        """
        return self._children[self._selected].nodes


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
        assert isinstance(
            other, LogicalPrimitiveCombinable
        ), f"The other object is not a logical primitive or a block, got {type(other)}."

        if isinstance(other, LogicalPrimitiveBlockParallel):
            return LogicalPrimitiveBlockParallel(
                children=self._children + other._children
            )

        self._children.append(other)
        self._uuid_to_nodes.update(other.nodes)
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
        assert isinstance(
            other, LogicalPrimitiveCombinable
        ), f"The other object is not a logical primitive block, got {type(other)}."

        if isinstance(other, LogicalPrimitiveBlockSerial):
            return LogicalPrimitiveBlockSerial(
                self._children + other._children)

        self._children.append(other)
        self._uuid_to_nodes.update(other.nodes)
        return self


class LogicalPrimitiveFactory(ObjectFactory):
    """
    The factory class for logical primitive collections.
    """

    def __init__(self):
        super().__init__([LogicalPrimitive])


class MeasurementPrimitive(LogicalPrimitive):
    """
    A measurement primitive is a logical primitive that is used to perform measurements.
    """

    def __init__(self, name: str, parameters: dict):
        """
        Initialize the measurement primitive.

        Parameters:
            name (str): The name of the measurement primitive.
            parameters (dict): The parameters of the measurement primitive.
        """
        super().__init__(name, parameters)

        self._results = []
        self._results_raw = []
        self._transform_function = None
        self._transform_function_kwargs = None
        self._default_result_id = 0
        self._result_id_offset = 0

        self._sweep_shape = None  # The shape of the sweep in this experiment
        self._number_of_measurements = None  # The max number of repeated use of mprims in this experiment

        # The measurement buffer for storing the measurement results
        self._raw_measurement_buffer = None

        # The measurement buffer after applying the transformation function (For example, GMM for state discrimination)
        self._transformed_measurement_buffer = None

    @staticmethod
    def _validate_parameters(parameters: dict):
        """
        Validate the parameters of the measurement primitive.
        """
        raise NotImplementedError()

    def is_buffer_allocated(self):
        """
        Check whether the measurement buffer is allocated.

        Returns:
            bool: Whether the measurement buffer is allocated.
        """
        return self._raw_measurement_buffer is not None

    def allocate_measurement_buffer(self, sweep_shape: list, number_of_measurements: int, data_shape: list,
                                    dtype: type = numpy.complex128):
        """
        Allocate the measurement buffer for the measurement primitive.

        Parameters:
            sweep_shape (list): The shape of the sweep.
            number_of_measurements: The max number of measurement repeated in a shot, for all measurement primitives.
            data_shape (list): The shape of the measurement data.
            dtype (type, Optional): The data type of the measurement buffer.
        """

        if self.is_buffer_allocated():
            msg = (f"Measurement buffer already allocated for {self._name}. It seems you are running the same "
                   f"experiment multiple times. Please create another experiment instance for this purpose.")
            logger.error(msg)
            raise RuntimeError(msg)

        shape = sweep_shape + number_of_measurements + data_shape
        self._number_of_measurements = number_of_measurements
        self._sweep_shape = sweep_shape

        self._raw_measurement_buffer = numpy.zeros(shape, dtype=dtype)
        self._transformed_measurement_buffer = numpy.zeros(shape, dtype=dtype)

    @log_event
    def set_transform_function(self, func: callable, **kwargs):
        """
        Set the transform function of the measurement primitive.
        When a measurement finish, the transform function will be called to transform the measurement result.
        For instance, the transform function can be implemented as using a GMM model to distinguish the measurement
        state, and return the distinguished label instead of raw data.

        When set to None, raw data will be returned.

        Parameters:
            func (callable): The transform function.
            kwargs: The keyword arguments of the transform function.
        """

        self._transform_function = func
        self._transform_function_kwargs = kwargs

    def get_transform_function(self):
        """
        Get the transform function of the measurement primitive.

        Returns:
            callable: The transform function.
            dict: The keyword arguments of the transform function.
        """
        return self._transform_function, self._transform_function_kwargs

    def result(self, result_id: int = None, raw_data: bool = False):
        """
        Retrieve the result of the measurement primitive.
        The result is the output of the transform function, available after the measurement is finished.

        Parameters:
            result_id (int, Optional): The id of the measurement result.
            raw_data (bool, Optional): Whether to return the raw data (the data before transformation).

        Note :
            Now we move to the concept of result ids. The result ids are used to identify the measurement results.
            When the measurement primitive is supplied into an experiment multiple times, then the first measurement
            result will have id 0, the second will have id 1, and so on. When retrieving the result, the result id
            will be used to identify the result.
        """

        if result_id is None:
            result_id = self._default_result_id

        result_data = self._transformed_measurement_buffer if not raw_data else self._raw_measurement_buffer

        total_dim = len(self._raw_measurement_buffer.shape)

        max_result_id = self._raw_measurement_buffer.shape[-2]

        if not self.is_buffer_allocated():
            msg = f"No measurement result is available for {self._name}."
            logger.error(msg)
            raise RuntimeError(msg)

        if result_id + self._result_id_offset >= max_result_id:
            msg = (
                f"The result id {result_id} is out of range, "
                f"the maximum result id is {max_result_id - 1 - self._result_id_offset}."
                f"Current result id offset is {self._result_id_offset}")
            logger.error(msg)
            raise ValueError(msg)

        slice_idx = tuple(
            slice(None) if dim != total_dim - 2 else result_id + self._result_id_offset
            for dim in range(total_dim)
        )

        return result_data[slice_idx]

    def set_default_result_id(self, result_id: int):
        """
        Set the default result id of the measurement primitive.
        The default result id is the id of the measurement result that will be returned when multiple results are
        available, but no result id is specified. The default result id is 0.

        The default result id is used when the measurement primitive is supplied into an experiment multiple times, and
        the user wants to retrieve the result of a specific measurement. The rest of the measurement could be just
        active reset.

        Parameters:
            result_id (int): The default result id.
        """
        self._default_result_id = result_id

    def get_default_result_id(self):
        """
        Get the default result id of the measurement primitive.

        Returns:
            int: The default result id.
        """
        return self._default_result_id

    def set_result_id_offset(self, offset: int):
        """
        Set the result id offset of the measurement primitive.

        The result id offset is used to offset the result id of the measurement primitive. For instance, if the result
        id offset is 1, then the first measurement result will have id 1, the second will have id 2, and so on. This
        is particularly useful when the user wants to have multiple active reset at the beginning, and then start a
        complex experiment. With this offset the user will not have to change the result id of the measurement in the
        experiment script.

        Parameters:
            offset (int): The result id offset.
        """
        self._result_id_offset = offset

    def get_result_id_offset(self):
        """
        Get the result id offset of the measurement primitive.

        Returns:
            int: The result id offset.
        """
        return self._result_id_offset

    def result_ids(self):
        """
        Get the ids of the measurement results.

        Returns:
            list: The ids of the measurement results.
        """

        return [key - self._result_id_offset for key in range(self._raw_measurement_buffer.shape[-2])]

    def result_raw(self, result_id: int = None):
        """
        Retrieve the raw data of the measurement primitive. Same as `result` with `raw_data` set to True.
        For compatibility with the old version.

        Parameters:
            result_id (int, Optional): The id of the measurement result.

        Returns:
            The raw data of the measurement result.
        """
        return self.result(result_id, raw_data=True)

    # def clear_results(self):
    #    """
    #    Clear the measurement results of the measurement primitive.
    #    """
    #    self._results = []
    #    self._results_raw = []

    def commit_measurement(self, indices: tuple, data: numpy.ndarray):
        """
        Commit a measurement result to the measurement primitive.

        The data is supplied in the following shape:
        [sweep index, result_id,  data]

        The result id denotes the id of multiple measurement happened in the same shot, For instance, if the same
        measurement primitive is used twice in the same shot, then the first measurement result will have id 0,
        the second will have id 1.

        Parameters:
            data (numpy.ndarray): The measurement result.
            indices (tuple): The indices of the measurement result.
        """

        if not self.is_buffer_allocated():
            msg = f"Measurement buffer not allocated for {self._name}. Please allocate the buffer first."
            logger.error(msg)
            raise RuntimeError(msg)

        data_raw = data

        if self._transform_function is not None:
            data_transformed = self._transform_function(
                data, **self._transform_function_kwargs
            )
        else:
            data_transformed = data

        self._transformed_measurement_buffer[indices] = data_transformed
        self._raw_measurement_buffer[indices] = data_raw


class MeasurementPrimitiveFactory(ObjectFactory):
    """
    The factory class for measurement primitives.
    """

    def __init__(self):
        super().__init__([MeasurementPrimitive])
