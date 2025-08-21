# This file contains the core functionality of the labchronicle package.
# 1. The `Chronicle` class is the main class of the package. It manages the log path, multiple different handler.
# It usually is used as a singleton.
# 2. The `LogRecord` class is used to represent a specific log. Providing the abstract interface for the handlers.
# 3. The `LogHandler` class is the abstract class for all the handlers. It
# provides the interface for the handlers.
import copy
import inspect
import json
import pathlib
import pickle
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .handlers import RecordHandlersBase, get_handler
from .logger import setup_logging
from .utils import find_methods_with_tag, get_system_info

logger = setup_logging(__name__)

_reserved_keys = [
    "_loggable",
    "_register_log_and_record_args_map",
    "logger",
    "_record_entry",
    "_browse_functions",
    "__dict__",
]


class SetBrowserFunctionAttributeMeta(type):
    """
    A metaclass that set class attributes from the base class to the subclass,
    making sure they do not share the same reference.
    """

    def __new__(mcls, name, bases, class_dict):
        new_class = super().__new__(mcls, name, bases, class_dict)
        new_class._browse_functions = []
        return new_class


class LoggableObject(metaclass=SetBrowserFunctionAttributeMeta):
    """
    This is the base class for all classes that want to be logged.
    Any new attribute added to the loggable objects will be recorded, and when `record` a function all,
    the entire object (alone with the attributes) will be saved as a checkpoint.

    The way loggable object records the attributes is by using the `__setattr__` method. The attribute ``_loggable`` is
    reserved to indicate the attribute to log.

    For the attributes that are not manually set, the loggable object will pickle them. For the records that are
    captured by the `__setattr__` method, the loggable object will record them in a separate file/record.
    """

    def __init__(self):
        self._register_log_and_record_args_map = {}
        self.logger = setup_logging(self.hrid)
        self._record_entry = None

    @property
    def hrid(self):
        """
        Get the human readable id of the loggable object. Default to the class name.
        Override this method to provide a more meaningful id.

        Returns:
            str: The human readable id of the loggable object.
        """
        return self.__class__.__qualname__ + "@" + str(id(self))

    def __setattr__(self, key, value):
        """
        Override the `__setattr__` method to record the attributes.

        Parameters:
            key (str): The name of the attribute.
            value (Any): The value of the attribute.
        """

        if key in _reserved_keys:
            # If the key is reserved, simply set the attribute.
            super().__setattr__(key, value)
            return

        if "_record_entry" not in self.__dict__ or self._record_entry is None:
            # If the record entry is not in the dict, it means that the object
            # is not being monitored.
            super().__setattr__(key, value)
            return

        self._record_entry.touch_attribute(key)
        super().__setattr__(key, value)

    def __repr__(self):
        """
        Override the `__repr__` method to provide a more meaningful representation of the loggable object.

        Returns:
            str: The representation of the loggable object.
        """

        return f"<{self.hrid}>"

    def __getattribute__(self, key):
        """
        Override the `__getattribute__` method to record the attributes.
        """

        # Make sure we use the original __getattribute__ method, to avoid infinite recursion
        # that may be induced by the child class overriding the
        # __getattribute__ method.
        get_attribute = super().__getattribute__

        result = get_attribute(key)

        if key in _reserved_keys:
            return result

        if (
                "_record_entry" not in get_attribute("__dict__")
                or get_attribute("_record_entry") is None
        ):
            # If the record entry is not in the dict, it means that the object
            # is not being monitored.
            return result

        if inspect.isroutine(result):
            return result

        get_attribute("_record_entry").touch_attribute(key)

        return result

    @staticmethod
    def _safe_deepcopy(obj):
        """
        Safely deepcopy an object.

        Parameters:
            obj (object): The object to deepcopy.

        Returns:
            object: The copied object.
        """
        return copy.deepcopy(obj)

    def get_browser_functions(self):
        """
        Get the browser functions of the loggable object.

        Returns:
            list: The browser functions of the loggable object.
        """
        return find_methods_with_tag(self, "_browser_function")

    def register_log_and_record_args(
            self,
            func: callable,
            args: list,
            kwargs: dict,
            record_details: dict,
            deepcopy: bool = True,
            overwrite_func_name=None
    ):
        """
        Register the arguments of the function.

        Parameters:
            func (function): The function to register the arguments from.
            args (list): The arguments of the function.
            kwargs (dict): The keyword arguments of the function.
            record_details (dict): The latest record entry details of the function call.
            deepcopy (bool): Whether to deepcopy the arguments.
            overwrite_func_name (str): Optional. The name of the function to be recorded. If not specified, the name of
                                        the function will be used.

        Returns:
            dict: The arguments of the function.
        """
        if deepcopy:
            args = self._safe_deepcopy(args)
            kwargs = self._safe_deepcopy(kwargs)

        name = overwrite_func_name if overwrite_func_name is not None else func.__qualname__

        self._register_log_and_record_args_map[name] = (
            args,
            kwargs,
            record_details,
        )

    @staticmethod
    def _rebuild_args_dict(
            func: Callable[..., Any], called_args: List[Any], called_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reconstruct the arguments dictionary for a given function based on its signature and the called arguments.

        This method fetches the signature of the function and tries to match the provided called arguments and
        keyword arguments to build the true arguments dictionary for the function call.

        Note that we have removed "self" from the called args, so be careful to remove it when it's a class method.

        Parameters:
        - func (Callable[..., Any]): The function for which the arguments dictionary needs to be built.
        - called_args (List[Any]): List of arguments with which the function was called.
        - called_kwargs (Dict[str, Any]): Dictionary of keyword arguments with which the function was called.

        Returns:
        - Dict[str, Any]: Dictionary containing the true arguments for the function call.

        Raises:
        - Exception: If there is a mismatch between the function's default arguments and its signature.
        """
        sig = inspect.signature(func)
        parameters = list(sig.parameters.values())

        mapped_args = {}

        # Remove "self" from the parameters if it's a class method
        if parameters[0].name == "self" or parameters[0].name == "cls":
            parameters = parameters[1:]

        # First, populate with defaults and arguments provided
        for param in parameters:
            if param.default != param.empty:  # if a default is provided
                mapped_args[param.name] = param.default
            else:
                mapped_args[param.name] = None

        # For positional arguments
        for param, value in zip(parameters, called_args):
            mapped_args[param.name] = value

        # For keyword arguments (this will override any values set before)
        mapped_args.update(called_kwargs)

        return mapped_args

    def retrieve_args(
            self,
            func: callable,
    ):
        """
        Retrieve the arguments of the function.

        Parameters:
            func (function): The function to retrieve the arguments from.

        Returns:
            dict: The arguments of the function.
        """

        if func.__qualname__ not in self._register_log_and_record_args_map:
            msg = f"Function {func.__qualname__} is not registered in {self.hrid}. "
            self.logger.error(msg)
            raise ValueError(msg)

        val = self._register_log_and_record_args_map[func.__qualname__]

        return self._rebuild_args_dict(func, val[0], val[1])

    def retrieve_latest_record_entry_details(self, func: callable):
        """
        Retrieve the latest record entry details of the function call.

        Parameters:
            func (function): The function to retrieve the arguments from.

        Returns:
            dict: The latest record entry details of the function call.
        """
        if func.__qualname__ not in self._register_log_and_record_args_map:
            msg = f"Function {func.__qualname__} is not registered in {self.hrid}. "
            self.logger.error(msg)
            raise ValueError(msg)

        return self._register_log_and_record_args_map[func.__qualname__][2]

    def set_record_entry(self, record_entry: Optional["RecordEntry"] = None):
        """
        Set the record entry of the loggable object. When record entry is set, the loggable object will start to
        record the attributes to the record entry.

        Parameters:
            record_entry (RecordEntry): The record entry of the loggable object.
        """
        self._record_entry = record_entry


class RecordBook(object):
    """
    A tree-like data structure that stores the records. Each node is a `RecordEntry` object.
    Each record book corresponds to a database (for example, a hdf5 file).
    """

    def __init__(self, config: dict, enable_write: bool = False):
        """
        Initialize the RecordBook class.

        Parameters:
            config (dict): The config of the record book.
            enable_write (bool): Whether to enable write to the record book.
        """

        self._config = config
        self._handler: RecordHandlersBase = get_handler(
            config["handler"])(config)
        self._enable_write = enable_write

        if not enable_write:
            # If it is read only, load the record book.
            self._handler.load_record_book()
            system_info_json = self._handler.get_record_by_path("system_info")
            self._system_info = json.loads(system_info_json)
        else:
            # If it is a new record book, initialize it.
            self._handler.init_new_record_book()
            self._system_info = get_system_info()
            system_info_json = json.dumps(self._system_info)
            self._handler.add_record("system_info", system_info_json)

        self._root_entry = RecordEntry(
            record_book=self,
            timestamp=0,
            record_id="root",
            record_order=0,
            base_path="/",
        )

        self._root_entry.set_name("root")

    def get_path(self):
        """
        Get the path of the record book.

        Returns:
            pathlib.Path: The path of the record book.
        """
        return pathlib.Path(self._config["log_path"])

    def get_record_by_path(self,
                           path: Union[pathlib.Path,
                                       str]) -> "RecordEntry":
        """
        Get a record by its path.

        Parameters:
            path (pathlib.Path or str): The path to the record.

        Returns:
            Any: The record.
        """

        if isinstance(path, str):
            path = pathlib.Path(path)

        return RecordEntry(full_path=path, record_book=self)

    def get_record_by_id(self, record_id: str):
        """
        Get a record by its id.

        Parameters:
            record_id (str): The id of the record.

        Returns:
            Any: The record.
        """
        path = self._handler.get_record_by_path(f"/uuid/{str(record_id)}").decode()
        return self.get_record_by_path(path)

    def get_available_record_ids(self):
        """
        Get the available record ids.

        Returns:
            list: The available record ids.
        """
        return self._handler.list_records("/uuid")

    @property
    def handler(self):
        """
        Get the handler of the record book.

        Returns:
            RecordHandlersBase: The handler of the record book.
        """
        return self._handler

    def get_root_entry(self):
        """
        Get the root entry of the record book.

        Returns:
            RecordEntry: The root entry of the record book.
        """
        return self._root_entry

    def get_start_time(self):
        """
        Get the start time of the record book.

        Returns:
            int: The start time of the record book.
        """
        return int(self._system_info["start_time"])


class RecordEntry(object):
    """
    The RecordEntry class is used to represent a specific log.

    Each record entry usually denote a specific function call. Contains the time, function name, arguments, and
    return values. It also contains the snapshot of the loggable class of the function, at the time of the function call.

    """

    def __init__(
            self,
            record_book: RecordBook,
            timestamp: Optional[int] = None,
            record_id: Optional[str] = None,
            record_order: Optional[int] = None,
            base_path: Optional[pathlib.Path] = None,
            full_path: Optional[pathlib.Path] = None,
    ):
        """
        Initialize the RecordEntry class.

        Parameters:
            timestamp (int): The timestamp of the record entry.
            record_id (str): The id of the record entry.
            record_order (int): The order of the record entry.
            base_path (pathlib.Path): The base path of the record entry.
            record_book (RecordBook): The record book of the record entry.
        """
        self._timestamp = timestamp
        self._record_id = record_id
        self._record_order = record_order
        self._name = None
        self._record_book = record_book
        self._base_path = base_path
        self._touched_attributes = []

        if (
                timestamp is None
                or record_id is None
                or record_order is None
                or base_path is None
        ):
            # If the record entry is not initiated, it means that it is loading
            # a record entry.
            assert full_path is not None
            self._base_path = full_path

            self._load_from_path(full_path)

    def _load_from_path(self, path: pathlib.Path):
        """
        Load the record entry from the path.

        Parameters:
            path (pathlib.Path): The path to the record entry.
        """

        if path == pathlib.Path("/root"):
            self._record_order, self._name = 0, "root"
            self._base_path = "/root"
            return

        path_name = path.name
        path_name_split = path_name.split('\\')
        if len(path_name_split) > 1:
            self._base_path = pathlib.Path('\\'.join(path_name_split[:-1]))
        else:
            self._base_path = path.parent
        path_name = path_name_split[-1]
        splits = path_name.split("-", 1)

        if len(splits) != 2:
            msg = f"Invalid path {path}."
            logger.error(msg)
            raise ValueError(msg)

        self._record_order, self._name = int(splits[0]), splits[1]
        self._load_metadata()

    @property
    def record_book(self):
        """
        Get the record book that the record entry belongs to.

        Returns:
            RecordBook: The record book of the record entry.
        """
        return self._record_book

    @property
    def record_time(self):
        """
        Get the record time of the record entry.

        Returns:
            int: The record time of the record entry.
        """
        return self._timestamp + self._record_book.get_start_time()

    def _check_initiated(self):
        """
        Check whether the record entry is initiated.

        Raises:
            ValueError: If the record entry is not initiated.
        """

        if self._name is None:
            msg = "Please set the name of the record entry first."
            logger.error(msg)
            raise ValueError(msg)

        if self._record_book.handler is None:
            msg = "Please initialize the record book first."
            logger.error(msg)
            raise ValueError(msg)

    def get_path(self):
        """
        Get the path of the record entry.

        Returns:
            pathlib.Path: The path of the record entry.
        """

        if self._name == "root":
            return pathlib.Path("/root")

        self._check_initiated()

        return self._base_path / (str(self._record_order) + "-" + self._name)

    def _get_attribute_path(self, key: str):
        """
        Get the path of the attribute.

        Parameters:
            key (str): The key of the attribute.

        Returns:
            pathlib.Path: The path of the attribute.
        """
        return self.get_path() / key

    def touch_attribute(self, key: str):
        """
        Touch an attribute of the record entry, and set it as dirty.
        The dirty attribute will be recorded separately when the record entry is saved.
        This function should only be called by the loggable objects, set_attr method.

        Parameters:
            key (str): The key of the attribute.
        """
        if key not in self._touched_attributes:
            self._touched_attributes.append(key)

    def set_name(self, name: str):
        """
        Set the name of the record entry, usually corresponds to the functional qualname.

        Parameters:
            name (str): The name of the record entry.
        """
        self._name = name

    def record_object(self, obj: LoggableObject):
        """
        Record the loggable object.

        Parameters:
            obj (LoggableObject): The loggable object to record.
        """
        # Save all the dirty attributes
        for attr in self._touched_attributes:
            self.save_attribute(attr, getattr(obj, attr))

        # Save the object itself
        self.save_attribute("__object__", obj)
        self.save_attribute("__touched_attributes__", self._touched_attributes)

    def record_return_values(self, return_values: Any):
        """
        Record the return values of the function.

        Parameters:
            return_values (Any): The return values of the function.
        """
        self.save_attribute("__return_values__", return_values)

    def record_error_info(self, error_info: Any):
        """
        Record the error info of the function.

        Parameters:
            error_info (Any): The error info of the function.
        """
        self.save_attribute("__error_info__", error_info)

    def load_return_values(self):
        """
        Load the return values of the function.

        Returns:
            Any: The return values of the function.
        """
        return self.load_attribute("__return_values__")

    def record_args(self, args: list, kwargs: dict):
        """
        Record the arguments of the function.

        Parameters:
            args (list): The arguments of the function.
            kwargs (dict): The keyword arguments of the function.
        """
        self.save_attribute("__args__", args)
        self.save_attribute("__kwargs__", kwargs)

    def record_metadata(self):
        """
        Record the metadata of the record entry.
        """
        self.save_attribute("__timestamp__", self._timestamp)
        self.save_attribute("__record_id__", self._record_id)
        self.save_attribute("__name__", self._name)

    def _load_metadata(self):
        """
        Load the metadata of the record entry.
        """
        self._timestamp = self.load_attribute("__timestamp__")
        self._record_id = self.load_attribute("__record_id__")
        self._name = self.load_attribute("__name__")
        self._touched_attributes = self.load_attribute(
            "__touched_attributes__")

    def get_object(self):
        """
        Get the loggable object.

        Returns:
            LoggableObject: The loggable object.
        """
        return self.load_attribute("__object__")

    def get_attribute(self, key: str):
        """
        Get an attribute of the record entry.

        Parameters:
            key (str): The key of the attribute.

        Returns:
            Any: The value of the attribute.
        """
        assert (
            key in self._touched_attributes
        ), f"Attribute {key} is not recorded. Please try access through the object."
        return self.load_attribute(key)

    def get_recorded_attribute_names(self):
        """
        Get the recorded attributes of the record entry.

        Returns:
            list: The recorded attributes of the record entry.
        """
        return self._touched_attributes

    def get_args(self):
        """
        Get the arguments of the function.

        Returns:
            tuple: The arguments of the function.
        """

        args = self.load_attribute("__args__")
        kwargs = self.load_attribute("__kwargs__")

        return args, kwargs

    def save_attribute(self, key: str, val: Any):
        """
        Save an attribute of the record entry.

        Parameters:
            key (str): The key of the attribute.
            val (Any): The value of the attribute.
        """

        self._check_initiated()

        path = self._get_attribute_path(key)
        self._record_book.handler.add_record(path, val)

    def load_attribute(self, key: str):
        """
        Load an attribute of the record entry.

        Parameters:
            key (str): The key of the attribute.

        Returns:
            Any: The value of the attribute.
        """

        self._check_initiated()

        path = self._get_attribute_path(key)
        loaded_data = self._record_book.handler.get_record_by_path(path)

        if isinstance(loaded_data, bytes):
            loaded_data = loaded_data.decode()

        if isinstance(loaded_data, np.void):
            try:
                loaded_data = pickle.loads(loaded_data.tobytes())
            except Exception as e:
                msg = f'Pickle loading failed:{e}'
                loaded_data = None
                logger.warning(msg)

        return loaded_data

    def get_children_number(self):
        """
        Get the number of children of the record entry.

        Returns:
            int: The number of children of the record entry.
        """
        return len(self._get_children_names())

    def _get_children_names(self):
        """
        Get the children names of the record entry.

        Returns:
            list: The children names of the record entry.
        """
        list_of_names = self._record_book.handler.list_records(self.get_path())

        # Sub record names always start with a number, while attributes cannot.
        # TODO: this search is not efficient, may significantly slow down when a same routine get repeatedly executed,
        #  should be improved.
        children_names = [
            name for name in list_of_names if name[0] in "0123456789"]

        return children_names

    @property
    def children(self):
        """
        Get the children of the record entry.

        Returns:
            list: The children of the record entry.
        """

        children_names = self._get_children_names()

        return [
            RecordEntry(record_book=self._record_book, full_path=self.get_path() / name)
            for name in children_names
        ]

    @property
    def parent(self):
        """
        Get the parent of the record entry.

        Returns:
            RecordEntry: The parent of the record entry.
        """
        return RecordEntry(
            record_book=self._record_book, full_path=self.get_path().parent
        )

    @property
    def record_id(self):
        """
        Get the id of the record entry.

        Returns:
            str: The id of the record entry.
        """
        return self._record_id

    @property
    def timestamp(self):
        """
        Get the timestamp of the record entry.

        Returns:
            int: The timestamp of the record entry.
        """
        return self._timestamp

    @property
    def record_order(self):
        """
        Get the order of the record entry.

        Returns:
            int: The order of the record entry.
        """
        return self._record_order

    @property
    def base_path(self):
        """
        Get the base path of the record entry.

        Returns:
            pathlib.Path: The base path of the record entry.
        """
        return self._base_path

    @property
    def name(self):
        """
        Get the name of the record entry.

        Returns:
            str: The name of the record entry.
        """
        return self._name

    def load_all_attributes(self):
        """
        Load all the attributes of the record entry.

        Returns:
            dict: The attributes of the record entry.
        """
        attr_dict = {
            attr: self.load_attribute(attr) for attr in self.get_recorded_attribute_names()
        }

        args_retval_dict = {
            "__args__": self.load_attribute("__args__"),
            "__kwargs__": self.load_attribute("__kwargs__"),
            "__return_values__": self.load_attribute("__return_values__"),
        }

        attr_dict.update(args_retval_dict)

        return attr_dict
