import logging

import getpass
import inspect
import sys
from pathlib import Path
import os
from typing import Any

_existing_logger = {}


def setup_logging(name, level=logging.INFO):
    """
    Setup the logging system

    Parameters:
        name (str): The name of the logger
        level (int): The logging level

    Returns:
        logger (logging.Logger): The logger
    """

    if name in _existing_logger:
        return _existing_logger[name]

    # Create the logger, and add handler
    logger = logging.getLogger(name)
    _existing_logger[name] = logger

    logger.setLevel(level)

    # Create the console handler with a recommended format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # Optionally, add a file handler as well
    # file_handler = logger.FileHandler('app.log')
    # file_handler.setLevel(logger.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    # Return the logger
    return logger


logger = setup_logging(__name__)


def get_calibration_log_path() -> Path:
    """
    Get the path to the log file. The path is saved in the environment variable LEEQ_CALIBRATION_LOG_PATH.

    The default path is at calibration_logs/{user_name}/calibration_log_{hrid}_{timestamp}.json
    The username is obtained from the environment variable JUPYTERHUB_USER, your jupyterhub login username.
    If the variable is not found, the username is obtained from getpass.getuser(), which is your Windows
    or Linux login username.

    Returns:
        pathlib.Path: The path to the log file.
    """

    user = os.environ.get(
        "JUPYTERHUB_USER", getpass.getuser()
    )  # Compatible to JupyterHub

    if "LEEQ_CALIBRATION_LOG_PATH" not in os.environ:
        path = Path.cwd() / "calibration_logs"
        logger.warning(
            f"LEEQ_CALIBRATION_LOG_PATH not found in environment variables. Using default path at {path}."
        )
    else:
        path = os.environ["LEEQ_CALIBRATION_LOG_PATH"]
        path = Path(path)

    return path / user


class Singleton(object):
    """
    The singleton class is used to make sure that there is only one instance of the Chronicle class.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the class if it doesn't exist. Otherwise return the existing instance.

        Parameters:
            args (list): The arguments to pass to the class constructor.
            kwargs (dict): The keyword arguments to pass to the class constructor.

        Returns:
            object: The instance of the class.
        """
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the singleton class.
        """
        if self._initialized:
            return

        self._initialized = True


class ObjectFactory(Singleton):
    """
    The factory class base allows registering custom types.
    """

    def __init__(self, accepted_template: list):
        """
        Initialize the factory class.

        Parameters:
            accepted_template (list): The list of accepted types.
        """

        if self._initialized:
            return

        self._registered_template = {}
        self._accepted_template = accepted_template

        super(ObjectFactory, self).__init__()

    def register_collection_template(self, collection_class: Any):
        """
        Register a template to the factory.

        Parameters:
            collection_class (LogicalPrimitiveCollection): The collection template to be registered.
        """

        if not inspect.isclass(collection_class):
            msg = f"The collection class must be a class. Got {collection_class}."
            logger.error(msg)
            raise RuntimeError(msg)

        # Check if collection class is at least one of the accepted types
        if not any(
                [
                    issubclass(collection_class, accepted_type)
                    for accepted_type in self._accepted_template
                ]
        ):
            msg = (
                f"The collection class must be a subclass of at least one of the accepted types. Acceptable:"
                f"{self._accepted_template}. Got: {collection_class}.")
            logger.error(msg)
            raise RuntimeError(msg)

        if collection_class.__qualname__ in self._registered_template:
            return
            msg = f"The collection class {collection_class.__qualname__} has already been registered."
            logger.warning(msg)

        self._registered_template[collection_class.__qualname__] = collection_class

    def __call__(self, class_name: str, *args, **kwargs):
        """
        Create a new instance.

        Parameters:
            class_name (str): The name of the class.
            args (list): The arguments to pass to the class constructor.
            kwargs (dict): The keyword arguments to pass to the class constructor.

        Returns:
            Object: The new instance of the class.
        """

        if class_name not in self._registered_template:
            msg = f"The class {class_name} is not registered."
            logger.error(msg)
            raise RuntimeError(msg)

        collection_class = self._registered_template[class_name]

        return collection_class(*args, **kwargs)


def elementwise_update_dict(original_dict: dict, update_value: dict):
    """
    Elementwise update a dictionary with another dictionary, to propagate the changes to the other
    object that shares the instance.

    Parameters:
        original_dict (dict): The original dictionary.
        update_value (dict): The dictionary to update.
    """
    for key, value in update_value.items():
        if isinstance(value, dict):
            elementwise_update_dict(original_dict[key], value)
        else:
            original_dict[key] = value


def is_running_in_jupyter():
    """
    Check if the code is running in Jupyter notebook.

    Returns:
        bool: True if running in Jupyter notebook.
    """

    # Just a dirty hack, but works for most of the time
    return sys.argv[-1].endswith("json")


def display_json_dict(data: dict, root: str = None):
    """
    Display a dictionary in JSON format. If the program is running in jupyter, we use IPython.display to
    display the dictionary. Otherwise, we use pprint.pprint to print the dictionary.

    Parameters:
        data (dict): The dictionary to display.
        root (str): The root name of the dictionary. If not None, the dictionary will be displayed as
            {root: data}.
    """

    if root is not None:
        root = 'root'

    if is_running_in_jupyter():
        from IPython.display import display, JSON
        display(JSON(data, root=root))
    else:
        import pprint
        pprint.pprint(data)