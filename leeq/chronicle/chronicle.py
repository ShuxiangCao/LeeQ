import copy
import datetime
import os
import pathlib
import uuid
from contextlib import contextmanager
from typing import Optional, Union

import yaml

from .core import RecordBook, RecordEntry
from .logger import setup_logging
from .utils import get_log_path

logger = setup_logging(__name__)


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


class Chronicle(Singleton):
    """
    The Chronicle class is the main class of the package. It manages the log path, multiple different handler.
    It is used as a singleton.
    """

    def __init__(self, config: dict = None, config_path: Optional[Union[pathlib.Path, str]] = None):
        """
        Initialize the Chronicle class.

        Parameters:
            config (dict): Optional. The configuration dictionary.
            config_path (str): Optional. The path to the configuration file.
        """

        if self._initialized:
            return

        self._config = self.load_config(config=config, config_path=config_path)
        self._active_record_book = None
        self._record_tracking_stack = None
        self._log_start_time = None

        super().__init__()

    def load_config(self, config_path: Optional[str] = None, config: dict = None):
        """
        Load the configuration file. If the configuration file is not specified, load the configuration from
        environment variables. If environment variables are not specified, load the default configuration.

        Parameters:
            config_path (str): Optional. The path to the configuration file.
            config (dict): Optional. The configuration dictionary.

        When both config and config_path are provided, the config with usee the dictionary provided in config
        and update it with the values.
        """

        _config_update = config if config is not None else {}

        # Load the configuration from the file
        if config_path is not None:
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            self._validate_config(config)

        else:
            config = {}

        config.update(_config_update)

        # Load the configuration from environment variables
        if "LAB_CHRONICLE_LOG_DIR" in os.environ and "log_path" not in config:
            config["log_path"] = os.environ["LAB_CHRONICLE_LOG_DIR"]
        if "LAB_CHRONICLE_HANDLER" in os.environ and "handlers" not in config:
            config["handler"] = os.environ["LAB_CHRONICLE_HANDLER"]

        # Load the default configuration
        if "log_path" not in config:
            config["log_path"] = "./log"
        if "handler" not in config:
            config["handler"] = "hdf5"

        # Validate the configuration
        self._validate_config(config)

        return config

    @staticmethod
    def _validate_config(config: dict):
        """
        Validate the configuration file.

        Parameters:
            config (dict): The configuration dictionary.
        """

        if not isinstance(config, dict):
            raise ValueError(
                "Invalid configuration file. Expected a dictionary, got a {type(config)} instead."
            )

        if "log_path" not in config:
            raise ValueError(
                "log_path is not specified in the configuration file. Please specify the"
                " LAB_CHRONICLE_LOG_DIR environment variable.")
        if "handler" not in config:
            raise ValueError(
                "handlers is not specified in the configuration file. Please specify the"
                " LAB_CHRONICLE_HANDLER environment variable.")

    def start_log(self, name: str = None):
        """
        Start a new log.

        Parameters:
            name (str): Optional. The name of the log.

        """
        if name is None:
            name = ""

        path = get_log_path(pathlib.Path(self._config["log_path"]), name=name)

        record_book_config = copy.deepcopy(self._config)
        record_book_config["log_path"] = path.as_posix()

        self._active_record_book = RecordBook(
            enable_write=True, config=record_book_config
        )
        self._record_tracking_stack = [
            self._active_record_book.get_root_entry()]
        self._log_start_time = self._active_record_book.get_start_time()

        logger.info(f"Log started at {path}")

    @contextmanager
    def new_record(self):
        """
        Create a new record. Use `with` statement with this function to create a new record.
        """

        if self._record_tracking_stack is None:
            logger.warning("No active log. Execution not recorded.")
            try:
                yield None
            finally:
                pass
            return

        record_timestamp = (
            int(datetime.datetime.now().timestamp()) - self._log_start_time
        )

        if len(self._record_tracking_stack) == 0:
            msg = "Log records compromised."
            logger.error(msg)
            raise ValueError(msg)
        else:
            record_order = self._record_tracking_stack[-1].get_children_number()
            record_path = self._record_tracking_stack[-1].get_path()

        new_record = RecordEntry(
            timestamp=record_timestamp,
            record_id=str(uuid.uuid4()),
            record_book=self._active_record_book,
            record_order=record_order,
            base_path=record_path,
        )

        self._record_tracking_stack.append(new_record)

        try:
            yield new_record
        finally:
            last_record = self._record_tracking_stack.pop()

            # Write the link of uuid to the path
            self._active_record_book.handler.add_record(
                f"/uuid/{last_record.record_id}", str(last_record.get_path())
            )

    def end_log(self):
        """
        End the current log.
        """
        self._active_record_book = None
        self._record_tracking_stack = None

    def open_record_book(
            self, path: Optional[Union[pathlib.Path, str]] = None):
        """
        Open a record book.

        Parameters:
            path (str): Optional. The path to the record book. If not specified, use the default path.
        """
        if path is None:
            path = self._config["log_path"]

        record_book_config = copy.deepcopy(self._config)
        record_book_config["log_path"] = path

        return RecordBook(enable_write=False, config=record_book_config)

    def is_recording(self) -> bool:
        """
        Return if currently an active record book is recording.
        Returns
            bool: True if an active record is recording, otherwise false.
        """
        return self._active_record_book is not None

    def launch_viewer(self, **kwargs):
        """
        Launch chronicle viewer dashboard for current session.

        This method launches a web-based dashboard for monitoring experiments
        in the active Chronicle session. The dashboard polls every 5 seconds
        to display newly completed experiments.

        Args:
            debug (bool): Whether to run in debug mode (default: True)
            port (int): Port to run the server on (default: 8051)
            **kwargs: Additional arguments passed to the Dash server

        Example:
            from leeq.chronicle import Chronicle
            chronicle = Chronicle()
            chronicle.start_log("my_session")
            chronicle.launch_viewer()  # Opens dashboard at localhost:8051
        """
        from leeq.chronicle.viewer.session_dashboard import start_viewer

        arguments = {
            'chronicle_instance': self,
            'debug': kwargs.get('debug', True),
            'port': kwargs.get('port', 8051)
        }
        arguments.update(kwargs)
        start_viewer(**arguments)

    def _get_record_by_path_or_id(self, record_book_path: Union[pathlib.Path, str], record_id: str = None,
                                  record_entry_path: Union[pathlib.Path, str] = None):
        """
        A helper function to get the record entry by either the record id or the record entry path.
        Note that record_id and record_entry_path cannot be both None.

        Parameters:
            record_book_path (str): The path to the record book.
            record_id (str): Optional. The id of the record entry.
            record_entry_path (str): Optional. The path to the record entry.

        Returns (RecordEntry): The record entry.
        """
        if record_id is None and record_entry_path is None:
            msg = "record_id and record_entry_path cannot be both None."
            logger.error(msg)
            raise ValueError(msg)

        record_book = self.open_record_book(record_book_path)
        if record_id is not None:
            record = record_book.get_record_by_id(record_id)
        else:
            record = record_book.get_record_by_path(record_entry_path)

        return record

    def load_attributes(self, record_book_path: Union[pathlib.Path, str], record_id: str = None,
                        record_entry_path: Union[pathlib.Path, str] = None):
        """
        A shortcut for loading all the attributes of a record entry.
        Note that record_id and record_entry_path cannot be both None.

        Parameters:
            record_book_path (str): The path to the record book.
            record_id (str): Optional. The id of the record entry.
            record_entry_path (str): Optional. The path to the record entry.


        Returns (dict): The dictionary of the record entry.
        """
        record = self._get_record_by_path_or_id(
            record_book_path, record_id, record_entry_path
        )

        return record.load_all_attributes()

    def load_object(self, record_book_path: Union[pathlib.Path, str], record_id: str = None,
                    record_entry_path: Union[pathlib.Path, str] = None):
        """
        A shortcut for loading all the attributes of a record entry.
        Note that record_id and record_entry_path cannot be both None.

        Parameters:
            record_book_path (str): The path to the record book.
            record_id (str): Optional. The id of the record entry.
            record_entry_path (str): Optional. The path to the record entry.
        """
        record = self._get_record_by_path_or_id(
            record_book_path, record_id, record_entry_path
        )

        return record.get_object()


def load_object(record_book_path: Union[pathlib.Path, str], record_id: str = None,
                record_entry_path: Union[pathlib.Path, str] = None):
    """
    A shortcut for loading all the attributes of a record entry.
    Note that record_id and record_entry_path cannot be both None.

    Parameters:
        record_book_path (str): The path to the record book.
        record_id (str): Optional. The id of the record entry.
        record_entry_path (str): Optional. The path to the record entry.

    Returns (LoggableObject): The logged object.
    """

    return Chronicle().load_object(record_book_path, record_id, record_entry_path)


def load_attributes(record_book_path: Union[pathlib.Path, str], record_id: str = None,
                    record_entry_path: Union[pathlib.Path, str] = None):
    """
    A shortcut for loading all the attributes of a record entry.
    Note that record_id and record_entry_path cannot be both None.

    Parameters:
        record_book_path (str): The path to the record book.
        record_id (str): Optional. The id of the record entry.
        record_entry_path (str): Optional. The path to the record entry.


    Returns (dict): The dictionary of the record entry.
    """
    return Chronicle().load_attributes(record_book_path, record_id, record_entry_path)
