# This file contains the abstract definition of the handlers.
from typing import Any, Union
import pathlib
from labchronicle.logger import setup_logging


class RecordHandlersBase(object):
    """
    The abstract class for all the handlers. It provides the interface for the handlers.
    """

    def __init__(self, config: dict):
        """
        Initialize the handler.

        Parameters:
            config (dict): The configuration dictionary.
        """
        self._config = config
        self._initiated = False
        self._logger = setup_logging(self.__class__.__qualname__)

    def _check_initiated(self):
        if not self._initiated:
            msg = "Record book is not initiated. Please call init_new_record_book() or load_record_book() first."
            self._logger.error(msg)
            raise RuntimeError(msg)

    def init_new_record_book(self):
        """
        Initialize a new record book.
        """
        raise NotImplementedError()

    def load_record_book(self):
        """
        Load an existing record book.
        """
        raise NotImplementedError()

    def add_record(self, record_path: Union[pathlib.Path, str], record: Any):
        """
        Add a record to the database.

        Parameters:
            record_path (pathlib.Path or str): The path to the record.
            record (Any): The record to add.
        """
        raise NotImplementedError()

    def get_record_by_path(self, record_path: Union[pathlib.Path, str]):
        """
        Get a record by its path.

        Parameters:
            record_path (pathlib.Path or str): The path to the record.

        Returns:
            Any: The record.
        """
        raise NotImplementedError()

    def get_record_by_id(self, record_id: str):
        """
        Get a record by its id.

        Parameters:
            record_id (pathlib.Path): The path to the record.

        Returns:
            Any: The record.
        """
        raise NotImplementedError()

    def get_record_by_timestamp(self, record_time: int):
        """
        Get a record by its timestamp.

        Parameters:
            record_time (str): The timestamp of the record.

        Returns:
            Any: The record.
        """
        raise NotImplementedError()

    def list_records(self, record_path: Union[pathlib.Path, str]) -> list:
        """
        List all the records under the given path.

        Parameters:
            record_path (pathlib.Path or str): The path to the record.

        Returns:
            list: A list of records.
        """
        raise NotImplementedError()
