import pathlib
from typing import Any, Union

from .handlers import RecordHandlersBase


class RecordHandlerDummy(RecordHandlersBase):
    """
    The handler that does not save anything.
    """

    def init_new_record_book(self):
        """
        Initialize a new record book.
        """
        self._initiated = True

    def load_record_book(self):
        """
        Load an existing record book.
        """
        self._initiated = True

    def add_record(self, record_path: Union[pathlib.Path, str], record: Any):
        """
        Add a record to the database.

        Parameters:
            record_path (pathlib.Path): The path to the record.
            record (Any): The record to add.
        """

        self._check_initiated()

    def get_record_by_path(self, record_path: Union[pathlib.Path, str]):
        """
        Get a record by its path.

        Parameters:
            record_path (pathlib.Path or str): The path to the record.

        Returns:
            Any: The record.
        """
        self._check_initiated()

        raise NotImplementedError("This handler does not save any records.")

    def list_records(self, record_path: Union[pathlib.Path, str]) -> list:
        """
        List all the records under the given path.

        Parameters:
            record_path (pathlib.Path or str): The path to the record.

        Returns:
            pathlib.Path: A list of records.
        """
        self._check_initiated()
        raise NotImplementedError("This handler does not save any records.")
