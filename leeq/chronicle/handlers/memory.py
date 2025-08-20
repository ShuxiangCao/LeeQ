import pathlib
import pickle
from typing import Any, Union
from collections import deque
import numbers

import numpy as np

from .handlers import RecordHandlersBase

from typing import Any, Union
import numbers
from collections import deque

import numpy as np

from .handlers import RecordHandlersBase

from typing import Any, Union
import numbers
from collections import deque

import numpy as np

from .handlers import RecordHandlersBase


class RecordHandlerMemory(RecordHandlersBase):
    """
    An in-memory record handler using nested dictionaries.
    """

    def __init__(self, config: dict):
        """
        Initialize the handler.
        """
        super().__init__(config)
        self.records = {}  # Dictionary to store records
        self.max_records = config.get('max_records', 5)
        self.record_keys = deque(maxlen=self.max_records)  # To track and limit records
        self._initiated = True

    def init_new_record_book(self):
        """
        Initialize a new record book.
        """
        self.records.clear()
        self.record_keys.clear()
        self._initiated = True

    def load_record_book(self):
        """
        Load an existing record book.
        """
        # No action needed for memory-based loading.
        self._initiated = True

    def add_record(self, record_path: str, record: Any):
        """
        Add a record to the memory.

        Parameters:
            record_path (str): A unique identifier for the record, structured as a nested path.
            record (Any): The record to add.
        """
        self._check_initiated()

        if not isinstance(record_path, str):
            record_path = str(record_path)

        # Navigate or create the nested dictionary structure based on the path
        path_parts = record_path.split('/')
        current_dict = self.records
        for part in path_parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]

        # Set the record at the final location in the path
        current_dict[path_parts[-1]] = record

        # Manage records to limit the number stored
        self.record_keys.append(record_path)
        if len(self.record_keys) > self.max_records:
            oldest_key = self.record_keys.popleft()
            self._remove_record(oldest_key)

    def get_record_by_path(self, record_path: str):
        """
        Get a record by its path.

        Parameters:
            record_path (str): The path to the record.

        Returns:
            Any: The record if found, otherwise None.
        """
        self._check_initiated()
        parts = record_path.split('/')
        current_dict = self.records
        for part in parts:
            if part in current_dict:
                current_dict = current_dict[part]
            else:
                return None
        return current_dict

    def _remove_record(self, record_path: str):
        """
        Remove a record by its path.
        """
        parts = record_path.split('/')
        current_dict = self.records
        for part in parts[:-1]:
            if part in current_dict:
                parent_dict = current_dict
                current_dict = current_dict[part]
            else:
                return  # The path does not exist

        # Remove the last part of the path if it exists
        if parts[-1] in current_dict:
            del parent_dict[parts[-1]]

    def list_records(self, record_path: Union[pathlib.Path, str] = None) -> list:
        """
        List all records' paths.

        Returns:
            list: A list of record paths.
        """
        self._check_initiated()
        return list(self.record_keys)
