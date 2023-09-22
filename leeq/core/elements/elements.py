import json
from datetime import datetime
from pathlib import Path
from typing import Union, Optional

from labchronicle import LoggableObject
from leeq.core import LeeQObject
from leeq.utils import get_calibration_log_path


class Element(LeeQObject):
    """
    Base class for all quantum elements, such as a transmon qubit.

    It saves the calibration configuration, and initiate the gate collections for further use.

    Note that the calibration dictionary is passed by share to the gate collections and measurement primitives.
    Therefore, modification of the calibration dictionary will affect all the gate collections and measurement,
    primitives, for the entire collection, and will be reflected in the new saved calibration log.

    If you wish to modify the calibration dictionary for a specific gate collection or measurement primitive,
    without touching the calibration dictionary for the entire element, you should use the `clone` method to
    duplicate the gate collection or measurement primitive, and modify the calibration dictionary of the
    duplicated object.
    """

    # The time format string for the calibration log file name
    _time_format_str = "%Y.%m.%d.%H.%M.%S"

    def __init__(self, name: str, parameters: dict = None):
        """
        Initialize the element. The name will be used as the human readable id.
        Parameters are stored in the class to derive the lpb collections and measurement primitives.
        If the parameters are not specified, an calibration with no parameters will be used.

        Parameters:
            name (str): The name of the element.
            parameters (dict, Optional): The parameters of the element.
        """
        super().__init__(name)

        if parameters is not None:
            self._parameters = parameters
        else:
            self._parameters = {
                'lpb_collections': {},
                'measurement_primitives': {}
            }

        self._lpb_collections = {}
        self._measurement_primitives = {}

        self._build_lpb_collections()
        self._build_measurement_primitives()

    def _build_lpb_collections(self):
        """
        Initiate the gate collections of the element.
        """
        raise NotImplementedError()

    def _build_measurement_primitives(self):
        """
        Initiate the measurement primitives of the element.
        """
        raise NotImplementedError()

    def _dump_lpb_collections(self):
        """
        Dump the gate collections of the element to dictionary.

        Returns:
            dict: The gate collections of the element.
        """
        raise NotImplementedError()

    def _dump_measurement_primitives(self):
        """
        Dump the measurement primitives of the element to dictionary.

        Returns:
            dict: The measurement primitives of the element.
        """
        raise NotImplementedError()

    def save_calibration_log(self):
        """
        Save the calibration log of the element.
        """

        calibration_log = {
            'gate_collections': self._dump_gate_collections(),
            'measurement_primitives': self._dump_measurement_primitives()
        }

        path = get_calibration_log_path()

        if not path.exists():
            path.mkdir(parents=True)

        file_name = f'calibration_log_{self.hrid}.{datetime.now().strftime(self._time_format_str)}.json'

        path = path / file_name

        with open(path, 'w') as f:
            json.dump(calibration_log, f)

    @classmethod
    def _find_latest_calibration_log(cls, name: str):
        """
        Find the latest calibration log with the specified name.

        Parameters:
            name (str): The name of the element.

        Returns:
            path (Path): The path to the calibration log.
        """

        base_path = get_calibration_log_path()

        if not base_path.exists():
            msg = f'Calibration log not found at {base_path}.'
            cls.logger.error(msg)
            raise FileNotFoundError(msg)

        latest_file_name = None
        latest_time = None

        for file_name in base_path.iterdir():

            splits = str(file_name).split('.')

            if len(splits) != 3:
                continue

            prefix_read, timestr, postfix = splits

            prefix = 'calibration_log_' + name

            if postfix != 'json' or prefix != prefix_read:
                continue

            try:
                parsed_time = datetime.strptime(timestr, cls.time_format_str)
            except ValueError:
                continue

            if latest_time is None or parsed_time > latest_time:
                latest_time = parsed_time
                latest_file_name = file_name

        if latest_file_name is None:
            msg = f'No calibration log found for {name} at {base_path}.'
            cls.logger.error(msg)
            raise FileNotFoundError(msg)

        return base_path / latest_file_name

    @classmethod
    def load_from_calibration_log(cls, name: str, path: Optional[Union[str, Path]]):
        """
        Load the calibration log and generate the element object.
        If the path is not specified, load the latest calibration log.

        Parameters:
            name (str): The name of the element.
            path (str or Path, Optional): The path to the calibration log.

        Returns:
            element (Element): The element object.

        Raise:
            FileNotFoundError: If the specified calibration log is not found.
                    Or no calibration log corresponds to the name is found.
        """

        base_directory = get_calibration_log_path()

        if path is not None:
            if isinstance(path, str):
                path = Path(path)

            # First check if the path is the absolute path
            if not path.exists():
                # If not, check if the path is relative to the base directory
                path = base_directory / path
                if not path.exists():
                    raise FileNotFoundError(f'Calibration log not found at {path}.')
        else:
            # If the path is not specified, iterate through the base directory and find the latest calibration log
            # with the name specified.
            path = cls._find_latest_calibration_log(name)

        with open(path, 'r') as f:
            calibration_log = json.load(f)

        return cls(name, calibration_log)

    @staticmethod
    def _validate_calibration_dict(calibration: dict):
        """
        Validate the calibration dictionary.

        Parameters:
            calibration (dict): The calibration dictionary.
        """

        assert 'gate_collections' in calibration, 'Gate collections not found in the calibration dictionary.'
        assert 'measurement_primitives' in calibration, \
            'Measurement primitives not found in the calibration dictionary.'
