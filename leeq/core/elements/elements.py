import json
import pathlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from leeq.core.base import LeeQObject
from leeq.core.primitives import LogicalPrimitiveCollectionFactory, LogicalPrimitiveFactory
from leeq.utils import display_json_dict, get_calibration_log_path, setup_logging
from leeq.utils.utils import setup_logging

logger = setup_logging(__name__)


logger = setup_logging(__name__)


class CalibrationEncoder(json.JSONEncoder):
    """
    The calibration encoder is used to encode the calibration dictionary to json.
    """

    def default(self, obj):
        if callable(obj):
            return obj.__repr__()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


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
    _time_format_str = "%Y_%m_%d_%H_%M_%S"

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
                "lpb_collections": {},
                "measurement_primitives": {}}

        self._validate_parameters(self._parameters)

        self._lpb_collections = {}
        self._measurement_primitives = {}

        self._build_lpb_collections()
        self._build_measurement_primitives()

    @staticmethod
    def _validate_parameters(parameters: dict):
        """
        Validate the parameters of the element.

        Parameters:
            parameters (dict): The parameters of the element.
        """
        assert (
            "lpb_collections" in parameters
        ), "LPB collections not found in the parameters."
        assert (
            "measurement_primitives" in parameters
        ), "Measurement primitives not found in the parameters."

    def _build_lpb_collections(self):
        """
        Initiate the gate collections of the element.
        """
        factory = LogicalPrimitiveCollectionFactory()
        for collection_name, collection_parameters in self._parameters[
            "lpb_collections"
        ].items():
            self._lpb_collections[collection_name] = factory(
                name=self._name + "." + collection_name,
                class_name=collection_parameters["type"],
                parameters=collection_parameters,
            )

        factory = LogicalPrimitiveFactory()
        for collection_name, collection_parameters in self._parameters["measurement_primitives"].items():
            collection_name = "m" + collection_name
            self._lpb_collections[collection_name] = factory(
                name=self._name + "." + collection_name,
                class_name=collection_parameters["type"],
                parameters=collection_parameters,
            )

    def _build_measurement_primitives(self):
        """
        Initiate the measurement primitives of the element.
        """

        factory = LogicalPrimitiveFactory()
        for primitive_name, primitive_parameters in self._parameters[
            "measurement_primitives"
        ].items():
            self._measurement_primitives[primitive_name] = factory(
                name=self._name + ".measurement." + str(primitive_name),
                class_name=primitive_parameters["type"],
                parameters=primitive_parameters,
            )

    def _dump_dict(self, parameter_dict: dict):
        """
        Dump the element to dictionary. It recursively dumps the gate collections and measurement primitives, ingore
        the parameters that starts with _.

        Returns:
            dict: The element.
        """
        dumped_dict = {}
        for key, value in parameter_dict.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict):
                dumped_dict[key] = self._dump_dict(value)
            else:
                dumped_dict[key] = value

        return dumped_dict

    def _dump_lpb_collections(self):
        """
        Dump the gate collections of the element to dictionary.

        Returns:
            dict: The gate collections of the element.
        """
        return self._dump_dict(self._parameters["lpb_collections"])

    def _dump_measurement_primitives(self):
        """
        Dump the measurement primitives of the element to dictionary.

        Returns:
            dict: The measurement primitives of the element.
        """
        return self._dump_dict(self._parameters["measurement_primitives"])

    def get_calibrations(self):
        """
        Get the calibration dictionary of the element.
        """
        calibration_log = {
            "lpb_collections": self._dump_lpb_collections(),
            "measurement_primitives": self._dump_measurement_primitives(),
        }
        return calibration_log

    def save_calibration_log(self):
        """
        Save the calibration log of the element.
        """

        calibration_log = self.get_calibrations()

        path = get_calibration_log_path()

        if not path.exists():
            path.mkdir(parents=True)

        file_name = f"calibration_log_{self.hrid}.{datetime.now().strftime(self._time_format_str)}.json"

        path = path / file_name

        with open(path, "w") as f:
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
            msg = f"Calibration log not found at {base_path}."
            logger.error(msg)
            raise FileNotFoundError(msg)

        latest_file_name = None
        latest_time = None

        for path in base_path.iterdir():
            file_name = pathlib.Path(path).name
            splits = str(file_name).split(".")

            if len(splits) != 3:
                continue

            prefix_read, timestr, postfix = splits

            prefix = "calibration_log_" + name

            if postfix != "json" or not prefix_read.endswith(prefix):
                continue

            try:
                parsed_time = datetime.strptime(timestr, cls._time_format_str)
            except ValueError:
                logger.info(f"Invalid time format {timestr}.")

            if latest_time is None or parsed_time > latest_time:
                latest_time = parsed_time
                latest_file_name = file_name

        if latest_file_name is None:
            msg = f"No calibration log found for {name} at {base_path}."
            logger.error(msg)
            raise FileNotFoundError(msg)

        return base_path / latest_file_name

    @classmethod
    def load_from_calibration_log(
            cls, name: str, path: Optional[Union[str, Path]] = None
    ):
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
                    raise FileNotFoundError(
                        f"Calibration log not found at {path}.")
        else:
            # If the path is not specified, iterate through the base directory and find the latest calibration log
            # with the name specified.
            path = cls._find_latest_calibration_log(name)

        with open(path, "r") as f:
            calibration_log = json.load(f)

        logger.info(f'Calibration log loaded from {path}')

        return cls(name, calibration_log)

    @staticmethod
    def _validate_calibration_dict(calibration: dict):
        """
        Validate the calibration dictionary.

        Parameters:
            calibration (dict): The calibration dictionary.
        """

        assert (
            "lpb_collections" in calibration
        ), "LPB collections not found in the calibration dictionary."
        assert (
            "measurement_primitives" in calibration
        ), "Measurement primitives not found in the calibration dictionary."

    def get_lpb_collection(self, name: str):
        """
        Get the gate collection with the specified name.

        Parameters:
            name (str): The name of the gate collection.

        Returns:
            collection (LogicalPrimitiveCollection): The gate collection.
        """
        return self._lpb_collections[name]

    def get_measurement_primitive(self, name: str):
        """
        Get the measurement primitive with the specified name.

        Parameters:
            name (str): The name of the measurement primitive.

        Returns:
            primitive (LogicalPrimitive): The measurement primitive.
        """

        if name not in self._measurement_primitives:
            name = str(name)
            if name not in self._measurement_primitives:
                raise KeyError(f"Measurement primitive {name} not found.")

        return self._measurement_primitives[name]

    def get_c1(self, name: str):
        """
        Same as get lpb_collection. To ensure compatibility with the old version.
        """
        return self.get_lpb_collection(name)

    def print_config_info(self):
        """
        Print the configuration information of the element.
        """
        from IPython.display import JSON, display

        calibrations = self.get_calibrations()

        display_json_dict(
            json.loads(
                CalibrationEncoder().encode(calibrations)),
            root=f'Element {self._name} parameters')
