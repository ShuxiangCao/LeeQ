"""
EPII Parameter Management

This module handles parameter management for LeeQ setups through
the EPII interface, mapping between EPII and LeeQ parameter types.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from leeq.setups.setup_base import ExperimentalSetup

logger = logging.getLogger(__name__)


class ParameterManager:
    """
    Manages LeeQ setup parameters through the EPII interface.

    Provides CRUD operations for setup parameters with type safety
    and validation. Maps between EPII parameter names (dot notation)
    and LeeQ setup/element parameters.
    """

    # Type mapping from Python types to EPII parameter types
    TYPE_MAP = {
        bool: "bool",
        int: "int",
        float: "float",
        str: "string",
        list: "array",
        dict: "object",
        type(None): "null"
    }

    # Reverse mapping for type conversion
    EPII_TO_PYTHON = {
        "bool": bool,
        "int": int,
        "float": float,
        "string": str,
        "array": list,
        "object": dict,
        "null": type(None)
    }

    def __init__(self, setup: Optional[ExperimentalSetup] = None):
        """
        Initialize the parameter manager with a LeeQ setup.

        Args:
            setup: LeeQ ExperimentalSetup instance
        """
        self.setup = setup
        self.parameter_cache: Dict[str, Any] = {}
        self._readonly_params = {
            "setup.name", "setup.type", "setup.active",
            "engine.status", "compiler.type"
        }

    def get_parameter(self, name: str) -> Optional[Any]:
        """
        Get a parameter value from the LeeQ setup.

        Args:
            name: Parameter name in dot notation (e.g., "status.shot_number")

        Returns:
            Parameter value or None if not found
        """
        if not self.setup:
            return self.parameter_cache.get(name)

        # Parse dot notation
        parts = name.lower().split(".")

        # Handle setup status parameters
        if parts[0] == "status" and len(parts) == 2:
            try:
                return self.setup.status.get_parameters(parts[1])
            except KeyError:
                logger.warning(f"Parameter {name} not found in setup status")
                return None

        # Handle element parameters (e.g., "q0.f01")
        elif hasattr(self.setup, "_elements") and parts[0] in self.setup._elements:
            element = self.setup._elements[parts[0]]
            if len(parts) == 2 and hasattr(element, "_parameters"):
                return element._parameters.get(parts[1])

        # Handle direct setup attributes
        elif parts[0] == "setup" and len(parts) == 2:
            attr_name = f"_{parts[1]}"
            if hasattr(self.setup, attr_name):
                return getattr(self.setup, attr_name)

        # Fallback to cache
        return self.parameter_cache.get(name)

    def set_parameter(self, name: str, value: Any) -> bool:
        """
        Set a parameter value in the LeeQ setup.

        Args:
            name: Parameter name in dot notation
            value: New parameter value

        Returns:
            True if successful, False otherwise
        """
        # Check if parameter is read-only
        if name.lower() in self._readonly_params:
            logger.warning(f"Parameter {name} is read-only")
            return False

        if not self.setup:
            self.parameter_cache[name] = value
            return True

        # Parse dot notation
        parts = name.lower().split(".")

        # Handle setup status parameters
        if parts[0] == "status" and len(parts) == 2:
            try:
                # Convert value type if needed
                converted_value = self._convert_value(value, parts[1])
                self.setup.status.set_parameter(parts[1], converted_value)
                return True
            except Exception as e:
                logger.error(f"Failed to set parameter {name}: {e}")
                return False

        # Handle element parameters
        elif hasattr(self.setup, "_elements") and parts[0] in self.setup._elements:
            element = self.setup._elements[parts[0]]
            if len(parts) == 2 and hasattr(element, "_parameters"):
                element._parameters[parts[1]] = value
                return True

        # Store in cache if not found in setup
        self.parameter_cache[name] = value
        return True

    def list_parameters(self) -> List[Dict[str, Any]]:
        """
        List all available parameters with their metadata.

        Returns:
            List of parameter descriptions with name, type, value, etc.
        """
        parameters = []

        if not self.setup:
            # Return cached parameters if no setup
            for name, value in self.parameter_cache.items():
                parameters.append({
                    "name": name,
                    "type": self._get_type_string(value),
                    "current_value": self._serialize_value(value),
                    "description": f"Cached parameter {name}",
                    "read_only": False
                })
            return parameters

        # Add setup status parameters
        if hasattr(self.setup, "status"):
            status_params = self.setup.status.get_parameters()
            for key, value in status_params.items():
                param_name = f"status.{key}"
                parameters.append({
                    "name": param_name,
                    "type": self._get_type_string(value),
                    "current_value": self._serialize_value(value),
                    "description": self._get_param_description(key),
                    "read_only": param_name.lower() in self._readonly_params
                })

        # Add element parameters if available
        if hasattr(self.setup, "_elements"):
            for elem_name, element in self.setup._elements.items():
                if hasattr(element, "_parameters"):
                    # Add calibration parameters
                    for key, value in element._parameters.items():
                        if isinstance(value, (int, float, str, bool)):
                            param_name = f"{elem_name}.{key}"
                            parameters.append({
                                "name": param_name,
                                "type": self._get_type_string(value),
                                "current_value": self._serialize_value(value),
                                "description": f"{elem_name} {key} parameter",
                                "read_only": False
                            })

        # Add cached parameters not in setup
        for name, value in self.parameter_cache.items():
            if not any(p["name"] == name for p in parameters):
                parameters.append({
                    "name": name,
                    "type": self._get_type_string(value),
                    "current_value": self._serialize_value(value),
                    "description": f"Custom parameter {name}",
                    "read_only": False
                })

        return parameters

    def validate_parameter(self, name: str, value: Any) -> bool:
        """
        Validate a parameter value before setting.

        Args:
            name: Parameter name
            value: Value to validate

        Returns:
            True if valid, False otherwise
        """
        # Check read-only
        if name.lower() in self._readonly_params:
            logger.warning(f"Parameter {name} is read-only")
            return False

        # Type validation for known parameters
        parts = name.lower().split(".")

        # Validate setup status parameters
        if parts[0] == "status" and len(parts) == 2:
            param_key = parts[1]

            # Special validation for specific parameters
            if param_key == "shot_number":
                if not isinstance(value, (int, str)):
                    return False
                try:
                    val = int(value) if isinstance(value, str) else value
                    return val > 0 and val <= 1000000
                except (ValueError, TypeError):
                    return False

            elif param_key == "shot_period":
                if not isinstance(value, (float, int, str)):
                    return False
                try:
                    val = float(value) if isinstance(value, str) else value
                    return val > 0
                except (ValueError, TypeError):
                    return False

            elif param_key in ["debug_plotter", "plot_result_in_jupyter",
                              "amp_warning", "ignore_plot_error"]:
                return isinstance(value, (bool, str))

        # Element parameter validation
        elif hasattr(self.setup, "_elements") if self.setup else False:
            if parts[0] in self.setup._elements and len(parts) == 2:
                param_key = parts[1]

                # Validate frequency parameters
                if param_key in ["f01", "f12", "anharmonicity"]:
                    try:
                        val = float(value) if isinstance(value, str) else value
                        return isinstance(val, (float, int)) and val > 0
                    except (ValueError, TypeError):
                        return False

                # Validate time parameters
                elif param_key in ["t1", "t2", "t2_echo"]:
                    try:
                        val = float(value) if isinstance(value, str) else value
                        return isinstance(val, (float, int)) and val > 0
                    except (ValueError, TypeError):
                        return False

                # Validate amplitude parameters
                elif param_key in ["pi_amp", "pi2_amp"]:
                    try:
                        val = float(value) if isinstance(value, str) else value
                        return isinstance(val, (float, int)) and 0 <= val <= 1
                    except (ValueError, TypeError):
                        return False

        # Default: accept any value
        return True

    def _get_type_string(self, value: Any) -> str:
        """Get EPII type string for a Python value."""
        for py_type, epii_type in self.TYPE_MAP.items():
            if isinstance(value, py_type):
                return epii_type
        return "object"  # Default for unknown types

    def _serialize_value(self, value: Any) -> str:
        """Serialize a value to string for EPII transmission."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return value
        elif isinstance(value, (list, dict)):
            return json.dumps(value)
        else:
            return str(value)

    def _convert_value(self, value: Union[str, Any], param_key: str) -> Any:
        """Convert string value to appropriate Python type based on parameter."""
        if not isinstance(value, str):
            return value

        # Try to infer type from current value if setup exists
        if self.setup and hasattr(self.setup.status, "_internal_dict"):
            current = self.setup.status._internal_dict.get(param_key.lower())
            if current is not None:
                target_type = type(current)
                try:
                    if target_type == bool:
                        return value.lower() in ("true", "1", "yes")
                    elif target_type == int:
                        return int(value)
                    elif target_type == float:
                        return float(value)
                    elif target_type == str:
                        return value
                except (ValueError, TypeError):
                    pass

        # Try JSON parsing for complex types
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Try basic conversions
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        elif value.lower() == "null":
            return None

        try:
            # Try int first
            if "." not in value and "e" not in value.lower():
                return int(value)
            # Then float
            return float(value)
        except ValueError:
            # Return as string if all else fails
            return value

    def _get_param_description(self, key: str) -> str:
        """Get human-readable description for a parameter."""
        descriptions = {
            "measurement_basis": "Measurement basis (<z>, <zs>, prob(0), etc.)",
            "shot_number": "Number of measurement shots per data point",
            "shot_period": "Time between measurement shots (μs)",
            "acquisition_type": "Type of data acquisition (IQ, IQ_average, traces)",
            "debug_plotter": "Enable pulse sequence visualization",
            "debug_plotter_ignore_readout": "Ignore readout in pulse plotter",
            "in_jupyter": "Running in Jupyter notebook environment",
            "plot_result_in_jupyter": "Display plots in Jupyter output",
            "amp_warning": "Show amplitude warning messages",
            "ignore_plot_error": "Continue on plotting errors",
            "resourceautoreleasetime": "Auto-release timeout (seconds)",
            "resourceautorelease": "Enable automatic resource release",
            "disablesweepprogressbar": "Disable sweep progress bar",
            "leavesweepprogressbar": "Keep progress bar after completion",
            "sweepprogressbardesc": "Progress bar description text",
            "globalprelpb": "Global pre-experiment logical primitive block",
            "globalpostlpb": "Global post-experiment logical primitive block",
            "high_level_simulation_mode": "Use high-level simulation mode",
            "aiautoinspectplots": "Enable AI plot inspection"
        }
        return descriptions.get(key.lower(), f"Setup parameter: {key}")
