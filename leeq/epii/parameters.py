"""
EPII Parameter Management - Simplified dictionary-based approach.

Provides direct access to all LeeQ parameters without validation.
Uses dot notation (e.g., 'q0.f01', 'status.shot_number').
"""

import numpy as np


class ParameterManager:
    """Simple parameter manager with direct dictionary access."""

    def __init__(self, setup=None):
        self.setup = setup
        self._cache = {}

    def _flatten_dict(self, d, parent_key="", sep="."):
        """Flatten nested dictionary with dot notation keys."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict) and not k.startswith("_"):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get_all_parameters(self):
        """Get all parameters as a single flat dictionary."""
        params = {}

        # Get setup status parameters
        if self.setup and hasattr(self.setup, "status"):
            if hasattr(self.setup.status, "get_parameters"):
                status_params = self.setup.status.get_parameters()
            elif hasattr(self.setup.status, "_internal_dict"):
                status_params = self.setup.status._internal_dict.copy()
            else:
                status_params = {}
            for k, v in status_params.items():
                params[f"status.{k}"] = v

        # Get element parameters
        if self.setup and hasattr(self.setup, "_elements"):
            for elem_name, element in self.setup._elements.items():
                if hasattr(element, "_parameters"):
                    elem_params = self._flatten_dict(element._parameters)
                    for k, v in elem_params.items():
                        params[f"{elem_name}.{k}"] = v

        # Get qubit parameters from qubits list
        if self.setup and hasattr(self.setup, "qubits"):
            for i, qubit in enumerate(self.setup.qubits):
                if hasattr(qubit, "_parameters"):
                    qubit_params = self._flatten_dict(qubit._parameters)
                    for k, v in qubit_params.items():
                        params[f"q{i}.{k}"] = v

        params.update(self._cache)
        return params

    def get_parameter(self, name):
        """Get parameter value by dot notation name."""
        return self.get_all_parameters().get(name)

    def _parse_value(self, value):
        """Parse string value to appropriate type."""
        if not isinstance(value, str):
            return value
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        elif value == "null":
            return None
        try:
            if "." not in value and "e" not in value.lower():
                return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    def set_parameter(self, name, value):
        """Set parameter value by dot notation name. No validation performed."""
        parts = name.split(".")
        parsed_value = self._parse_value(value)

        # Try to set in appropriate location
        if parts[0] == "status" and self.setup and len(parts) >= 2:
            if hasattr(self.setup, "status"):
                if hasattr(self.setup.status, "set_parameter"):
                    self.setup.status.set_parameter(parts[1], parsed_value)
                    return True
                elif hasattr(self.setup.status, "_internal_dict"):
                    self.setup.status._internal_dict[parts[1]] = parsed_value
                    return True

        elif parts[0].startswith("q") and parts[0][1:].isdigit():
            idx = int(parts[0][1:])
            if self.setup and hasattr(self.setup, "qubits"):
                if idx < len(self.setup.qubits):
                    qubit = self.setup.qubits[idx]
                    if hasattr(qubit, "_parameters"):
                        target = qubit._parameters
                        for part in parts[1:-1]:
                            if part not in target:
                                target[part] = {}
                            target = target[part]
                        target[parts[-1]] = parsed_value
                        return True

        elif self.setup and hasattr(self.setup, "_elements"):
            if parts[0] in self.setup._elements:
                element = self.setup._elements[parts[0]]
                if hasattr(element, "_parameters"):
                    target = element._parameters
                    for part in parts[1:-1]:
                        if part not in target:
                            target[part] = {}
                        target = target[part]
                    target[parts[-1]] = parsed_value
                    return True

        # Fallback to cache
        self._cache[name] = parsed_value
        return True

    def serialize_value(self, value):
        """Serialize value for EPII transmission."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return value
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (list, dict)):
            return value
        else:
            return str(value)

