"""
EPII Experiment Router

This module maps EPII experiment names to LeeQ experiment classes
and handles experiment discovery and execution routing.
"""

import logging
from typing import Dict, Type, Optional, Any, List
import inspect

# Import LeeQ experiment classes
from leeq.experiments.builtin.basic.calibrations.rabi import NormalisedRabi, MultiQubitRabi
from leeq.experiments.builtin.basic.calibrations.ramsey import SimpleRamseyMultilevel
from leeq.experiments.builtin.basic.calibrations.drag import (
    DragCalibrationSingleQubitMultilevel,
    DragPhaseCalibrationMultiQubitsMultilevel
)
from leeq.experiments.builtin.basic.characterizations.t1 import SimpleT1, MultiQubitT1
from leeq.experiments.builtin.basic.characterizations.t2 import SpinEchoMultiLevel
from leeq.experiments.builtin.basic.characterizations.randomized_benchmarking import (
    RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem,
    SingleQubitRandomizedBenchmarking
)

logger = logging.getLogger(__name__)


class ExperimentRouter:
    """
    Routes EPII experiment requests to appropriate LeeQ experiment classes.

    Manages the mapping between EPII standard experiment names and
    LeeQ's internal experiment implementations.
    """

    def __init__(self):
        """Initialize the experiment router with default mappings."""
        self.experiment_map: Dict[str, Type] = {}
        self.parameter_map: Dict[str, Dict[str, str]] = {}
        self._initialize_experiment_map()
        self._initialize_parameter_map()

    def _initialize_experiment_map(self):
        """
        Initialize the mapping of EPII names to LeeQ experiments.

        Maps EPII standard experiment names to corresponding LeeQ classes.
        """
        self.experiment_map = {
            # Basic calibration experiments
            "rabi": NormalisedRabi,
            "multi_qubit_rabi": MultiQubitRabi,

            # T1 relaxation experiments
            "t1": SimpleT1,
            "multi_qubit_t1": MultiQubitT1,

            # Ramsey experiments (T2*)
            "ramsey": SimpleRamseyMultilevel,

            # Echo experiments (T2 echo)
            "echo": SpinEchoMultiLevel,
            "spin_echo": SpinEchoMultiLevel,

            # DRAG calibration
            "drag": DragCalibrationSingleQubitMultilevel,
            "drag_phase": DragPhaseCalibrationMultiQubitsMultilevel,

            # Randomized benchmarking
            "randomized_benchmarking": RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem,
            "single_qubit_rb": SingleQubitRandomizedBenchmarking,
        }

    def _initialize_parameter_map(self):
        """
        Initialize parameter mappings between EPII names and LeeQ parameter names.

        This handles cases where EPII uses different parameter names than LeeQ.
        """
        self.parameter_map = {
            "rabi": {
                "amplitude": "amp",
                "start_width": "start",
                "stop_width": "stop",
                "width_step": "step",
                "qubit": "dut_qubit",
            },
            "t1": {
                "qubit": "qubit",
                "time_max": "time_length",
                "time_step": "time_resolution",
            },
            "ramsey": {
                "qubit": "dut",
                "start_time": "start",
                "stop_time": "stop",
                "time_step": "step",
                "frequency_offset": "set_offset",
            },
            "echo": {
                "qubit": "dut",
                "evolution_time": "free_evolution_time",
                "time_step": "time_resolution",
            },
            "drag": {
                "qubit": "dut",
                "repetitions": "N",
                "alpha_start": "inv_alpha_start",
                "alpha_stop": "inv_alpha_stop",
                "num_points": "num",
            },
            "randomized_benchmarking": {
                "qubits": "dut_list",
                "max_length": "seq_length",
                "num_sequences": "kinds",
                "clifford_set": "cliff_set",
            },
        }

    def get_experiment(self, name: str) -> Optional[Type]:
        """
        Get the LeeQ experiment class for an EPII experiment name.

        Args:
            name: EPII standard experiment name

        Returns:
            LeeQ experiment class or None if not found
        """
        experiment_class = self.experiment_map.get(name)
        if not experiment_class:
            logger.warning(f"Experiment '{name}' not found in router mappings")
        return experiment_class

    def list_experiments(self) -> Dict[str, str]:
        """
        List all available experiments with their descriptions.

        Returns:
            Dictionary mapping experiment names to descriptions
        """
        descriptions = {
            "rabi": "Rabi oscillation experiment for calibrating pulse amplitude",
            "multi_qubit_rabi": "Multi-qubit Rabi oscillation experiment",
            "t1": "T1 relaxation time measurement",
            "multi_qubit_t1": "Multi-qubit T1 relaxation measurement",
            "ramsey": "Ramsey experiment for T2* measurement and frequency calibration",
            "echo": "Spin echo experiment for T2 echo measurement",
            "spin_echo": "Spin echo experiment for T2 echo measurement (alias)",
            "drag": "DRAG coefficient calibration using AllXY sequence",
            "drag_phase": "DRAG phase calibration for multiple qubits",
            "randomized_benchmarking": "Randomized benchmarking for gate fidelity",
            "single_qubit_rb": "Single qubit randomized benchmarking",
        }

        # Only return descriptions for experiments that are actually mapped
        return {name: desc for name, desc in descriptions.items()
                if name in self.experiment_map}

    def get_experiment_parameters(self, name: str) -> Dict[str, Any]:
        """
        Get the parameter schema for an experiment.

        Args:
            name: EPII standard experiment name

        Returns:
            Dictionary describing the experiment's parameters
        """
        experiment_class = self.get_experiment(name)
        if not experiment_class:
            return {}

        # Get the run method signature
        try:
            run_method = getattr(experiment_class, 'run')
            signature = inspect.signature(run_method)

            parameters = {}
            for param_name, param in signature.parameters.items():
                # Skip 'self' parameter
                if param_name == 'self':
                    continue

                # Get parameter info
                param_info = {
                    "name": param_name,
                    "required": param.default == inspect.Parameter.empty,
                    "default": None if param.default == inspect.Parameter.empty else param.default,
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
                }

                parameters[param_name] = param_info

            return parameters

        except AttributeError:
            logger.error(f"Experiment class {experiment_class.__name__} has no 'run' method")
            return {}

    def map_parameters(self, experiment_name: str, epii_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map EPII parameter names to LeeQ parameter names.

        Args:
            experiment_name: Name of the experiment
            epii_params: Parameters with EPII naming convention

        Returns:
            Parameters with LeeQ naming convention
        """
        if experiment_name not in self.parameter_map:
            # No mapping needed, return as is
            return epii_params

        mapping = self.parameter_map[experiment_name]
        leeq_params = {}

        for epii_name, value in epii_params.items():
            # Map the parameter name if a mapping exists
            leeq_name = mapping.get(epii_name, epii_name)
            leeq_params[leeq_name] = value

        return leeq_params

    def validate_parameters(self, experiment_name: str, parameters: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate parameters for an experiment.

        Args:
            experiment_name: Name of the experiment
            parameters: Parameters to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Get the experiment class
        experiment_class = self.get_experiment(experiment_name)
        if not experiment_class:
            return False, [f"Unknown experiment: {experiment_name}"]

        # Get parameter schema
        schema = self.get_experiment_parameters(experiment_name)

        # Check required parameters
        for param_name, param_info in schema.items():
            if param_info["required"] and param_name not in parameters:
                errors.append(f"Missing required parameter: {param_name}")

        # Validate parameter types (basic validation)
        for param_name, value in parameters.items():
            if param_name not in schema:
                # Warning, but not an error - might be an optional parameter
                logger.warning(f"Unknown parameter '{param_name}' for experiment '{experiment_name}'")
            else:
                # Basic type checking could be added here
                pass

        return len(errors) == 0, errors

    def discover_experiments(self) -> List[Dict[str, Any]]:
        """
        Discover all available experiments and their capabilities.

        Returns:
            List of experiment capability dictionaries
        """
        capabilities = []

        for name, experiment_class in self.experiment_map.items():
            capability = {
                "name": name,
                "class": experiment_class.__name__,
                "module": experiment_class.__module__,
                "parameters": self.get_experiment_parameters(name),
                "description": self.list_experiments().get(name, "")
            }
            capabilities.append(capability)

        return capabilities
