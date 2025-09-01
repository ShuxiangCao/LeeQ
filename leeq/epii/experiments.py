"""
EPII Experiment Router

This module maps EPII experiment names to LeeQ experiment classes
and handles experiment discovery and execution routing.
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ExperimentRouter:
    """
    Routes EPII experiment requests to appropriate LeeQ experiment classes.

    Manages the mapping between EPII standard experiment names and
    LeeQ's internal experiment implementations.
    """

    def __init__(self):
        """Initialize the experiment router with dynamic discovery."""
        self.experiment_map: Dict[str, Type] = {}
        self._discover_experiments()  # Changed from _initialize_experiment_map

    def _discover_experiments(self):
        """
        Dynamically discover all experiments with EPII_INFO.
        Replaces the static _initialize_experiment_map method.
        """
        from leeq.experiments import builtin

        # Walk through all modules in builtin
        for importer, modname, ispkg in pkgutil.walk_packages(
            builtin.__path__,
            prefix="leeq.experiments.builtin."
        ):
            try:
                module = importlib.import_module(modname)

                # Check each class in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Only process classes actually defined in this module (not imported)
                    if obj.__module__ != modname:
                        continue

                    # Include if has EPII_INFO and run method (not a base class)
                    if hasattr(obj, 'EPII_INFO') and hasattr(obj, 'run'):
                        # Build experiment name with module prefix for duplicates
                        # Extract the last parts of module path for context
                        module_parts = modname.split('.')
                        if 'characterizations' in module_parts:
                            exp_name = f"characterizations.{name}"
                        elif 'calibrations' in module_parts:
                            exp_name = f"calibrations.{name}"
                        elif 'multi_qubit_gates' in module_parts:
                            exp_name = f"multi_qubit_gates.{name}"
                        elif 'tomography' in module_parts:
                            exp_name = f"tomography.{name}"
                        elif 'hamiltonian_tomography' in module_parts:
                            exp_name = f"hamiltonian_tomography.{name}"
                        elif 'optimal_control' in module_parts:
                            exp_name = f"optimal_control.{name}"
                        else:
                            exp_name = name

                        # Check for duplicates
                        if exp_name in self.experiment_map:
                            # Add more specific module path to disambiguate
                            specific_path = '.'.join(module_parts[-2:]) if len(module_parts) > 1 else module_parts[-1]
                            exp_name = f"{specific_path}.{name}"

                        self.experiment_map[exp_name] = obj
                        logger.debug(f"Discovered experiment: {exp_name} -> {name}")

            except Exception as e:
                logger.warning(f"Failed to import {modname}: {e}")

        logger.info(f"Discovered {len(self.experiment_map)} experiments with EPII_INFO")


    def get_experiment_info(self, name: str) -> Dict[str, Any]:
        """
        Get EPII_INFO and run docstring for an experiment.
        New method to support enhanced responses.
        """
        experiment_class = self.get_experiment(name)
        if not experiment_class:
            return {}

        info = {
            'epii_info': getattr(experiment_class, 'EPII_INFO', {}),
            'run_docstring': None
        }

        # Get run method docstring
        if hasattr(experiment_class, 'run'):
            run_method = experiment_class.run
            if run_method.__doc__:
                info['run_docstring'] = inspect.cleandoc(run_method.__doc__)

        return info

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
        Updated to use EPII_INFO descriptions.
        """
        descriptions = {}
        for name, exp_class in self.experiment_map.items():
            if hasattr(exp_class, 'EPII_INFO'):
                epii_info = exp_class.EPII_INFO
                descriptions[name] = epii_info.get('description', f'{name} experiment')
            else:
                descriptions[name] = f'{name} experiment'
        return descriptions

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
            run_method = experiment_class.run
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

    def map_parameters(self, experiment_name: str, epii_params: Dict[str, Any], setup=None) -> Dict[str, Any]:
        """
        Resolve qubit references in experiment parameters.
        
        Args:
            experiment_name: Name of the experiment
            epii_params: Parameters dictionary
            setup: Optional setup instance to resolve qubit references

        Returns:
            Parameters with resolved qubit objects
        """
        # With aliases removed, parameters are passed through directly
        # Only qubit resolution is performed
        leeq_params = epii_params.copy()

        # Resolve qubit references if setup is provided
        if setup:
            leeq_params = self._resolve_qubit_references(leeq_params, setup)

        return leeq_params

    def _resolve_qubit_references(self, params: Dict[str, Any], setup) -> Dict[str, Any]:
        """
        Resolve qubit string references to actual qubit objects.
        
        Args:
            params: Parameters dictionary
            setup: Setup instance with qubit registry
            
        Returns:
            Parameters with resolved qubit objects
        """
        resolved = params.copy()

        # List of parameter names that typically contain qubit references
        qubit_param_names = ['qubit', 'dut_qubit', 'dut', 'qubits', 'dut_list']

        for param_name in qubit_param_names:
            if param_name in resolved:
                value = resolved[param_name]

                # Handle single qubit reference
                if isinstance(value, str):
                    try:
                        resolved[param_name] = self._get_qubit_from_setup(value, setup)
                    except ValueError as e:
                        logger.warning(f"Failed to resolve qubit for {param_name}: {e}")
                        # Keep the string value if resolution fails
                        # This allows experiments to handle string references themselves
                        pass
                # Handle list of qubit references
                elif isinstance(value, list):
                    resolved_list = []
                    for q in value:
                        if isinstance(q, str):
                            try:
                                resolved_list.append(self._get_qubit_from_setup(q, setup))
                            except ValueError as e:
                                logger.warning(f"Failed to resolve qubit '{q}': {e}")
                                # Keep string if resolution fails
                                resolved_list.append(q)
                        else:
                            resolved_list.append(q)
                    resolved[param_name] = resolved_list

        return resolved

    def _get_qubit_from_setup(self, qubit_name: str, setup):
        """
        Get a qubit object from the setup by name.
        
        Args:
            qubit_name: Name of the qubit (e.g., '0', '1', 'q0', 'Q0')
            setup: Setup instance
            
        Returns:
            Qubit object from setup
            
        Raises:
            ValueError: If qubit cannot be resolved
        """
        # Handle pure number format ("0", "1", etc.)
        if qubit_name.isdigit():
            index = int(qubit_name)
            # Try qubits list first
            if hasattr(setup, 'qubits') and isinstance(setup.qubits, list):
                if index < len(setup.qubits):
                    return setup.qubits[index]
            # Try q{index} attribute
            if hasattr(setup, f'q{index}'):
                return getattr(setup, f'q{index}')

        # Handle "q0", "Q0" format
        elif qubit_name.lower().startswith('q'):
            try:
                index = int(qubit_name[1:])
                # Try qubits list
                if hasattr(setup, 'qubits') and index < len(setup.qubits):
                    return setup.qubits[index]
                # Try direct attribute
                if hasattr(setup, qubit_name.lower()):
                    return getattr(setup, qubit_name.lower())
            except ValueError:
                pass

        # Fail explicitly instead of returning string
        raise ValueError(f"Cannot resolve qubit '{qubit_name}' from setup. Available: {getattr(setup, 'qubits', [])}")

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
        for param_name, _value in parameters.items():
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
