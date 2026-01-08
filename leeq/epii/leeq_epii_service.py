"""LeeQ EPII Service Definition.

This module defines the LeeQ experiment platform as an EPII service using
the declarative decorator-based pattern from CalibrationNTKAgent.

Run with:
    python -m quantum_calibration_agent.epii.server.epii_server \\
        leeq/epii/leeq_epii_service.py --port 50051

Or with custom config:
    LEEQ_EPII_CONFIG=my_config.yml python -m quantum_calibration_agent.epii.server.epii_server \\
        leeq/epii/leeq_epii_service.py
"""

import base64
import inspect
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from quantum_calibration_agent.epii.server.declarative import (
    service_info,
    parameter,
    experiment,
    get_parameter,
    set_parameter
)

logger = logging.getLogger(__name__)

# ============================================================================
# Global State (Thread-Safe Initialization)
# ============================================================================

_setup = None
_experiment_router = None
_setup_lock = threading.Lock()


def _get_setup():
    """Lazy initialization of LeeQ setup (thread-safe).

    Uses double-check locking pattern to ensure thread safety
    when multiple concurrent requests try to initialize the setup.

    Returns:
        Tuple of (setup, experiment_router)
    """
    global _setup, _experiment_router

    # Fast path: already initialized
    if _setup is not None and _experiment_router is not None:
        return _setup, _experiment_router

    # Slow path: acquire lock and initialize
    with _setup_lock:
        # Double-check after acquiring lock
        if _setup is None or _experiment_router is None:
            from leeq.epii.config import EPIIConfig
            from leeq.epii.experiments import ExperimentRouter
            from leeq.experiments import ExperimentManager

            # Load config from environment or default
            config_path = os.environ.get('LEEQ_EPII_CONFIG', 'epii_config.yml')

            # Try to load config, fall back to defaults if file doesn't exist
            try:
                config = EPIIConfig.from_file(config_path)
            except FileNotFoundError:
                logger.warning(f"Config file {config_path} not found, using defaults")
                config = EPIIConfig()

            _setup = config.create_setup()

            # Register as default setup for ExperimentManager
            ExperimentManager().register_setup(_setup, set_as_default=True)

            _experiment_router = ExperimentRouter(_setup)
            logger.info(f"LeeQ setup initialized with {len(_experiment_router.experiment_map)} experiments")

    return _setup, _experiment_router


# ============================================================================
# Service Configuration
# ============================================================================

service_info(
    name="LeeQ",
    version="1.0.0",
    description="LeeQ quantum experiment platform for superconducting qubit calibration and characterization",
    supported_backends=["simulation", "hardware"]
)


# ============================================================================
# Parameter Definitions
# ============================================================================

# Parameters are managed dynamically by ParameterManager
# Static definitions can be added here if needed
PARAMETERS = {
    # Example static parameter:
    # "default_num_averages": parameter(
    #     type="int",
    #     default=1000,
    #     min=1,
    #     max=100000,
    #     description="Default number of averages for measurements"
    # ),
}


# ============================================================================
# Result Formatting
# ============================================================================

def _make_json_serializable(obj):
    """Recursively convert numpy arrays and other types to JSON-serializable format.

    Args:
        obj: Any Python object

    Returns:
        JSON-serializable version of the object
    """
    import json

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle objects with __dict__ (like plotly traces)
        try:
            return {k: _make_json_serializable(v) for k, v in obj.__dict__.items()
                    if not k.startswith('_')}
        except Exception:
            return str(obj)
    else:
        # Try to serialize, if it fails return string representation
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def _load_experiment_attributes(experiment_instance) -> Optional[Dict[str, Any]]:
    """Load experiment attributes from Chronicle or directly from the instance.

    Tries multiple methods to extract experiment data:
    1. Direct _record_entry if still available
    2. get_experiment_details() to load from Chronicle by path
    3. Direct attribute extraction as fallback

    Args:
        experiment_instance: Executed LeeQ experiment instance

    Returns:
        Dictionary of attributes or None if extraction failed
    """
    # Method 1: Try direct _record_entry (may still be available)
    if hasattr(experiment_instance, '_record_entry') and experiment_instance._record_entry is not None:
        try:
            return experiment_instance._record_entry.load_all_attributes()
        except Exception as e:
            logger.debug(f"Failed to load from _record_entry: {e}")

    # Method 2: Use get_experiment_details() to load from Chronicle
    if hasattr(experiment_instance, 'get_experiment_details'):
        try:
            details = experiment_instance.get_experiment_details()
            record_details = details.get('record_details')

            if record_details:
                from leeq.chronicle import Chronicle
                chronicle = Chronicle()

                # Load attributes using record path info
                record_book_path = record_details.get('record_book_path')
                record_entry_path = record_details.get('record_entry_path')

                if record_book_path and record_entry_path:
                    attrs = chronicle.load_attributes(
                        record_book_path=record_book_path,
                        record_entry_path=record_entry_path
                    )
                    if attrs:
                        return attrs
        except Exception as e:
            logger.debug(f"Failed to load from Chronicle: {e}")

    # Method 3: Extract common result attributes directly from instance
    try:
        attrs = {}

        # Common result attribute names
        result_attrs = [
            'fit_params', 'frequencies', 'signal', 'data', 'result',
            'resonance_frequency', 't1', 't2', 'amplitude', 'phase'
        ]

        for attr_name in result_attrs:
            if hasattr(experiment_instance, attr_name):
                value = getattr(experiment_instance, attr_name)
                if value is not None:
                    attrs[attr_name] = value

        if attrs:
            return attrs
    except Exception as e:
        logger.debug(f"Failed to extract direct attributes: {e}")

    return None


def _format_experiment_result(experiment_instance) -> Dict[str, Any]:
    """Convert LeeQ experiment result to EPII format.

    Args:
        experiment_instance: Executed LeeQ experiment instance

    Returns:
        Dict with structure expected by EPIIServer:
        {
            "data": {
                "param_name": {"type": "number|array|text|boolean", "value": ..., "description": ...},
                ...
            },
            "plots": [
                {"description": "...", "plotly_json": {...}},
                ...
            ],
            "metadata": {...}
        }
    """
    result = {
        "data": {},
        "plots": [],
        "metadata": {}
    }

    # Try to extract data from Chronicle record
    all_attrs = _load_experiment_attributes(experiment_instance)

    if all_attrs:
        for key, value in all_attrs.items():
            # Skip internal attributes
            if key.startswith('__') or key == 'EPII_INFO':
                continue

            if isinstance(value, np.ndarray):
                result["data"][key] = {
                    "type": "array",
                    "value": {
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "data": base64.b64encode(value.tobytes()).decode('utf-8')
                    },
                    "description": f"Array data: {key}"
                }
            # IMPORTANT: Check bool BEFORE int/float because bool is subclass of int
            elif isinstance(value, bool):
                result["data"][key] = {
                    "type": "boolean",
                    "value": value,
                    "description": f"Boolean value: {key}"
                }
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                result["data"][key] = {
                    "type": "number",
                    "value": float(value),
                    "description": f"Numeric value: {key}"
                }
            elif isinstance(value, str):
                result["data"][key] = {
                    "type": "text",
                    "value": value,
                    "description": f"Text value: {key}"
                }
            # Skip complex types that can't be serialized
            else:
                logger.debug(f"Skipping non-serializable attribute: {key} ({type(value)})")

    # Extract plots from browser functions
    if hasattr(experiment_instance, 'get_browser_functions'):
        try:
            for func_name, func_method in experiment_instance.get_browser_functions():
                try:
                    plot_result = func_method()
                    if plot_result is not None:
                        # Convert plotly figure to JSON
                        if hasattr(plot_result, 'to_plotly_json'):
                            plotly_json = plot_result.to_plotly_json()
                        elif hasattr(plot_result, 'to_dict'):
                            plotly_json = plot_result.to_dict()
                        elif isinstance(plot_result, dict):
                            plotly_json = plot_result
                        else:
                            logger.debug(f"Skipping non-dict plot result: {func_name}")
                            continue

                        # Ensure plotly_json is JSON-serializable
                        plotly_json = _make_json_serializable(plotly_json)

                        result["plots"].append({
                            "description": func_name,
                            "plotly_json": plotly_json
                        })
                except Exception as e:
                    logger.debug(f"Failed to extract plot {func_name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to get browser functions: {e}")

    # Add metadata
    result["metadata"] = {
        "experiment_class": experiment_instance.__class__.__name__,
        "experiment_module": experiment_instance.__class__.__module__
    }

    return result


# ============================================================================
# Dynamic Experiment Registration
# ============================================================================

def _map_python_type_to_epii(type_str: str) -> str:
    """Map Python type annotations to EPII type strings."""
    type_map = {
        'float': 'float',
        'int': 'int',
        'str': 'string',
        'bool': 'bool',
        'Any': 'string',
    }
    # Handle typing annotations like "typing.Optional[float]"
    for py_type, epii_type in type_map.items():
        if py_type in str(type_str):
            return epii_type
    return 'string'


def _register_all_experiments():
    """Dynamically register all LeeQ experiments with EPII_INFO.

    This function discovers all experiments in the ExperimentRouter
    and registers them with the declarative framework.

    Call this at module load time if using dynamic registration.
    """
    try:
        setup, router = _get_setup()
    except Exception as e:
        logger.error(f"Failed to initialize setup for dynamic registration: {e}")
        return

    registered_count = 0

    for exp_name, exp_class in router.experiment_map.items():
        try:
            epii_info = getattr(exp_class, 'EPII_INFO', {})
            params_schema = router.get_experiment_parameters(exp_name)

            # Build params list for decorator
            params = []
            for p_name, p_info in params_schema.items():
                params.append((
                    p_name,
                    _map_python_type_to_epii(p_info.get('type', 'Any')),
                    p_info.get('required', False),
                    str(p_info.get('default', '')) if p_info.get('default') is not None else '',
                    f"Parameter {p_name}"
                ))

            # Create wrapper function with proper signature
            def make_wrapper(exp_cls, name, param_list):
                def wrapper(**kwargs):
                    setup, router = _get_setup()
                    resolved = router.map_parameters(name, kwargs, setup)
                    exp_instance = exp_cls(**resolved)
                    return _format_experiment_result(exp_instance)

                # Add proper function signature for DeclarativeServiceAdapter
                wrapper.__signature__ = inspect.Signature([
                    inspect.Parameter(
                        p_name,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=inspect.Parameter.empty if required else default
                    )
                    for p_name, _type, required, default, _desc in param_list
                ])
                return wrapper

            # Register with decorator
            wrapper_fn = make_wrapper(exp_class, exp_name, params)
            experiment(
                name=exp_name,
                description=epii_info.get('description', f'{exp_name} experiment'),
                params=params,
                outputs=list(epii_info.get('attributes', {}).keys())
            )(wrapper_fn)

            registered_count += 1
            logger.debug(f"Registered dynamic experiment: {exp_name}")

        except Exception as e:
            logger.warning(f"Failed to register experiment {exp_name}: {e}")

    logger.info(f"Dynamically registered {registered_count} experiments")


# ============================================================================
# Module Initialization
# ============================================================================

# Enable dynamic registration of ALL experiments discovered by ExperimentRouter
_register_all_experiments()
