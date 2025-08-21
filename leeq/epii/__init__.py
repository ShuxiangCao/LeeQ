"""
LeeQ EPII (Experiment Platform Intelligence Interface) Implementation

This module provides a gRPC-based interface for executing quantum experiments
through the EPII v1.0 standard, enabling external orchestrators to interact
with LeeQ experiments in a standardized way.
"""

__version__ = "1.0.0"

# Import core components when they are implemented
# from .service import EPIIService
# from .daemon import run_daemon
# from .experiments import ExperimentRouter
# from .parameters import ParameterManager
# from .serialization import serialize_numpy_array, deserialize_numpy_array
# from .config import load_config

__all__ = [
    "__version__",
    # Components will be added as they are implemented
    # "EPIIService",
    # "run_daemon",
    # "ExperimentRouter",
    # "ParameterManager",
    # "serialize_numpy_array",
    # "deserialize_numpy_array",
    # "load_config",
]
