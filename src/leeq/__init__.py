"""Top-level public API for LeeQ.

The package root intentionally avoids importing the full experiment tree.
Public convenience exports are resolved lazily from :mod:`leeq.api`.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

__all__ = [
    "Experiment",
    "LeeQAIExperiment",
    "ExperimentManager",
    "Sweeper",
    "SweepParametersSideEffect",
    "SweepParametersSideEffectAttribute",
    "SweepParametersSideEffectFactory",
    "SweepParametersSideEffectFunction",
    "basic_run",
    "setup",
]

try:
    __version__ = metadata.version("leeq")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"


def __getattr__(name: str) -> Any:
    if name in __all__:
        from leeq import api

        return getattr(api, name)
    raise AttributeError(f"module 'leeq' has no attribute {name!r}")
