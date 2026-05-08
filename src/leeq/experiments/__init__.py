"""Public experiment API exports resolved lazily."""

from __future__ import annotations

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


def __getattr__(name: str) -> Any:
    if name in {"Experiment", "LeeQAIExperiment", "ExperimentManager", "basic_run", "setup"}:
        from leeq.experiments import experiments

        return getattr(experiments, name)

    if name in {
        "Sweeper",
        "SweepParametersSideEffect",
        "SweepParametersSideEffectAttribute",
        "SweepParametersSideEffectFactory",
        "SweepParametersSideEffectFunction",
    }:
        from leeq.experiments import sweeper

        return getattr(sweeper, name)

    raise AttributeError(f"module 'leeq.experiments' has no attribute {name!r}")
