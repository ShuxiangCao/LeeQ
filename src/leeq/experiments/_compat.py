"""Helpers for compatibility import packages."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Iterable


def reexport_package(target_globals: dict, canonical_name: str, submodules: Iterable[str] = ()) -> ModuleType:
    """Re-export a canonical package and register old submodule aliases."""
    package = importlib.import_module(canonical_name)

    public_names = []
    for name, value in package.__dict__.items():
        if not name.startswith("_"):
            target_globals[name] = value
            public_names.append(name)

    old_name = target_globals["__name__"]
    for submodule in submodules:
        old_submodule = f"{old_name}.{submodule}"
        new_submodule = f"{canonical_name}.{submodule}"
        module = importlib.import_module(new_submodule)
        sys.modules[old_submodule] = module
        target_globals[submodule.split(".")[0]] = importlib.import_module(
            f"{canonical_name}.{submodule.split('.')[0]}"
        )

    target_globals["__all__"] = sorted(set(public_names))
    return package
