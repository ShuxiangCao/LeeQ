"""
LeeQ Chronicle Module - Vendored LabChronicle Package

This module contains a vendored copy of the LabChronicle package, integrated
directly into LeeQ to eliminate external dependencies. LabChronicle provides
experiment logging, data persistence, and tracking capabilities for quantum
computing experiments.

The vendored package maintains the same API as the original LabChronicle,
allowing existing code to work without modification after import path updates.

Key Components:
- Chronicle: Main logging and persistence manager
- LoggableObject: Base class for objects that can be logged and persisted
- Decorators: log_and_record, log_event for automatic logging
- Handlers: Storage backends (HDF5, memory, dummy)

Original LabChronicle repository: https://github.com/ShuxiangCao/LabChronicle
"""

from .chronicle import Chronicle, load_attributes, load_object
from .core import LoggableObject
from .decorators import log_and_record, log_event, register_browser_function
from .logger import setup_logging

browser_function = register_browser_function  # Alias for backwards compatibility

log = None  # Chronical log instance, None if not running
logger = setup_logging(__name__)
