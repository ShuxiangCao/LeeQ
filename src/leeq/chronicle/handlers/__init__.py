from ..logger import setup_logging

from .dummy import RecordHandlerDummy
from .handlers import RecordHandlersBase
from .hdf5 import RecordHandlerHDF5
from .memory import RecordHandlerMemory

logger = setup_logging(__name__)

available_handlers = {
    "hdf5": RecordHandlerHDF5,
    "dummy": RecordHandlerDummy,
    "memory": RecordHandlerMemory
}


def get_handler(handler_name: str) -> RecordHandlersBase:
    """
    Get a handler by its name.

    Parameters:
        handler_name (str): The name of the handler.

    Returns:
        RecordHandlersBase: The handler.
    """
    if handler_name not in available_handlers:
        msg = f"Handler {handler_name} is not available."
        logger.error(msg)
        raise ValueError(msg)
    return available_handlers[handler_name]
