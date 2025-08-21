import inspect

import decorator

from .chronicle import Chronicle
from .core import LoggableObject
from .logger import setup_logging

logger = setup_logging(__name__)


@decorator.decorator
def log_and_record(func, overwrite_func_name=None, *args, **kwargs):
    """
    Decorator function for the functions that want to be logged. The function must be a method of a LoggableObject.
    Using this decorator will record the object and modified attributes within this function call
    after the function execution.

    Parameters:
        func (function): The function to be logged.
        overwrite_func_name (str): Optional. The name of the function to be recorded. If not specified, the name of
                                    the function will be used.
        args (list): The arguments of the function.
        kwargs (dict): The keyword arguments of the function.

    Returns:
        Any: The return value of the function.
    """
    return _log_and_record(func, args, kwargs, overwrite_func_name=overwrite_func_name)


@decorator.decorator
def log_event(func, overwrite_func_name=None, *args, **kwargs):
    """
    Decorator function for the functions that want to be logged. The function must be a method of a LoggableObject.
    Using this decorator will only record return values and arguments of the function.

    Parameters:
        func (function): The function to be logged.
        overwrite_func_name (str): Optional. The name of the function to be recorded. If not specified, the name of
                                    the function will be used.
        args (list): The arguments of the function.
        kwargs (dict): The keyword arguments of the function.

    Returns:
        Any: The return value of the function.
    """
    return _log_and_record(func, args, kwargs, record_details=False, overwrite_func_name=overwrite_func_name)


def _log_and_record(func, args, kwargs, record_details=True, overwrite_func_name=None):
    """
    Decorator function for the functions that want to be logged. The function must be a method of a LoggableObject.

    Parameters:
        func (function): The function to be logged.
        args (list): The arguments of the function.
        kwargs (dict): The keyword arguments of the function.
        record_details (bool): Optional. Whether to record the object and attributes after the function execution.
                                If false, only the arguments and return values are recorded.
        overwrite_func_name (str): Optional. The name of the function to be recorded. If not specified, the name of
                                    the function will be used.

    Returns:
        Any: The return value of the function.

    Raises:
        RuntimeError: If the function is not a method of a LoggableObject.
    """
    if len(args) == 0:
        msg = f"Function {func.__qualname__} is not a class method."
        logger.error(msg)
        raise RuntimeError(msg)

    self = args[0]

    if not isinstance(self, LoggableObject):
        msg = f"Function {func.__qualname__} is not a method of a LoggableObject. Object type: {type(self)}."
        logger.error(msg)
        raise RuntimeError(msg)

    chronicle = Chronicle()

    with chronicle.new_record() as record:
        if record is None:
            # There are no active record books. Simply execute the function.
            return func(*args, **kwargs)

        # Finalize the setup of the record.

        name = overwrite_func_name if overwrite_func_name is not None else func.__qualname__

        record.set_name(name)
        record.record_metadata()
        record.record_args(args[1:], kwargs)

        # Save the argument to the class as well.

        if record_details:
            self.set_record_entry(record)

        # Set attributes to the object to indicate the latest record details.
        record_details = {
            "record_id": str(record.record_id),
            "record_entry_path": str(record.get_path()),
            "record_book_path": str(record.record_book.get_path()),
            "record_time": record.record_time,
        }

        self.register_log_and_record_args(
            func, args[1:], kwargs, record_details=record_details
        )

        try:
            retval = func(*args, **kwargs)
            error_info = None
        except Exception as e:
            retval = None
            error_info = e
            record.record_error_info(error_info)

        if record_details:
            self.set_record_entry()

        record.record_return_values(retval)

        # Could be too detailed. Comment out for now.
        # self.logger.info(f'{record.record_id}: {func.__qualname__} recorded.')

        # Set attributes to the object to indicate the latest record details.
        record_details = {
            "record_id": str(record.record_id),
            "record_entry_path": str(record.get_path()),
            "record_book_path": str(record.record_book.get_path()),
            "record_time": record.record_time,
        }

        self.register_log_and_record_args(
            func, args[1:], kwargs, record_details=record_details, overwrite_func_name=overwrite_func_name
        )

        # Take a snapshot of the object after finish the function execution.
        if record_details:
            record.record_object(self)

    if error_info is not None:
        raise error_info

    return retval


def register_browser_function(*args, **kwargs):
    """
    Decorator function for the functions that used to visualize data of the class.
     The function must be a method of a LoggableObject.

    Parameters:
        args (list): The arguments of the function.
        kwargs (dict): The keyword arguments of the function.

    Returns:
        Any: The return value of the function.
    """

    def inner_func(func):
        """
        Decorator function for the functions that used to visualize data of the class.
        The function must be a method of a LoggableObject.

        Parameters:
            func (function): The function to be registered.

        Returns:
            Any: The same function.
        """

        func._browser_function = True
        func._browser_function_args = args
        func._browser_function_kwargs = kwargs

        return func

    return inner_func
