import inspect
import os
import datetime
import getpass
from pathlib import Path
import platform


def get_system_info():
    """
    Get the system information, includes the time, computer name, operating system version, python version user name.
    """

    platform_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "node": platform.node(),
        "python_version": platform.python_version(),
        "python_build": platform.python_build(),
        "python_compiler": platform.python_compiler(),
        "python_branch": platform.python_branch(),
        "python_implementation": platform.python_implementation(),
        "python_revision": platform.python_revision(),
        "python_version_tuple": platform.python_version_tuple(),
        "uname": platform.uname(),
    }

    jupyter_user = os.environ.get("JUPYTERHUB_USER", getpass.getuser())

    info = {
        "platform_info": platform_info,
        "environ": {os.environ[k]: k for k in os.environ.keys()},
        "start_time": datetime.datetime.now().timestamp(),
        "user": jupyter_user,
    }

    return info


def get_log_path(main_dir: Path, name: str) -> Path:
    """
    Generate the log path according to the rule

    Parameters:
        main_dir (str): The main directory of the log
        name (str): The name of the log

    Returns:
        log_dir (str): The log directory
    """

    jupyter_user = os.environ.get("JUPYTERHUB_USER", getpass.getuser())
    now = datetime.datetime.now()

    year_month = now.strftime("%Y-%m")
    day = now.strftime("%Y-%m-%d")
    day_time = now.strftime("%H.%M.%S" + name)

    log_dir = main_dir / jupyter_user / year_month / day / day_time

    return log_dir


def find_methods_with_tag(instance: object, tag_name: str) -> list:
    """
    Find all methods with the given tag name.

    Parameters:
        instance (object): The instance to search for the methods.
        tag_name (str): The name of the tag.

    Returns:
        tagged_methods (list): A list of tuples of the method name and the method.
    """
    tagged_methods = []
    for name, method in inspect.getmembers(
            instance, predicate=inspect.ismethod):
        if getattr(method, tag_name, None):
            tagged_methods.append((name, method))
    return tagged_methods
