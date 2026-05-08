"""
Centralized configuration management for LeeQ.

This module provides centralized configuration management with environment variable support,
path validation, and development mode settings for the LeeQ quantum computing framework.
"""

import os
from pathlib import Path


class Config:
    """
    Centralized configuration management for LeeQ.

    This class provides a single source of truth for all configuration settings
    in the LeeQ framework. Settings can be overridden using environment variables
    with the LEEQ_ prefix.

    Attributes
    ----------
    LOG_LEVEL : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Default: 'INFO', override with LEEQ_LOG_LEVEL.
    SUPPRESS_LOGGING : bool
        Whether to suppress logging output.
        Default: False, override with LEEQ_SUPPRESS_LOGGING.
    CONFIG_PATH : Path
        Path to the main configuration file.
        Default: 'configs/default.json', override with LEEQ_CONFIG_PATH.
    CALIBRATION_LOG_PATH : Path
        Path to store calibration logs.
        Default: './calibration_logs', override with LEEQ_CALIBRATION_LOG_PATH.
    DEBUG_MODE : bool
        Whether debug mode is enabled.
        Default: False, override with LEEQ_DEBUG.

    Examples
    --------
    >>> from leeq.config import Config
    >>> Config.validate()  # Ensure paths exist
    True
    >>> print(Config.LOG_LEVEL)
    INFO
    >>> print(Config.CALIBRATION_LOG_PATH)
    ./calibration_logs
    """

    # Logging configuration
    LOG_LEVEL = os.getenv('LEEQ_LOG_LEVEL', 'INFO')
    SUPPRESS_LOGGING = os.getenv('LEEQ_SUPPRESS_LOGGING', 'false').lower() == 'true'

    # Path configuration
    CONFIG_PATH = Path(os.getenv('LEEQ_CONFIG_PATH', 'configs/default.json'))
    CALIBRATION_LOG_PATH = Path(os.getenv('LEEQ_CALIBRATION_LOG_PATH', './calibration_logs'))

    # Development configuration
    DEBUG_MODE = os.getenv('LEEQ_DEBUG', 'false').lower() == 'true'

    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration settings and create necessary directories.

        This method checks that all configured paths are valid and creates
        any missing directories. It should be called during application
        initialization to ensure the environment is properly set up.

        Returns
        -------
        bool
            True if validation succeeds, raises exception on failure.

        Raises
        ------
        ValueError
            If LOG_LEVEL is not a valid logging level.

        Examples
        --------
        >>> Config.validate()
        True
        """
        # Validate log level
        valid_log_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if cls.LOG_LEVEL.upper() not in valid_log_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: {cls.LOG_LEVEL}. "
                f"Must be one of {valid_log_levels}"
            )

        # Create config directory if it doesn't exist
        if not cls.CONFIG_PATH.parent.exists():
            cls.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Create calibration log directory if it doesn't exist
        if not cls.CALIBRATION_LOG_PATH.exists():
            cls.CALIBRATION_LOG_PATH.mkdir(parents=True, exist_ok=True)

        return True

    @classmethod
    def get_config_dict(cls) -> dict:
        """
        Get all configuration values as a dictionary.

        This method is useful for debugging and logging the current
        configuration state.

        Returns
        -------
        dict
            Dictionary containing all configuration key-value pairs.

        Examples
        --------
        >>> config_dict = Config.get_config_dict()
        >>> 'LOG_LEVEL' in config_dict
        True
        """
        return {
            'LOG_LEVEL': cls.LOG_LEVEL,
            'SUPPRESS_LOGGING': cls.SUPPRESS_LOGGING,
            'CONFIG_PATH': str(cls.CONFIG_PATH),
            'CALIBRATION_LOG_PATH': str(cls.CALIBRATION_LOG_PATH),
            'DEBUG_MODE': cls.DEBUG_MODE,
        }

    @classmethod
    def update_from_env(cls) -> None:
        """
        Re-read configuration from environment variables.

        This method allows runtime updates of configuration values
        if environment variables have changed.

        Examples
        --------
        >>> import os
        >>> os.environ['LEEQ_LOG_LEVEL'] = 'DEBUG'
        >>> Config.update_from_env()
        >>> Config.LOG_LEVEL
        'DEBUG'
        """
        cls.LOG_LEVEL = os.getenv('LEEQ_LOG_LEVEL', 'INFO')
        cls.SUPPRESS_LOGGING = os.getenv('LEEQ_SUPPRESS_LOGGING', 'false').lower() == 'true'
        cls.CONFIG_PATH = Path(os.getenv('LEEQ_CONFIG_PATH', 'configs/default.json'))
        cls.CALIBRATION_LOG_PATH = Path(os.getenv('LEEQ_CALIBRATION_LOG_PATH', './calibration_logs'))
        cls.DEBUG_MODE = os.getenv('LEEQ_DEBUG', 'false').lower() == 'true'
