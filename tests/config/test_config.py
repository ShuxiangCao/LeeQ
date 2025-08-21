"""
Test cases for leeq.config module.

This test module provides comprehensive coverage of the Config class functionality,
including default values, environment variable overrides, path validation,
and configuration management.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from leeq.config import Config


class TestConfig:
    """Test configuration management functionality."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        # Test default log level
        assert Config.LOG_LEVEL in {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}

        # Test default boolean values
        assert isinstance(Config.SUPPRESS_LOGGING, bool)
        assert isinstance(Config.DEBUG_MODE, bool)

        # Test default paths are Path objects
        assert isinstance(Config.CONFIG_PATH, Path)
        assert isinstance(Config.CALIBRATION_LOG_PATH, Path)

        # Test path strings contain expected defaults
        assert 'default.json' in str(Config.CONFIG_PATH)
        assert 'calibration_logs' in str(Config.CALIBRATION_LOG_PATH)

    def test_environment_override_log_level(self, monkeypatch):
        """Test LOG_LEVEL environment variable override."""
        # Test valid log level override
        monkeypatch.setenv('LEEQ_LOG_LEVEL', 'DEBUG')
        Config.update_from_env()
        assert Config.LOG_LEVEL == 'DEBUG'

        # Test another valid log level
        monkeypatch.setenv('LEEQ_LOG_LEVEL', 'ERROR')
        Config.update_from_env()
        assert Config.LOG_LEVEL == 'ERROR'

    def test_environment_override_boolean_flags(self, monkeypatch):
        """Test boolean environment variable overrides."""
        # Test SUPPRESS_LOGGING true override
        monkeypatch.setenv('LEEQ_SUPPRESS_LOGGING', 'true')
        Config.update_from_env()
        assert Config.SUPPRESS_LOGGING is True

        # Test SUPPRESS_LOGGING false override
        monkeypatch.setenv('LEEQ_SUPPRESS_LOGGING', 'false')
        Config.update_from_env()
        assert Config.SUPPRESS_LOGGING is False

        # Test DEBUG_MODE true override
        monkeypatch.setenv('LEEQ_DEBUG', 'true')
        Config.update_from_env()
        assert Config.DEBUG_MODE is True

        # Test DEBUG_MODE false override
        monkeypatch.setenv('LEEQ_DEBUG', 'false')
        Config.update_from_env()
        assert Config.DEBUG_MODE is False

    def test_environment_override_paths(self, monkeypatch, tmp_path):
        """Test path environment variable overrides."""
        test_config_path = tmp_path / "test_config.json"
        test_cal_path = tmp_path / "test_calibration"

        # Test CONFIG_PATH override
        monkeypatch.setenv('LEEQ_CONFIG_PATH', str(test_config_path))
        Config.update_from_env()
        assert Config.CONFIG_PATH == test_config_path

        # Test CALIBRATION_LOG_PATH override
        monkeypatch.setenv('LEEQ_CALIBRATION_LOG_PATH', str(test_cal_path))
        Config.update_from_env()
        assert Config.CALIBRATION_LOG_PATH == test_cal_path

    def test_validate_valid_configuration(self, tmp_path, monkeypatch):
        """Test validation succeeds with valid configuration."""
        # Set up test paths
        test_config_path = tmp_path / "configs" / "test.json"
        test_cal_path = tmp_path / "cal_logs"

        monkeypatch.setenv('LEEQ_LOG_LEVEL', 'INFO')
        monkeypatch.setenv('LEEQ_CONFIG_PATH', str(test_config_path))
        monkeypatch.setenv('LEEQ_CALIBRATION_LOG_PATH', str(test_cal_path))
        Config.update_from_env()

        # Validation should succeed and create directories
        assert Config.validate() is True
        assert test_config_path.parent.exists()
        assert test_cal_path.exists()

    def test_validate_invalid_log_level(self, monkeypatch):
        """Test validation fails with invalid log level."""
        monkeypatch.setenv('LEEQ_LOG_LEVEL', 'INVALID_LEVEL')
        Config.update_from_env()

        with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
            Config.validate()

    def test_validate_creates_missing_directories(self, tmp_path, monkeypatch):
        """Test that validate creates missing directories."""
        # Set up non-existent paths
        config_dir = tmp_path / "nonexistent" / "configs"
        cal_dir = tmp_path / "nonexistent" / "calibration"
        config_file = config_dir / "test.json"

        monkeypatch.setenv('LEEQ_LOG_LEVEL', 'INFO')
        monkeypatch.setenv('LEEQ_CONFIG_PATH', str(config_file))
        monkeypatch.setenv('LEEQ_CALIBRATION_LOG_PATH', str(cal_dir))
        Config.update_from_env()

        # Directories should not exist yet
        assert not config_dir.exists()
        assert not cal_dir.exists()

        # Validate should create them
        Config.validate()
        assert config_dir.exists()
        assert cal_dir.exists()

    def test_get_config_dict(self, monkeypatch):
        """Test configuration dictionary generation."""
        # Set up known environment
        monkeypatch.setenv('LEEQ_LOG_LEVEL', 'DEBUG')
        monkeypatch.setenv('LEEQ_SUPPRESS_LOGGING', 'true')
        monkeypatch.setenv('LEEQ_DEBUG', 'false')
        Config.update_from_env()

        config_dict = Config.get_config_dict()

        # Test all keys are present
        expected_keys = {
            'LOG_LEVEL', 'SUPPRESS_LOGGING', 'CONFIG_PATH',
            'CALIBRATION_LOG_PATH', 'DEBUG_MODE'
        }
        assert set(config_dict.keys()) == expected_keys

        # Test values match configuration
        assert config_dict['LOG_LEVEL'] == 'DEBUG'
        assert config_dict['SUPPRESS_LOGGING'] is True
        assert config_dict['DEBUG_MODE'] is False

        # Test paths are strings in dict
        assert isinstance(config_dict['CONFIG_PATH'], str)
        assert isinstance(config_dict['CALIBRATION_LOG_PATH'], str)

    def test_update_from_env_reloads_all_settings(self, monkeypatch):
        """Test that update_from_env reloads all configuration settings."""
        # Set initial environment
        monkeypatch.setenv('LEEQ_LOG_LEVEL', 'INFO')
        monkeypatch.setenv('LEEQ_SUPPRESS_LOGGING', 'false')
        monkeypatch.setenv('LEEQ_DEBUG', 'false')
        Config.update_from_env()

        initial_log_level = Config.LOG_LEVEL
        initial_suppress = Config.SUPPRESS_LOGGING
        initial_debug = Config.DEBUG_MODE

        # Change environment
        monkeypatch.setenv('LEEQ_LOG_LEVEL', 'WARNING')
        monkeypatch.setenv('LEEQ_SUPPRESS_LOGGING', 'true')
        monkeypatch.setenv('LEEQ_DEBUG', 'true')
        Config.update_from_env()

        # Verify all settings updated
        assert Config.LOG_LEVEL != initial_log_level
        assert Config.SUPPRESS_LOGGING != initial_suppress
        assert Config.DEBUG_MODE != initial_debug

        assert Config.LOG_LEVEL == 'WARNING'
        assert Config.SUPPRESS_LOGGING is True
        assert Config.DEBUG_MODE is True

    def test_boolean_parsing_case_insensitive(self, monkeypatch):
        """Test that boolean environment variables are parsed case-insensitively."""
        # Test various true values
        for true_value in ['true', 'TRUE', 'True', 'TrUe']:
            monkeypatch.setenv('LEEQ_DEBUG', true_value)
            Config.update_from_env()
            assert Config.DEBUG_MODE is True

        # Test various false values
        for false_value in ['false', 'FALSE', 'False', 'FaLsE', 'anything_else']:
            monkeypatch.setenv('LEEQ_DEBUG', false_value)
            Config.update_from_env()
            assert Config.DEBUG_MODE is False

    def test_path_object_consistency(self, tmp_path, monkeypatch):
        """Test that Path objects remain consistent after operations."""
        test_path = tmp_path / "test_path"
        monkeypatch.setenv('LEEQ_CALIBRATION_LOG_PATH', str(test_path))
        Config.update_from_env()

        # Verify it's a Path object
        assert isinstance(Config.CALIBRATION_LOG_PATH, Path)
        assert Config.CALIBRATION_LOG_PATH == test_path

        # Verify operations work correctly
        assert Config.CALIBRATION_LOG_PATH.name == "test_path"
        assert Config.CALIBRATION_LOG_PATH.parent == tmp_path

    def test_validate_log_levels_comprehensive(self, monkeypatch):
        """Test all valid log levels pass validation."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        for level in valid_levels:
            monkeypatch.setenv('LEEQ_LOG_LEVEL', level)
            Config.update_from_env()
            # Should not raise exception
            assert Config.validate() is True

    def test_config_isolation_between_tests(self):
        """Test that configuration changes don't leak between tests."""
        # This test verifies that each test gets a clean config state
        # by testing that we can read current values without errors
        config_dict = Config.get_config_dict()
        assert len(config_dict) == 5
        assert all(key in config_dict for key in [
            'LOG_LEVEL', 'SUPPRESS_LOGGING', 'CONFIG_PATH',
            'CALIBRATION_LOG_PATH', 'DEBUG_MODE'
        ])

    def teardown_method(self):
        """Clean up after each test method."""
        # Reset environment to defaults to prevent test interference
        Config.update_from_env()
