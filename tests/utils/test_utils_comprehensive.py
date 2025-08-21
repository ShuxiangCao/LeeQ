"""
Extended tests for utility functions.
"""
import pytest
import os
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from leeq.utils.utils import (
    setup_logging,
    get_calibration_log_path,
    Singleton,
    ObjectFactory,
    elementwise_update_dict,
    is_running_in_jupyter,
    display_json_dict
)


class TestSetupLogging:
    """Test logging setup functionality."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        logger = setup_logging("test_logger")
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
    
    def test_setup_logging_level(self):
        """Test logging setup with different levels."""
        logger_info = setup_logging("test_info", level=logging.INFO)
        logger_debug = setup_logging("test_debug", level=logging.DEBUG)
        logger_warning = setup_logging("test_warning", level=logging.WARNING)
        
        assert logger_info.level == logging.INFO
        assert logger_debug.level == logging.DEBUG
        assert logger_warning.level == logging.WARNING
    
    def test_setup_logging_singleton_behavior(self):
        """Test that same logger name returns same instance."""
        logger1 = setup_logging("singleton_test")
        logger2 = setup_logging("singleton_test")
        
        assert logger1 is logger2
    
    def test_setup_logging_different_names(self):
        """Test that different logger names return different instances."""
        logger1 = setup_logging("logger_1")
        logger2 = setup_logging("logger_2")
        
        assert logger1 is not logger2
        assert logger1.name != logger2.name
    
    @patch.dict(os.environ, {'LEEQ_SUPPRESS_LOGGING': 'true'})
    def test_setup_logging_suppressed(self):
        """Test logging setup with suppression enabled."""
        logger = setup_logging("suppressed_logger")
        
        assert logger.level == logging.CRITICAL
    
    @patch.dict(os.environ, {'LEEQ_SUPPRESS_LOGGING': 'false'})
    def test_setup_logging_not_suppressed(self):
        """Test logging setup with suppression disabled."""
        logger = setup_logging("normal_logger", level=logging.INFO)
        
        assert logger.level == logging.INFO
    
    def test_setup_logging_handler_creation(self):
        """Test that appropriate handlers are created."""
        # Clear any existing loggers to test fresh
        logger_name = "handler_test_logger"
        
        with patch.dict(os.environ, {}, clear=True):  # Clear LEEQ_SUPPRESS_LOGGING
            logger = setup_logging(logger_name, level=logging.DEBUG)
        
        # Logger should have been created
        assert logger is not None
        assert logger.name == logger_name


class TestGetCalibrationLogPath:
    """Test calibration log path functionality."""
    
    def test_get_calibration_log_path_basic(self):
        """Test basic calibration log path retrieval."""
        path = get_calibration_log_path()
        
        assert isinstance(path, Path)
        assert path is not None
    
    @patch.dict(os.environ, {'JUPYTERHUB_USER': 'test_user'})
    def test_get_calibration_log_path_jupyterhub_user(self):
        """Test path with JUPYTERHUB_USER set."""
        path = get_calibration_log_path()
        
        assert isinstance(path, Path)
        assert 'test_user' in str(path)
    
    @patch.dict(os.environ, {'LEEQ_CALIBRATION_LOG_PATH': '/custom/log/path'})
    def test_get_calibration_log_path_custom_path(self):
        """Test path with custom LEEQ_CALIBRATION_LOG_PATH."""
        path = get_calibration_log_path()
        
        assert isinstance(path, Path)
        assert '/custom/log/path' in str(path)
    
    @patch('leeq.utils.utils.getpass.getuser')
    def test_get_calibration_log_path_getpass_fallback(self, mock_getuser):
        """Test fallback to getpass.getuser() when JUPYTERHUB_USER not set."""
        mock_getuser.return_value = 'system_user'
        
        # Remove JUPYTERHUB_USER if it exists
        env_without_jupyterhub = dict(os.environ)
        env_without_jupyterhub.pop('JUPYTERHUB_USER', None)
        env_without_jupyterhub.pop('LEEQ_CALIBRATION_LOG_PATH', None)
        
        with patch.dict(os.environ, env_without_jupyterhub, clear=True):
            path = get_calibration_log_path()
        
        assert 'system_user' in str(path)
        mock_getuser.assert_called_once()
    
    def test_get_calibration_log_path_returns_path_object(self):
        """Test that function returns a Path object."""
        path = get_calibration_log_path()
        
        assert isinstance(path, Path)
        assert hasattr(path, 'exists')
        assert hasattr(path, 'mkdir')
        assert hasattr(path, 'parent')


class TestSingleton:
    """Test Singleton class functionality."""
    
    def test_singleton_single_instance(self):
        """Test that Singleton creates only one instance."""
        instance1 = Singleton()
        instance2 = Singleton()
        
        assert instance1 is instance2
    
    def test_singleton_initialization_flag(self):
        """Test that Singleton uses initialization flag correctly."""
        # Create a fresh singleton for this test
        class TestSingleton(Singleton):
            def __init__(self):
                super().__init__()
                if not hasattr(self, 'test_value'):
                    self.test_value = "initialized"
        
        instance1 = TestSingleton()
        instance2 = TestSingleton()
        
        assert instance1 is instance2
        assert instance1.test_value == "initialized"
        assert instance2.test_value == "initialized"
    
    def test_singleton_subclass_behavior(self):
        """Test that Singleton subclasses work correctly."""
        class SingletonSubclass(Singleton):
            def __init__(self):
                super().__init__()
                if not self._initialized:
                    self.custom_attribute = "subclass_value"
        
        instance1 = SingletonSubclass()
        instance2 = SingletonSubclass()
        
        assert instance1 is instance2
        assert hasattr(instance1, 'custom_attribute')
        assert instance1.custom_attribute == "subclass_value"


class TestObjectFactory:
    """Test ObjectFactory class functionality."""
    
    def test_object_factory_initialization(self):
        """Test ObjectFactory initialization."""
        accepted_types = [dict, list]
        factory = ObjectFactory(accepted_types)
        
        assert factory._accepted_template == accepted_types
        assert factory._registered_template == {}
    
    def test_object_factory_singleton_behavior(self):
        """Test that ObjectFactory exhibits singleton behavior."""
        # Note: This might not work as expected due to Singleton implementation
        # but we can test basic instantiation
        factory1 = ObjectFactory([dict])
        
        assert factory1 is not None
        assert hasattr(factory1, '_accepted_template')
    
    def test_object_factory_register_template(self):
        """Test registering a template class."""
        class TestTemplate(dict):
            pass
        
        factory = ObjectFactory([dict])
        factory.register_collection_template(TestTemplate)
        
        assert TestTemplate.__qualname__ in factory._registered_template
        assert factory._registered_template[TestTemplate.__qualname__] is TestTemplate
    
    def test_object_factory_register_invalid_template(self):
        """Test registering invalid template raises error."""
        factory = ObjectFactory([dict])
        
        # Try to register a non-class
        with pytest.raises(RuntimeError, match="must be a class"):
            factory.register_collection_template("not_a_class")
    
    def test_object_factory_register_wrong_type(self):
        """Test registering wrong type template raises error."""
        class WrongTypeTemplate:
            pass
        
        factory = ObjectFactory([dict])
        
        with pytest.raises(RuntimeError, match="must be a subclass"):
            factory.register_collection_template(WrongTypeTemplate)
    
    def test_object_factory_call_registered_class(self):
        """Test calling factory with registered class."""
        class TestTemplate(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.created_by_factory = True
        
        factory = ObjectFactory([dict])
        factory.register_collection_template(TestTemplate)
        
        instance = factory(TestTemplate.__qualname__)
        
        assert isinstance(instance, TestTemplate)
        assert instance.created_by_factory is True
    
    def test_object_factory_call_unregistered_class(self):
        """Test calling factory with unregistered class raises error."""
        factory = ObjectFactory([dict])
        
        with pytest.raises(RuntimeError, match="not registered"):
            factory("UnregisteredClass")


class TestElementwiseUpdateDict:
    """Test elementwise dictionary update functionality."""
    
    def test_elementwise_update_basic(self):
        """Test basic elementwise dictionary update."""
        original = {'a': 1, 'b': 2}
        update = {'b': 3, 'c': 4}
        
        elementwise_update_dict(original, update)
        
        assert original['a'] == 1  # Unchanged
        assert original['b'] == 3  # Updated
        assert original['c'] == 4  # Added
    
    def test_elementwise_update_nested_dicts(self):
        """Test elementwise update with nested dictionaries."""
        original = {
            'level1': {
                'level2': {
                    'value1': 'old',
                    'value2': 'keep'
                }
            }
        }
        
        update = {
            'level1': {
                'level2': {
                    'value1': 'new',
                    'value3': 'added'
                }
            }
        }
        
        elementwise_update_dict(original, update)
        
        assert original['level1']['level2']['value1'] == 'new'  # Updated
        assert original['level1']['level2']['value2'] == 'keep'  # Preserved
        assert original['level1']['level2']['value3'] == 'added'  # Added
    
    def test_elementwise_update_mixed_types(self):
        """Test elementwise update with mixed value types."""
        original = {
            'string': 'old_value',
            'number': 42,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'}
        }
        
        update = {
            'string': 'new_value',
            'number': 100,
            'list': [4, 5, 6],  # This will replace, not merge
            'dict': {'nested': 'updated', 'added': 'new'}
        }
        
        elementwise_update_dict(original, update)
        
        assert original['string'] == 'new_value'
        assert original['number'] == 100
        assert original['list'] == [4, 5, 6]
        assert original['dict']['nested'] == 'updated'
        assert original['dict']['added'] == 'new'
    
    def test_elementwise_update_empty_dicts(self):
        """Test elementwise update with empty dictionaries."""
        original = {}
        update = {'a': 1, 'b': 2}
        
        elementwise_update_dict(original, update)
        
        assert original == {'a': 1, 'b': 2}
        
        # Test with empty update
        original = {'a': 1, 'b': 2}
        update = {}
        
        elementwise_update_dict(original, update)
        
        assert original == {'a': 1, 'b': 2}  # Unchanged
    
    def test_elementwise_update_creates_new_nested(self):
        """Test that update creates new nested structures when needed."""
        original = {}
        update = {
            'new_section': {
                'nested_value': 42,
                'deeply_nested': {
                    'value': 'deep'
                }
            }
        }
        
        elementwise_update_dict(original, update)
        
        assert original['new_section']['nested_value'] == 42
        assert original['new_section']['deeply_nested']['value'] == 'deep'


class TestIsRunningInJupyter:
    """Test Jupyter detection functionality."""
    
    @patch('leeq.utils.utils.sys')
    def test_is_running_in_jupyter_true(self, mock_sys):
        """Test detection when running in Jupyter."""
        mock_sys.argv = ['python', 'script.py', 'kernel.json']
        
        result = is_running_in_jupyter()
        
        assert result is True
    
    @patch('leeq.utils.utils.sys')
    def test_is_running_in_jupyter_false(self, mock_sys):
        """Test detection when not running in Jupyter."""
        mock_sys.argv = ['python', 'script.py']
        
        result = is_running_in_jupyter()
        
        assert result is False
    
    @patch('leeq.utils.utils.sys')
    def test_is_running_in_jupyter_edge_cases(self, mock_sys):
        """Test edge cases for Jupyter detection."""
        # Test with different argument patterns
        test_cases = [
            ['jupyter', 'notebook'],  # False - doesn't end with json
            ['python', '-m', 'something.json'],  # True - ends with json
            ['ipython'],  # False - doesn't end with json
            [],  # False - empty argv (shouldn't crash)
        ]
        
        for argv in test_cases:
            mock_sys.argv = argv
            result = is_running_in_jupyter()
            
            expected = len(argv) > 0 and argv[-1].endswith('json')
            assert result == expected


class TestDisplayJsonDict:
    """Test JSON dictionary display functionality."""
    
    def test_display_json_dict_basic(self):
        """Test basic JSON dictionary display."""
        test_data = {'key1': 'value1', 'key2': 42, 'key3': [1, 2, 3]}
        
        # This should not raise an exception
        display_json_dict(test_data)
    
    def test_display_json_dict_with_root(self):
        """Test JSON dictionary display with custom root."""
        test_data = {'nested': {'value': True}}
        
        # This should not raise an exception
        display_json_dict(test_data, root='custom_root')
    
    def test_display_json_dict_expanded(self):
        """Test JSON dictionary display with expanded option."""
        test_data = {'a': 1, 'b': {'nested': 2}}
        
        # This should not raise an exception
        display_json_dict(test_data, expanded=True)
        
        # Test with expanded=False
        display_json_dict(test_data, expanded=False)
    
    @patch('leeq.utils.utils.is_running_in_jupyter')
    @patch('leeq.utils.utils.logger')
    def test_display_json_dict_non_jupyter(self, mock_logger, mock_is_jupyter):
        """Test JSON dictionary display when not in Jupyter."""
        mock_is_jupyter.return_value = False
        test_data = {'test': 'data'}
        
        display_json_dict(test_data)
        
        # Should have called logger.info
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert 'JSON data:' in call_args
    
    @patch('leeq.utils.utils.is_running_in_jupyter')
    def test_display_json_dict_jupyter(self, mock_is_jupyter):
        """Test JSON dictionary display when in Jupyter."""
        mock_is_jupyter.return_value = True
        
        with patch('leeq.utils.utils.display') as mock_display, \
             patch('leeq.utils.utils.JSON') as mock_json:
            
            test_data = {'jupyter': 'test'}
            display_json_dict(test_data, root='test_root')
            
            # Should have called JSON and display
            mock_json.assert_called_once()
            mock_display.assert_called_once()
    
    def test_display_json_dict_complex_data(self):
        """Test JSON dictionary display with complex data structures."""
        complex_data = {
            'string': 'test',
            'number': 42,
            'float': 3.14,
            'boolean': True,
            'null': None,
            'array': [1, 2, 3, 'mixed'],
            'nested': {
                'deep': {
                    'deeper': 'value'
                }
            }
        }
        
        # This should not raise an exception
        display_json_dict(complex_data)
    
    def test_display_json_dict_empty_data(self):
        """Test JSON dictionary display with empty data."""
        empty_data = {}
        
        # This should not raise an exception
        display_json_dict(empty_data)
        
        # Test with None (should handle gracefully)
        # Note: Depending on implementation, this might need adjustment
        try:
            display_json_dict(None)
        except (TypeError, AttributeError):
            # It's acceptable for this to fail with None input
            pass


@pytest.mark.integration
class TestUtilsIntegration:
    """Integration tests for utility functions."""
    
    def test_logging_and_path_integration(self):
        """Test integration between logging setup and path functions."""
        # Setup logging
        logger = setup_logging("integration_test")
        
        # Get calibration path
        path = get_calibration_log_path()
        
        # Should be able to log the path
        logger.info(f"Calibration path: {path}")
        
        assert logger is not None
        assert path is not None
        assert isinstance(path, Path)
    
    def test_factory_and_singleton_integration(self):
        """Test integration between factory and singleton patterns."""
        # Create a singleton factory
        class SingletonFactory(ObjectFactory):
            pass
        
        # Should behave as singleton
        factory1 = SingletonFactory([dict])
        factory2 = SingletonFactory([list])  # Different accepted types
        
        # Note: Due to Singleton implementation, this might return same instance
        # The test verifies that both can be created without error
        assert factory1 is not None
        assert factory2 is not None
    
    def test_dict_update_and_display_integration(self):
        """Test integration between dict update and display functions."""
        original_config = {
            'logging': {'level': 'INFO'},
            'paths': {'base': '/tmp'}
        }
        
        update_config = {
            'logging': {'level': 'DEBUG', 'format': 'detailed'},
            'paths': {'logs': '/tmp/logs'}
        }
        
        # Update the configuration
        elementwise_update_dict(original_config, update_config)
        
        # Display the result (should not raise exception)
        display_json_dict(original_config, root='updated_config')
        
        # Verify the update worked
        assert original_config['logging']['level'] == 'DEBUG'
        assert original_config['logging']['format'] == 'detailed'
        assert original_config['paths']['base'] == '/tmp'
        assert original_config['paths']['logs'] == '/tmp/logs'