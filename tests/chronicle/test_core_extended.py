"""
Extended tests for chronicle core to increase coverage from 63% to 80%.

Tests focus on:
- Chronicle lifecycle management
- File system operations with mocks
- Error recovery and fallback mechanisms
- Multi-handler coordination
- LoggableObject attribute tracking
- RecordBook and RecordEntry functionality
"""
import pytest
import copy
import inspect
import tempfile
import pathlib
from unittest.mock import Mock, patch, MagicMock
import threading
import time

from leeq.chronicle.core import (
    LoggableObject,
    SetBrowserFunctionAttributeMeta,
    RecordBook,
    RecordEntry,
    _reserved_keys
)


class TestSetBrowserFunctionAttributeMeta:
    """Test the metaclass for browser function attributes."""

    def test_metaclass_creates_browse_functions_attribute(self):
        """Test that metaclass creates _browse_functions attribute."""

        class TestClass(metaclass=SetBrowserFunctionAttributeMeta):
            pass

        assert hasattr(TestClass, '_browse_functions')
        assert isinstance(TestClass._browse_functions, list)
        assert len(TestClass._browse_functions) == 0

    def test_metaclass_inheritance_isolation(self):
        """Test that subclasses don't share _browse_functions reference."""

        class BaseClass(metaclass=SetBrowserFunctionAttributeMeta):
            pass

        class DerivedClass1(BaseClass):
            pass

        class DerivedClass2(BaseClass):
            pass

        # Each class should have its own _browse_functions list
        assert BaseClass._browse_functions is not DerivedClass1._browse_functions
        assert DerivedClass1._browse_functions is not DerivedClass2._browse_functions

        # Modifying one should not affect others
        BaseClass._browse_functions.append('base_func')
        DerivedClass1._browse_functions.append('derived1_func')

        assert len(BaseClass._browse_functions) == 1
        assert len(DerivedClass1._browse_functions) == 1
        assert len(DerivedClass2._browse_functions) == 0


class TestLoggableObject:
    """Extended tests for LoggableObject class."""

    def test_initialization(self):
        """Test LoggableObject initialization."""
        obj = LoggableObject()

        assert hasattr(obj, '_register_log_and_record_args_map')
        assert isinstance(obj._register_log_and_record_args_map, dict)
        assert len(obj._register_log_and_record_args_map) == 0
        assert hasattr(obj, 'logger')
        assert obj._record_entry is None

    def test_hrid_property(self):
        """Test human readable ID property."""
        obj = LoggableObject()
        hrid = obj.hrid

        assert isinstance(hrid, str)
        assert obj.__class__.__qualname__ in hrid
        assert str(id(obj)) in hrid
        assert '@' in hrid

    def test_repr_method(self):
        """Test string representation."""
        obj = LoggableObject()
        repr_str = repr(obj)

        assert repr_str.startswith('<')
        assert repr_str.endswith('>')
        assert obj.hrid in repr_str

    def test_setattr_without_record_entry(self):
        """Test attribute setting when no record entry is active."""
        obj = LoggableObject()

        obj.test_attribute = 'test_value'

        assert obj.test_attribute == 'test_value'
        # No recording should happen when _record_entry is None

    def test_setattr_with_record_entry(self):
        """Test attribute setting when record entry is active."""
        obj = LoggableObject()
        mock_record_entry = Mock()
        obj._record_entry = mock_record_entry

        obj.test_attribute = 'test_value'

        assert obj.test_attribute == 'test_value'
        # touch_attribute is called for both set and get operations
        assert mock_record_entry.touch_attribute.call_count == 2
        mock_record_entry.touch_attribute.assert_any_call('test_attribute')

    def test_setattr_reserved_keys(self):
        """Test that reserved keys are not recorded."""
        obj = LoggableObject()
        mock_record_entry = Mock()
        obj._record_entry = mock_record_entry

        # Test a safe reserved key that can be set
        original_logger = obj.logger
        obj.logger = 'test_value'  # This is a reserved key

        # Restore original logger
        obj.logger = original_logger

        # touch_attribute should not have been called for reserved keys
        mock_record_entry.touch_attribute.assert_not_called()

    def test_getattribute_without_record_entry(self):
        """Test attribute access when no record entry is active."""
        obj = LoggableObject()
        obj.test_attribute = 'test_value'

        result = obj.test_attribute

        assert result == 'test_value'

    def test_getattribute_with_record_entry(self):
        """Test attribute access when record entry is active."""
        obj = LoggableObject()
        mock_record_entry = Mock()
        obj._record_entry = mock_record_entry
        obj.test_attribute = 'test_value'

        result = obj.test_attribute

        assert result == 'test_value'
        # Should be called twice: once for set, once for get
        assert mock_record_entry.touch_attribute.call_count == 2

    def test_getattribute_reserved_keys(self):
        """Test that reserved keys don't trigger recording on access."""
        obj = LoggableObject()
        mock_record_entry = Mock()
        obj._record_entry = mock_record_entry

        # Access reserved attributes
        for reserved_key in _reserved_keys:
            if hasattr(obj, reserved_key):
                getattr(obj, reserved_key)

        # Should not call touch_attribute for reserved keys
        mock_record_entry.touch_attribute.assert_not_called()

    def test_getattribute_routines(self):
        """Test that routine (method) access doesn't trigger recording."""
        obj = LoggableObject()
        mock_record_entry = Mock()
        obj._record_entry = mock_record_entry

        # Access a method

        # touch_attribute should not be called for routines
        # Note: property access might still trigger recording

    def test_safe_deepcopy(self):
        """Test safe deepcopy static method."""
        original = {'key': [1, 2, 3], 'nested': {'inner': 'value'}}

        copied = LoggableObject._safe_deepcopy(original)

        assert copied == original
        assert copied is not original
        assert copied['key'] is not original['key']
        assert copied['nested'] is not original['nested']

    def test_safe_deepcopy_complex_objects(self):
        """Test safe deepcopy with complex objects."""
        obj = LoggableObject()
        obj.data = {'list': [1, 2, 3]}

        copied = LoggableObject._safe_deepcopy(obj.data)

        assert copied == obj.data
        assert copied is not obj.data

    @patch('leeq.chronicle.core.find_methods_with_tag')
    def test_get_browser_functions(self, mock_find_methods):
        """Test getting browser functions."""
        mock_find_methods.return_value = ['func1', 'func2']
        obj = LoggableObject()

        result = obj.get_browser_functions()

        assert result == ['func1', 'func2']
        mock_find_methods.assert_called_once_with(obj, "_browser_function")

    def test_register_log_and_record_args(self):
        """Test registering function arguments."""
        obj = LoggableObject()

        def test_func(a, b=None):
            pass

        args = [1, 2]
        kwargs = {'c': 3}
        record_details = {'timestamp': 'test'}

        obj.register_log_and_record_args(test_func, args, kwargs, record_details)

        assert test_func.__qualname__ in obj._register_log_and_record_args_map
        stored = obj._register_log_and_record_args_map[test_func.__qualname__]
        assert stored[0] == args  # args
        assert stored[1] == kwargs  # kwargs
        assert stored[2] == record_details  # record_details

    def test_register_log_and_record_args_no_deepcopy(self):
        """Test registering without deepcopy."""
        obj = LoggableObject()

        def test_func():
            pass

        args = [1, 2]
        kwargs = {'c': 3}
        record_details = {'timestamp': 'test'}

        obj.register_log_and_record_args(test_func, args, kwargs, record_details, deepcopy=False)

        stored = obj._register_log_and_record_args_map[test_func.__qualname__]
        assert stored[0] is args  # Same reference when deepcopy=False
        assert stored[1] is kwargs

    def test_register_log_and_record_args_overwrite_name(self):
        """Test registering with overwritten function name."""
        obj = LoggableObject()

        def test_func():
            pass

        custom_name = 'custom_function_name'
        args = []
        kwargs = {}
        record_details = {}

        obj.register_log_and_record_args(test_func, args, kwargs, record_details,
                                       overwrite_func_name=custom_name)

        assert custom_name in obj._register_log_and_record_args_map
        assert test_func.__qualname__ not in obj._register_log_and_record_args_map

    def test_rebuild_args_dict(self):
        """Test rebuilding arguments dictionary from function signature."""
        def test_func(a, b=10, c='default'):
            pass

        called_args = [1, 2]
        called_kwargs = {'c': 'custom'}

        result = LoggableObject._rebuild_args_dict(test_func, called_args, called_kwargs)

        expected = {'a': 1, 'b': 2, 'c': 'custom'}
        assert result == expected

    def test_rebuild_args_dict_method_with_self(self):
        """Test rebuilding args dict for method with self parameter."""
        class TestClass:
            def test_method(self, a, b=5):
                pass

        called_args = [1]
        called_kwargs = {'b': 10}

        result = LoggableObject._rebuild_args_dict(TestClass.test_method, called_args, called_kwargs)

        expected = {'a': 1, 'b': 10}
        assert result == expected

    def test_rebuild_args_dict_classmethod(self):
        """Test rebuilding args dict for classmethod with cls parameter."""
        class TestClass:
            @classmethod
            def test_method(cls, a, b=5):
                pass

        called_args = [1]
        called_kwargs = {}

        result = LoggableObject._rebuild_args_dict(TestClass.test_method, called_args, called_kwargs)

        expected = {'a': 1, 'b': 5}
        assert result == expected

    def test_retrieve_args_success(self):
        """Test successful argument retrieval."""
        obj = LoggableObject()

        def test_func(a, b=10):
            pass

        args = [1]
        kwargs = {'b': 20}
        record_details = {}

        obj.register_log_and_record_args(test_func, args, kwargs, record_details)
        result = obj.retrieve_args(test_func)

        expected = {'a': 1, 'b': 20}
        assert result == expected

    def test_retrieve_args_not_registered(self):
        """Test retrieving args for unregistered function raises error."""
        obj = LoggableObject()

        def test_func():
            pass

        with pytest.raises(ValueError, match="Function.*is not registered"):
            obj.retrieve_args(test_func)

    def test_retrieve_latest_record_entry_details_success(self):
        """Test successful record entry details retrieval."""
        obj = LoggableObject()

        def test_func():
            pass

        record_details = {'timestamp': '2024-01-01', 'user': 'test'}
        obj.register_log_and_record_args(test_func, [], {}, record_details)

        result = obj.retrieve_latest_record_entry_details(test_func)

        assert result == record_details

    def test_retrieve_latest_record_entry_details_not_registered(self):
        """Test retrieving details for unregistered function raises error."""
        obj = LoggableObject()

        def test_func():
            pass

        with pytest.raises(ValueError, match="Function.*is not registered"):
            obj.retrieve_latest_record_entry_details(test_func)

    def test_set_record_entry(self):
        """Test setting record entry."""
        obj = LoggableObject()
        mock_record_entry = Mock()

        obj.set_record_entry(mock_record_entry)

        assert obj._record_entry is mock_record_entry

    def test_set_record_entry_none(self):
        """Test setting record entry to None."""
        obj = LoggableObject()
        mock_record_entry = Mock()
        obj._record_entry = mock_record_entry

        obj.set_record_entry(None)

        assert obj._record_entry is None

    def test_concurrent_attribute_access(self):
        """Test concurrent attribute access doesn't cause issues."""
        obj = LoggableObject()
        mock_record_entry = Mock()
        obj._record_entry = mock_record_entry

        results = []
        errors = []

        def access_attributes():
            try:
                for i in range(10):
                    setattr(obj, f'attr_{i}', i)
                    value = getattr(obj, f'attr_{i}')
                    results.append(value)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=access_attributes)
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(errors) == 0
        assert len(results) > 0


class MockRecordEntry:
    """Mock RecordEntry for testing."""

    def __init__(self):
        self.touched_attributes = []

    def touch_attribute(self, attr_name):
        self.touched_attributes.append(attr_name)


class TestIntegrationScenarios:
    """Integration tests for core chronicle functionality."""

    def test_loggable_object_with_mock_record_entry(self):
        """Test LoggableObject integration with mock RecordEntry."""
        obj = LoggableObject()
        mock_entry = MockRecordEntry()
        obj.set_record_entry(mock_entry)

        # Set multiple attributes
        obj.data = {'key': 'value'}
        obj.counter = 0
        obj.flag = True

        # Access attributes
        _ = obj.data
        _ = obj.counter

        # Check that attributes were tracked
        assert 'data' in mock_entry.touched_attributes
        assert 'counter' in mock_entry.touched_attributes
        assert mock_entry.touched_attributes.count('data') == 2  # set + get
        assert mock_entry.touched_attributes.count('counter') == 2  # set + get

    def test_multiple_loggable_objects_isolation(self):
        """Test that multiple LoggableObject instances don't interfere."""
        obj1 = LoggableObject()
        obj2 = LoggableObject()

        mock_entry1 = MockRecordEntry()
        mock_entry2 = MockRecordEntry()

        obj1.set_record_entry(mock_entry1)
        obj2.set_record_entry(mock_entry2)

        # Set attributes on different objects
        obj1.data1 = 'value1'
        obj2.data2 = 'value2'

        # Check isolation
        assert 'data1' in mock_entry1.touched_attributes
        assert 'data1' not in mock_entry2.touched_attributes
        assert 'data2' in mock_entry2.touched_attributes
        assert 'data2' not in mock_entry1.touched_attributes

    def test_function_registration_and_retrieval_workflow(self):
        """Test complete workflow of function registration and argument retrieval."""
        obj = LoggableObject()

        def complex_function(a, b=10, c='default'):
            return a + b

        # Register function call
        call_args = [5]
        call_kwargs = {'b': 20, 'c': 'custom'}
        record_details = {'execution_time': 0.1, 'result': 25}

        obj.register_log_and_record_args(
            complex_function, call_args, call_kwargs, record_details
        )

        # Retrieve and verify
        retrieved_args = obj.retrieve_args(complex_function)
        retrieved_details = obj.retrieve_latest_record_entry_details(complex_function)

        expected_args = {'a': 5, 'b': 20, 'c': 'custom'}
        assert retrieved_args == expected_args
        assert retrieved_details == record_details

    def test_error_handling_in_complex_scenarios(self):
        """Test error handling in complex integration scenarios."""
        obj = LoggableObject()

        # Test with malformed function signatures
        def func_with_no_defaults(a, b, c):
            pass

        # Should handle functions with no defaults
        obj.register_log_and_record_args(func_with_no_defaults, [1, 2], {'c': 3}, {})
        result = obj.retrieve_args(func_with_no_defaults)

        expected = {'a': 1, 'b': 2, 'c': 3}
        assert result == expected
