"""
Unit tests for the simplified EPII parameter management.
These tests will be used to validate the new implementation in Phase 2.
"""

import pytest
import numpy as np
from unittest.mock import Mock

# Import the new simplified implementation
from leeq.epii.parameters import ParameterManager
from tests.epii.test_fixtures_new import (
    get_test_setup,
    get_test_parameters_flat,
    get_test_types,
    get_empty_setup
)


class TestSimplifiedParameterManager:
    """Test suite for simplified ParameterManager"""

    def test_get_all_parameters_flattened(self):
        """Test that get_all_parameters returns flattened dictionary."""
        # This test will be enabled in Phase 2
        # Test enabled for Phase 2

        setup = get_test_setup()
        pm = ParameterManager(setup)

        all_params = pm.get_all_parameters()

        # Check we get all expected parameters
        assert "status.shot_number" in all_params
        assert "q0.f01" in all_params
        assert "q1.f01" in all_params
        assert "res0.frequency" in all_params

        # Check nested parameters are flattened
        assert "q0.nested.level1.level2" in all_params
        assert all_params["q0.nested.level1.level2"] == "deep_value"

        # Verify count
        expected = get_test_parameters_flat()
        assert len(all_params) >= len(expected)

    def test_get_parameter_direct_access(self):
        """Test get_parameter with dot notation."""
        # Test enabled for Phase 2

        setup = get_test_setup()
        pm = ParameterManager(setup)

        # Status parameter
        assert pm.get_parameter("status.shot_number") == 2000

        # Element parameter
        assert pm.get_parameter("q0.f01") == 5.0e9

        # Nested parameter
        assert pm.get_parameter("q0.nested.level1.level2") == "deep_value"

        # Non-existent returns None
        assert pm.get_parameter("nonexistent.param") is None

    def test_set_parameter_updates(self):
        """Test set_parameter updates values correctly."""
        # Test enabled for Phase 2

        setup = get_test_setup()
        pm = ParameterManager(setup)

        # Set status parameter
        pm.set_parameter("status.shot_number", 5000)
        assert pm.get_parameter("status.shot_number") == 5000

        # Set element parameter
        pm.set_parameter("q0.f01", 5.5e9)
        assert pm.get_parameter("q0.f01") == 5.5e9

        # Set new parameter (goes to cache)
        pm.set_parameter("new.parameter", "test_value")
        assert pm.get_parameter("new.parameter") == "test_value"

    def test_serialize_value_types(self):
        """Test serialize_value handles all supported types."""
        # Test enabled for Phase 2

        pm = ParameterManager()
        types = get_test_types()

        # Test each type
        assert pm.serialize_value(types["int_value"]) == "42"
        assert pm.serialize_value(types["float_value"]) == "3.14159"
        assert pm.serialize_value(types["bool_true"]) == "true"
        assert pm.serialize_value(types["bool_false"]) == "false"
        assert pm.serialize_value(types["string_value"]) == "test_string"
        assert pm.serialize_value(types["none_value"]) == "null"

        # Numpy array becomes list
        serialized = pm.serialize_value(types["numpy_array"])
        assert serialized == [1.0, 2.0, 3.0]

        # List and dict pass through
        assert pm.serialize_value(types["list_value"]) == [1, 2, 3]
        assert pm.serialize_value(types["dict_value"]) == {"key": "value"}

    def test_no_validation_logic(self):
        """Test that no validation is performed - any value is accepted."""
        # Test enabled for Phase 2

        setup = get_test_setup()
        pm = ParameterManager(setup)

        # Should accept any value without validation
        pm.set_parameter("status.shot_number", "not_a_number")  # Should work
        assert pm.get_parameter("status.shot_number") == "not_a_number"

        pm.set_parameter("q0.f01", -999)  # Negative frequency - should work
        assert pm.get_parameter("q0.f01") == -999

        pm.set_parameter("any.random.path", {"complex": "object"})  # Should work
        assert pm.get_parameter("any.random.path") == {"complex": "object"}

    def test_cache_fallback(self):
        """Test that unknown parameters are stored in cache."""
        # Test enabled for Phase 2

        setup = get_test_setup()
        pm = ParameterManager(setup)

        # Set parameter that doesn't exist in setup
        pm.set_parameter("cached.param", 123)

        # Should be retrievable
        assert pm.get_parameter("cached.param") == 123

        # Should appear in get_all_parameters
        all_params = pm.get_all_parameters()
        assert "cached.param" in all_params
        assert all_params["cached.param"] == 123

    def test_empty_setup(self):
        """Test with minimal/empty setup."""
        # Test enabled for Phase 2

        setup = get_empty_setup()
        pm = ParameterManager(setup)

        all_params = pm.get_all_parameters()

        # Should at least have status parameter
        assert "status.shot_number" in all_params
        assert all_params["status.shot_number"] == 1000

        # No elements
        assert not any(k.startswith("q0.") for k in all_params)

    def test_line_count_requirement(self):
        """Verify the implementation is under 150 lines."""
        # Test enabled for Phase 2
        import os

        # Read the source file directly
        file_path = os.path.join(os.path.dirname(__file__), '../../leeq/epii/parameters.py')
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Total line count
        total_lines = len(lines)

        assert total_lines < 150, f"Implementation has {total_lines} lines, should be < 150"


class TestCompatibility:
    """Test compatibility with existing EPII service."""

    def test_service_integration(self):
        """Test that the new manager works with EPII service."""
        pytest.skip("Waiting for Phase 3 implementation")

        # This will be implemented in Phase 3
        pass

    def test_experiments_still_work(self):
        """Test that existing experiments continue to function."""
        pytest.skip("Waiting for Phase 3 implementation")

        # This will be implemented in Phase 3
        pass
