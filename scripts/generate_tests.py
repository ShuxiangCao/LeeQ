#!/usr/bin/env python3
"""Generate test templates for untested modules."""

import os
from pathlib import Path
import ast


def generate_test_template(module_path: Path) -> str:
    """Generate a test template for a module."""
    module_name = module_path.stem
    module_import = str(module_path).replace('/', '.').replace('.py', '')
    
    template = f'''"""
Tests for {module_import}
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from {module_import} import *


class Test{module_name.title().replace("_", "")}:
    """Test suite for {module_name} module."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data and mocks."""
        return {{
            'test_data': np.array([1, 2, 3]),
            'mock_config': Mock(),
        }}
    
    def test_basic_functionality(self, setup_data):
        """Test core functionality."""
        # TODO: Implement actual test
        assert True
        
    def test_edge_cases(self, setup_data):
        """Test edge cases and error handling."""
        # TODO: Implement edge case tests
        with pytest.raises(ValueError):
            pass  # Replace with actual test
            
    @pytest.mark.parametrize("input_val,expected", [
        (1, 1),
        (2, 4),
        (3, 9),
    ])
    def test_parametrized(self, input_val, expected):
        """Test with multiple input values."""
        # TODO: Implement parametrized test
        result = input_val ** 2
        assert result == expected
'''
    return template


# Priority modules needing tests (excluding simulators and AI)
MODULES_NEEDING_TESTS = [
    "leeq/theory/tomography/state_tomography.py",
    "leeq/theory/fits/multilevel_decay.py", 
    "leeq/theory/fits/fit_exp.py",
    "leeq/compiler/lbnl_qubic/circuit_list_compiler.py",  # Fixed: was compiler.py
    "leeq/core/elements/elements.py",  # Fixed: was qubit.py
    "leeq/core/primitives/logical_primitives.py",
]

if __name__ == "__main__":
    for module_path in MODULES_NEEDING_TESTS:
        path = Path(module_path)
        test_path = Path("tests") / path.relative_to("leeq").with_name(f"test_{path.name}")
        
        if not test_path.exists():
            test_path.parent.mkdir(parents=True, exist_ok=True)
            test_content = generate_test_template(path)
            test_path.write_text(test_content)
            print(f"Created test template: {test_path}")
        else:
            print(f"Test already exists: {test_path}")