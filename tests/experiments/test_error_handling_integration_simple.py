"""
Phase 3, Task 3.1: Simple Error Handling Integration Tests

This test module validates the key error handling functionality:
1. Plot error masking is fixed (Phase 1)
2. Environment-specific behavior works (Phase 2) 
3. Data saving happens via chronicle
4. End-to-end error handling workflow

These tests focus on the most critical integration points.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock, call
import sys

# Clean up any mocked plotly modules before importing
for module in list(sys.modules.keys()):
    if module.startswith('plotly'):
        del sys.modules[module]

from leeq.experiments.builtin.basic.calibrations.qubit_spectroscopy import QubitSpectroscopyFrequency
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.chronicle.decorators import log_and_record
from leeq.chronicle.core import LoggableObject


class TestCoreErrorHandlingIntegration:
    """Test the core error handling integration functionality."""

    @pytest.fixture
    def mock_qubit(self):
        """Create a mock qubit for testing."""
        qubit = Mock(spec=TransmonElement)
        qubit.name = "test_qubit"
        qubit.lpb_collections = {
            'f01': {'freq': 5000.0, 'type': 'SimpleDriveCollection'},
            'f12': {'freq': 4800.0, 'type': 'SimpleDriveCollection'},
            'readout': {'freq': 6000.0, 'type': 'SimpleReadoutCollection'}
        }
        return qubit

    def test_plot_error_masking_fixed(self, mock_qubit):
        """Test that Phase 1 fix prevents plot errors from masking the original error."""
        # Import first, then patch the imported module
        from leeq.experiments import experiments
        with patch.object(experiments, 'logger') as mock_logger:
            
            # Create a mock experiment instance with retrieve_args that fails
            mock_exp = Mock()
            mock_exp.retrieve_args.side_effect = ValueError("function not registered")
            mock_exp.live_plots.return_value = Mock()
            
            # Import and test the actual experiments module
            from leeq.experiments.experiments import ExperimentManager
            experiments = ExperimentManager()
            experiments._active_experiment_instance = mock_exp
            
            # Mock the setup dependencies 
            mock_setup = Mock()
            mock_setup.get_live_status.return_value = {
                'engine_status': {'step_no': [1, 1]}  # Non-zero to proceed
            }
            experiments.get_default_setup = Mock(return_value=mock_setup)
            
            # Call get_live_plots which should now properly log the error
            fig = experiments.get_live_plots()
            
            # Verify the error is logged with context (Phase 1 fix)
            warning_calls = mock_logger.warning.call_args_list
            assert len(warning_calls) >= 2, "Should have multiple warning calls"
            
            # Check that the first call contains the error message
            first_call_args = str(warning_calls[0])
            assert "Experiment not registered" in first_call_args
            assert "function not registered" in first_call_args
            
            # Check that the second call contains helpful context
            second_call_args = str(warning_calls[1])
            assert "initialization error" in second_call_args.lower()

    def test_environment_behavior_simulation(self):
        """Test environment-specific behavior using proper LoggableObject."""
        
        # Test both environments
        test_cases = [
            (True, "Jupyter mode should log and continue"),
            (False, "CLI mode should re-raise after data saving")
        ]
        
        for jupyter_mode, description in test_cases:
            with patch('leeq.chronicle.decorators.is_running_in_jupyter', return_value=jupyter_mode), \
                 patch('leeq.chronicle.decorators.Chronicle') as mock_chronicle_cls, \
                 patch('leeq.chronicle.decorators.logger') as mock_logger:
                
                # Mock chronicle as a proper context manager
                mock_record = Mock()
                mock_chronicle_instance = Mock()
                mock_chronicle_instance.new_record.return_value = MagicMock()
                mock_chronicle_instance.new_record.return_value.__enter__ = Mock(return_value=mock_record)
                mock_chronicle_instance.new_record.return_value.__exit__ = Mock(return_value=None)
                mock_chronicle_cls.return_value = mock_chronicle_instance
                
                # Create a test class that inherits from LoggableObject
                class TestLoggableClass(LoggableObject):
                    @log_and_record
                    def failing_method(self):
                        raise ValueError("Test environment error")
                
                test_obj = TestLoggableClass()
                
                if jupyter_mode:
                    # In Jupyter mode, should not raise exception
                    test_obj.failing_method()  # Should complete without exception
                    
                    # Verify error was logged with Jupyter message
                    error_calls = [str(call) for call in mock_logger.error.call_args_list]
                    assert any("Test environment error" in call for call in error_calls), f"{description}: Should log error"
                    assert any("Continuing in Jupyter mode" in call for call in error_calls), f"{description}: Should mention Jupyter mode"
                    
                else:
                    # In CLI mode, should re-raise exception
                    with pytest.raises(ValueError, match="Test environment error"):
                        test_obj.failing_method()
                        
                # Verify chronicle was called in both modes
                mock_chronicle_cls.assert_called(), f"{description}: Chronicle should be instantiated"

    def test_data_preservation_priority(self):
        """Test that data is saved before error handling occurs."""
        
        execution_log = []
        
        with patch('leeq.chronicle.decorators.Chronicle') as mock_chronicle_cls, \
             patch('leeq.chronicle.decorators.is_running_in_jupyter', return_value=False):
            
            # Mock chronicle to track when data saving occurs
            mock_record = Mock()
            mock_chronicle_instance = Mock()
            
            def track_enter(*args, **kwargs):
                execution_log.append("data_save_start")
                return mock_record
            
            def track_exit(*args, **kwargs):
                execution_log.append("data_save_end")
                return None
                
            mock_chronicle_instance.new_record.return_value = MagicMock()
            mock_chronicle_instance.new_record.return_value.__enter__ = track_enter
            mock_chronicle_instance.new_record.return_value.__exit__ = track_exit
            mock_chronicle_cls.return_value = mock_chronicle_instance
            
            # Create test object
            class TestDataSaving(LoggableObject):
                @log_and_record
                def test_method(self):
                    execution_log.append("function_execution")
                    raise RuntimeError("Data saving test error")
            
            test_obj = TestDataSaving()
            
            # Execute and catch the re-raised error
            try:
                test_obj.test_method()
            except RuntimeError:
                execution_log.append("error_reraise")
            
            # Verify execution order: data saving should happen around function execution
            assert "function_execution" in execution_log, "Function should execute"
            assert "data_save_start" in execution_log, "Data saving should start" 
            assert "data_save_end" in execution_log, "Data saving should complete"
            
            # Data saving should wrap around function execution
            func_idx = execution_log.index("function_execution")
            start_idx = execution_log.index("data_save_start")
            end_idx = execution_log.index("data_save_end")
            
            assert start_idx < func_idx, "Data saving should start before function execution"
            assert func_idx < end_idx, "Data saving should complete after function execution"

    def test_jupyter_environment_detection(self):
        """Test that we can properly detect and mock different environments."""
        from leeq.utils.utils import is_running_in_jupyter
        
        # Test Jupyter environment detection
        with patch('leeq.utils.utils.sys') as mock_sys:
            mock_sys.argv = ['python', 'kernel-12345.json']
            result = is_running_in_jupyter()
            assert result is True, "Should detect Jupyter environment"
        
        # Test CLI environment detection  
        with patch('leeq.utils.utils.sys') as mock_sys:
            mock_sys.argv = ['python', 'script.py']
            result = is_running_in_jupyter()
            assert result is False, "Should detect CLI environment"

    def test_error_types_handling(self):
        """Test that different error types are handled consistently."""
        
        error_types = [
            RuntimeError("Runtime error test"),
            ValueError("Value error test"),
            KeyError("Key error test"),
            AttributeError("Attribute error test")
        ]
        
        for error in error_types:
            with patch('leeq.chronicle.decorators.Chronicle') as mock_chronicle_cls, \
                 patch('leeq.chronicle.decorators.is_running_in_jupyter', return_value=True):
                
                # Mock chronicle properly
                mock_chronicle_instance = Mock()
                mock_chronicle_instance.new_record.return_value = MagicMock()
                mock_chronicle_cls.return_value = mock_chronicle_instance
                
                class TestErrorTypes(LoggableObject):
                    def __init__(self, error_to_raise):
                        super().__init__()
                        self.error_to_raise = error_to_raise
                        
                    @log_and_record
                    def failing_method(self):
                        raise self.error_to_raise
                
                test_obj = TestErrorTypes(error)
                
                # Should not raise in Jupyter mode
                test_obj.failing_method()
                
                # Verify chronicle was called
                mock_chronicle_cls.assert_called(), f"Chronicle should be called for {type(error).__name__}"


class TestRegressionSafety:
    """Test that existing functionality is not broken."""
    
    def test_valid_experiment_still_works(self):
        """Test that valid experiments still work normally."""
        
        # This is a minimal test to ensure basic functionality isn't broken
        # We'll just verify we can import and instantiate without errors
        from leeq.experiments.experiments import ExperimentManager
        
        # Should be able to create experiment manager without issues
        manager = ExperimentManager()
        assert manager is not None
        
        # Basic methods should exist
        assert hasattr(manager, 'get_live_plots')
        assert callable(manager.get_live_plots)

    def test_environment_detection_robustness(self):
        """Test that environment detection handles edge cases."""
        from leeq.utils.utils import is_running_in_jupyter
        
        edge_cases = [
            [],  # Empty argv
            ['python'],  # Minimal argv
            ['jupyter', 'notebook'],  # Jupyter but no .json
            ['python', 'script.py', 'extra.json'],  # .json but not kernel
        ]
        
        for argv in edge_cases:
            with patch('leeq.utils.utils.sys') as mock_sys:
                mock_sys.argv = argv
                # Should not crash
                result = is_running_in_jupyter()
                assert isinstance(result, bool), f"Should return boolean for argv: {argv}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])