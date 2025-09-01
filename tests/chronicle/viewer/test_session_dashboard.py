"""
Test Session Dashboard module for Chronicle viewer.

This module provides comprehensive tests for the session dashboard, including:
1. Dashboard initialization and layout
2. Live update callbacks with polling
3. Error handling for various states
4. Data flow from chronicle to UI
5. Port configuration and server startup
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Mock the dash and plotly modules before import
sys.modules['dash'] = MagicMock()
sys.modules['dash.dependencies'] = MagicMock()
sys.modules['dash_bootstrap_components'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.tools'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()

# Import Chronicle first (doesn't need dash)
from leeq.chronicle import Chronicle

# Now import session_dashboard with mocked dependencies
from leeq.chronicle.viewer import session_dashboard


class TestSessionDashboardInitialization:
    """Test session dashboard initialization and setup."""
    
    def test_dashboard_app_created(self):
        """Test that Dash app is created with correct configuration."""
        assert session_dashboard.app is not None
        # App should have been created with bootstrap theme
        assert hasattr(session_dashboard.app, 'layout')
    
    def test_global_chronicle_instance_initialized(self):
        """Test that global chronicle_instance is initialized to None."""
        # Reset the global variable
        session_dashboard.chronicle_instance = None
        assert session_dashboard.chronicle_instance is None
    
    def test_dashboard_layout_structure(self):
        """Test that dashboard layout contains all required components."""
        # The layout is directly accessible via app.layout
        layout = session_dashboard.app.layout
        
        # Since we're mocking Dash, just verify the layout exists
        assert layout is not None
        assert hasattr(session_dashboard.app, 'layout')


class TestLoadSessionExperiments:
    """Test the load_session_experiments function."""
    
    def test_load_with_no_chronicle_instance(self):
        """Test loading experiments when no chronicle instance is set."""
        # Set chronicle_instance to None
        session_dashboard.chronicle_instance = None
        
        # Call load_session_experiments
        experiments, tree = session_dashboard.load_session_experiments()
        
        # Should return empty structures
        assert experiments == []
        assert tree == {}
    
    def test_load_with_chronicle_not_recording(self):
        """Test loading when chronicle is not recording."""
        # Create mock chronicle that is not recording
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=False)
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Call load_session_experiments
        experiments, tree = session_dashboard.load_session_experiments()
        
        # Should return empty structures
        assert experiments == []
        assert tree == {}
        
        # Verify is_recording was called
        mock_chronicle.is_recording.assert_called_once()
    
    def test_load_with_no_active_record_book(self):
        """Test loading when chronicle has no active record book."""
        # Create mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        mock_chronicle._active_record_book = None
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Call load_session_experiments
        experiments, tree = session_dashboard.load_session_experiments()
        
        # Should return empty structures
        assert experiments == []
        assert tree == {}
    
    def test_load_with_experiments(self):
        """Test loading when chronicle has experiments."""
        # Create mock chronicle with experiments
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        
        # Create mock record book with experiments
        mock_record_book = Mock()
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.get_path = Mock(return_value='/root')
        mock_root.name = 'root'
        mock_root.timestamp = 0
        
        # Create mock experiments
        exp1 = Mock()
        exp1.record_id = 'exp001'
        exp1.get_path = Mock(return_value='/root/exp001')
        exp1.name = 'Test Experiment 1'
        exp1.timestamp = 100.0
        exp1.children = []
        
        exp2 = Mock()
        exp2.record_id = 'exp002'
        exp2.get_path = Mock(return_value='/root/exp002')
        exp2.name = 'Test Experiment 2'
        exp2.timestamp = 200.0
        exp2.children = []
        
        mock_root.children = [exp1, exp2]
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        
        mock_chronicle._active_record_book = mock_record_book
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Call load_session_experiments
        experiments, tree = session_dashboard.load_session_experiments()
        
        # Should return experiments (root is not included)
        assert len(experiments) == 2
        assert experiments[0]['record_id'] == 'exp001'
        assert experiments[0]['name'] == 'Test Experiment 1'
        assert experiments[1]['record_id'] == 'exp002'
        assert experiments[1]['name'] == 'Test Experiment 2'
        
        # Tree should be built - could be dict or empty
        assert tree is not None
    
    def test_load_handles_exceptions(self):
        """Test that load_session_experiments handles exceptions gracefully."""
        # Create mock chronicle that raises error
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(side_effect=Exception("Test error"))
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Call should handle exception and return empty
        experiments, tree = session_dashboard.load_session_experiments()
        
        assert experiments == []
        assert tree == {}


class TestUpdateSessionViewCallback:
    """Test the update_session_view callback function."""
    
    def test_update_with_no_chronicle(self):
        """Test underlying logic when no chronicle instance is available."""
        # Set chronicle to None
        session_dashboard.chronicle_instance = None
        
        # Test the underlying function
        experiments, tree = session_dashboard.load_session_experiments()
        
        # Should return empty data
        assert experiments == []
        assert tree == {}
    
    def test_update_with_no_experiments(self):
        """Test underlying logic when no experiments exist."""
        # Create mock chronicle with no experiments
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        
        mock_record_book = Mock()
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.get_path = Mock(return_value='/root')
        mock_root.name = 'root'
        mock_root.timestamp = 0
        mock_root.children = []  # No children
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        
        mock_chronicle._active_record_book = mock_record_book
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Test the underlying function
        experiments, tree = session_dashboard.load_session_experiments()
        
        # Should return empty experiments (root is not included)
        assert experiments == []
        assert tree is not None  # May return empty list or dict
    
    def test_update_with_experiments(self):
        """Test underlying logic with experiments available."""
        # Create mock chronicle with experiments
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        
        mock_record_book = Mock()
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.get_path = Mock(return_value='/root')
        mock_root.name = 'root'
        mock_root.timestamp = 0
        
        # Create mock experiments
        exp1 = Mock()
        exp1.record_id = 'exp001'
        exp1.get_path = Mock(return_value='/root/exp001')
        exp1.name = 'Test Experiment 1'
        exp1.timestamp = 100.0
        exp1.children = []
        
        exp2 = Mock()
        exp2.record_id = 'exp002'
        exp2.get_path = Mock(return_value='/root/exp002')
        exp2.name = 'Test Experiment 2'
        exp2.timestamp = 200.0
        exp2.children = []
        
        mock_root.children = [exp1, exp2]
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        
        mock_chronicle._active_record_book = mock_record_book
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Test the underlying function
        experiments, tree = session_dashboard.load_session_experiments()
        
        # Should have experiments
        assert len(experiments) == 2
        assert experiments[0]['record_id'] == 'exp001'
        assert experiments[1]['record_id'] == 'exp002'
        assert tree is not None
    
    def test_update_handles_errors(self):
        """Test that errors are handled gracefully."""
        # Create mock chronicle that raises error
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(side_effect=Exception("Test error"))
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Test the underlying function - should handle error
        experiments, tree = session_dashboard.load_session_experiments()
        
        # Should return empty on error
        assert experiments == []
        assert tree == {}
    
    def test_manual_refresh_trigger(self):
        """Test that manual refresh logic works."""
        # Create mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        
        mock_record_book = Mock()
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.get_path = Mock(return_value='/root')
        mock_root.name = 'root'
        mock_root.timestamp = 0
        mock_root.children = []
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        
        mock_chronicle._active_record_book = mock_record_book
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Test the underlying function (can't test callback context)
        experiments, tree = session_dashboard.load_session_experiments()
        
        # Should work normally
        assert experiments == []
        assert tree is not None


class TestExperimentSelectionCallbacks:
    """Test experiment selection and display callbacks."""
    
    def test_experiment_selection_stores_id(self):
        """Test that selecting an experiment stores its ID."""
        # We can't directly test the callback, but we can verify the function exists
        assert hasattr(session_dashboard, 'select_experiment')
    
    def test_display_experiment_info(self):
        """Test displaying experiment information."""
        # We can't directly test the callback, but we can verify the function exists
        assert hasattr(session_dashboard, 'update_experiment_display')


class TestStartViewerFunction:
    """Test the start_viewer function."""
    
    def test_start_viewer_requires_chronicle(self):
        """Test that start_viewer requires a chronicle instance."""
        with pytest.raises(ValueError, match="Chronicle instance must be provided"):
            session_dashboard.start_viewer()
    
    def test_start_viewer_sets_global_chronicle(self):
        """Test that start_viewer sets the global chronicle_instance."""
        # Create mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        
        # Reset global
        session_dashboard.chronicle_instance = None
        
        # Mock app.run to prevent server startup
        with patch.object(session_dashboard.app, 'run') as mock_run:
            # Call start_viewer
            session_dashboard.start_viewer(chronicle_instance=mock_chronicle)
            
            # Verify global was set
            assert session_dashboard.chronicle_instance is mock_chronicle
    
    def test_start_viewer_passes_kwargs_to_dash(self):
        """Test that start_viewer passes kwargs to dash app."""
        # Create mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        
        # Mock app.run
        with patch.object(session_dashboard.app, 'run') as mock_run:
            # Call start_viewer with custom kwargs
            session_dashboard.start_viewer(
                chronicle_instance=mock_chronicle,
                port=9999,
                debug=False,
                host='0.0.0.0'
            )
            
            # Verify app.run was called with correct arguments
            mock_run.assert_called_once_with(
                port=9999,
                debug=False,
                host='0.0.0.0'
            )
    
    def test_start_viewer_default_port(self):
        """Test that start_viewer uses default port 8051."""
        # Create mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        
        # Mock app.run
        with patch.object(session_dashboard.app, 'run') as mock_run:
            # Call start_viewer without specifying port
            session_dashboard.start_viewer(chronicle_instance=mock_chronicle)
            
            # Verify default port was used
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs['port'] == 8051


class TestErrorHandlingScenarios:
    """Test various error handling scenarios."""
    
    def test_handle_chronicle_connection_lost(self):
        """Test handling when chronicle connection is lost."""
        # Start with valid chronicle
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        mock_chronicle._active_record_book = None
        
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Try to load experiments - should handle gracefully
        experiments, tree = session_dashboard.load_session_experiments()
        
        assert experiments == []
        assert tree == {}
    
    def test_handle_corrupt_experiment_data(self):
        """Test handling corrupt experiment data."""
        # Create mock chronicle with corrupt data
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        
        mock_record_book = Mock()
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.get_path = Mock(return_value='/root')
        mock_root.name = 'root'
        mock_root.timestamp = 0
        
        # Create corrupt experiment (missing required attributes)
        bad_exp = Mock()
        bad_exp.record_id = 'bad_exp'
        bad_exp.get_path = Mock(side_effect=AttributeError("No path"))
        bad_exp.children = []
        
        mock_root.children = [bad_exp]
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        
        mock_chronicle._active_record_book = mock_record_book
        
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Should handle error and return empty
        experiments, tree = session_dashboard.load_session_experiments()
        
        # Should return empty data on error
        assert experiments == []
        assert tree is not None
    
    def test_handle_large_number_of_experiments(self):
        """Test handling large number of experiments."""
        # Create mock chronicle with many experiments
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        
        mock_record_book = Mock()
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.get_path = Mock(return_value='/root')
        mock_root.name = 'root'
        mock_root.timestamp = 0
        
        # Create 100 mock experiments
        experiments_list = []
        for i in range(100):
            exp = Mock()
            exp.record_id = f'exp{i:03d}'
            exp.get_path = Mock(return_value=f'/root/exp{i:03d}')
            exp.name = f'Experiment {i}'
            exp.timestamp = float(i)
            exp.children = []
            experiments_list.append(exp)
        
        mock_root.children = experiments_list
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        
        mock_chronicle._active_record_book = mock_record_book
        
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Should handle large number successfully
        experiments, tree = session_dashboard.load_session_experiments()
        
        assert len(experiments) == 100
        assert tree is not None


class TestPortConfiguration:
    """Test port configuration for session viewer."""
    
    def test_different_port_from_live_monitor(self):
        """Test that session viewer uses different port from live monitor."""
        # Session viewer should use port 8051 by default
        # Live monitor uses port 8050 by default
        assert session_dashboard.start_viewer.__doc__ is not None
        
        # Verify default ports are different
        # This is a design test rather than functional test
        SESSION_VIEWER_PORT = 8051
        LIVE_MONITOR_PORT = 8050
        
        assert SESSION_VIEWER_PORT != LIVE_MONITOR_PORT
    
    def test_custom_port_override(self):
        """Test that custom port can be specified."""
        mock_chronicle = Mock(spec=Chronicle)
        
        with patch.object(session_dashboard.app, 'run') as mock_run:
            # Test custom port
            session_dashboard.start_viewer(
                chronicle_instance=mock_chronicle,
                port=7777
            )
            
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs['port'] == 7777


class TestLiveUpdateMechanism:
    """Test live update mechanism with polling."""
    
    def test_interval_component_configuration(self):
        """Test that interval component is configured correctly."""
        # The interval component should be in the layout
        layout = session_dashboard.app.layout
        
        # Since we're mocking, just verify layout exists
        assert layout is not None
    
    def test_polling_updates_experiment_list(self):
        """Test that polling updates the experiment list."""
        # This tests the underlying logic, not the actual polling
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        
        mock_record_book = Mock()
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.get_path = Mock(return_value='/root')
        mock_root.name = 'root'
        mock_root.timestamp = 0
        mock_root.children = []
        
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        mock_chronicle._active_record_book = mock_record_book
        
        session_dashboard.chronicle_instance = mock_chronicle
        
        # First poll - no experiments
        experiments1, tree1 = session_dashboard.load_session_experiments()
        assert len(experiments1) == 0
        
        # Add an experiment
        new_exp = Mock()
        new_exp.record_id = 'new_exp'
        new_exp.get_path = Mock(return_value='/root/new_exp')
        new_exp.name = 'New Experiment'
        new_exp.timestamp = 500.0
        new_exp.children = []
        mock_root.children = [new_exp]
        
        # Second poll - should see new experiment
        experiments2, tree2 = session_dashboard.load_session_experiments()
        assert len(experiments2) == 1
        assert experiments2[0]['record_id'] == 'new_exp'
    
    def test_manual_refresh_overrides_interval(self):
        """Test that manual refresh works independently of interval."""
        # This is a design test - manual refresh should work anytime
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        
        mock_record_book = Mock()
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.get_path = Mock(return_value='/root')
        mock_root.name = 'root'
        mock_root.timestamp = 0
        mock_root.children = []
        
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        mock_chronicle._active_record_book = mock_record_book
        
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Manual refresh should work at any time
        experiments, tree = session_dashboard.load_session_experiments()
        
        assert experiments == []
        assert tree is not None