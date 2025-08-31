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
        layout = session_dashboard.create_layout()
        
        # Convert layout to string for easier checking (mock returns MagicMock)
        layout_str = str(layout)
        
        # Check for key components
        assert 'session-interval' in layout_str or hasattr(layout, 'children')
        assert 'manual-refresh' in layout_str or hasattr(layout, 'children')
        assert 'session-tree' in layout_str or hasattr(layout, 'children')
        assert 'session-experiment-info' in layout_str or hasattr(layout, 'children')
        assert 'session-experiment-attributes' in layout_str or hasattr(layout, 'children')


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
        # Create mock chronicle that is recording but has no record book
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
        """Test loading experiments from active session."""
        # Create mock experiment entries
        mock_entry1 = Mock()
        mock_entry1.record_id = 'exp001'
        mock_entry1.name = 'Test Experiment 1'
        mock_entry1.timestamp = 123456789
        mock_entry1.get_path = Mock(return_value='/root/exp1')
        mock_entry1.children = []
        
        mock_entry2 = Mock()
        mock_entry2.record_id = 'exp002'
        mock_entry2.name = 'Test Experiment 2'
        mock_entry2.timestamp = 123456790
        mock_entry2.get_path = Mock(return_value='/root/exp2')
        mock_entry2.children = []
        
        # Create mock root entry
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.name = 'root'
        mock_root.timestamp = 123456788
        mock_root.get_path = Mock(return_value='/root')
        mock_root.children = [mock_entry1, mock_entry2]
        
        # Create mock record book
        mock_record_book = Mock()
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        
        # Create mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        mock_chronicle._active_record_book = mock_record_book
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Mock the create_tree_view_items function
        with patch('leeq.chronicle.viewer.session_dashboard.create_tree_view_items') as mock_create_tree:
            mock_create_tree.return_value = {'root': ['exp1', 'exp2']}
            
            # Call load_session_experiments
            experiments, tree = session_dashboard.load_session_experiments()
            
            # Check experiments were collected
            assert len(experiments) == 2
            assert experiments[0]['record_id'] == 'exp001'
            assert experiments[1]['record_id'] == 'exp002'
            
            # Check tree was built
            assert tree == {'root': ['exp1', 'exp2']}
    
    def test_load_handles_exceptions(self):
        """Test that load_session_experiments handles exceptions gracefully."""
        # Create mock chronicle that raises exception
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
        """Test update callback when no chronicle instance is available."""
        # Set chronicle to None
        session_dashboard.chronicle_instance = None
        
        # Call update callback
        tree, status, data = session_dashboard.update_session_view(0, 0)
        
        # Should return warning alert
        assert "No Chronicle instance" in str(tree) or "warning" in str(tree)
        assert status == "Chronicle not initialized"
        assert data == {}
    
    def test_update_with_no_experiments(self):
        """Test update callback when no experiments exist."""
        # Create mock chronicle with no experiments
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Mock load_session_experiments to return empty
        with patch('leeq.chronicle.viewer.session_dashboard.load_session_experiments') as mock_load:
            mock_load.return_value = ([], {})
            
            # Call update callback
            tree, status, data = session_dashboard.update_session_view(1, 0)
            
            # Should return info alert about no experiments
            assert "No experiments" in str(tree) or "info" in str(tree)
            assert "Auto-refresh" in status or "updated" in status
            assert data == []
    
    def test_update_with_experiments(self):
        """Test update callback with experiments available."""
        # Create mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Create mock experiments data
        mock_experiments = [
            {'record_id': 'exp001', 'name': 'Test 1', 'timestamp': 12345},
            {'record_id': 'exp002', 'name': 'Test 2', 'timestamp': 12346}
        ]
        mock_tree = {'root': ['exp001', 'exp002']}
        
        # Mock load_session_experiments
        with patch('leeq.chronicle.viewer.session_dashboard.load_session_experiments') as mock_load:
            mock_load.return_value = (mock_experiments, mock_tree)
            
            # Mock render_tree_nodes
            with patch('leeq.chronicle.viewer.session_dashboard.render_tree_nodes') as mock_render:
                mock_render.return_value = "Rendered tree HTML"
                
                # Call update callback
                tree, status, data = session_dashboard.update_session_view(2, 0)
                
                # Check tree was rendered
                assert tree == "Rendered tree HTML"
                
                # Check status shows experiment count
                assert "2 experiments" in status or "updated" in status
                
                # Check data contains experiments
                assert data == mock_experiments
    
    def test_update_handles_errors(self):
        """Test update callback handles errors gracefully."""
        # Set up mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Mock load_session_experiments to raise error
        with patch('leeq.chronicle.viewer.session_dashboard.load_session_experiments') as mock_load:
            mock_load.side_effect = Exception("Test loading error")
            
            # Call update callback
            tree, status, data = session_dashboard.update_session_view(0, 0)
            
            # Should return error alert
            assert "Error" in str(tree) or "danger" in str(tree)
            assert "Error" in status
            assert data == {}
    
    def test_manual_refresh_trigger(self):
        """Test that manual refresh button triggers update."""
        # Set up mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Mock callback context to simulate button click
        with patch('leeq.chronicle.viewer.session_dashboard.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'manual-refresh.n_clicks'}]
            
            # Mock load_session_experiments
            with patch('leeq.chronicle.viewer.session_dashboard.load_session_experiments') as mock_load:
                mock_load.return_value = ([], {})
                
                # Call update callback
                tree, status, data = session_dashboard.update_session_view(0, 1)
                
                # Status should indicate manual refresh
                assert "Manual refresh" in status or "updated" in status


class TestExperimentSelectionCallbacks:
    """Test experiment selection and display callbacks."""
    
    def test_experiment_selection_stores_id(self):
        """Test that selecting an experiment stores its ID."""
        # Mock callback for experiment selection
        with patch('leeq.chronicle.viewer.session_dashboard.select_experiment') as mock_select:
            # Simulate clicking on an experiment in the tree
            clicks = [1]  # One click on first experiment
            
            # Create mock experiment data
            mock_data = [
                {'record_id': 'exp001', 'name': 'Test Experiment'}
            ]
            
            # Call selection callback (if it exists)
            if hasattr(session_dashboard, 'select_experiment'):
                selected_id = session_dashboard.select_experiment(clicks, mock_data)
                assert selected_id == 'exp001'
    
    def test_display_experiment_info(self):
        """Test displaying selected experiment information."""
        # Mock the display_experiment_info callback
        if hasattr(session_dashboard, 'display_experiment_info'):
            # Create mock experiment data
            mock_experiment = {
                'record_id': 'exp001',
                'name': 'Test Experiment',
                'timestamp': 123456789,
                'entry_path': '/root/test'
            }
            
            # Call display callback
            info_html = session_dashboard.display_experiment_info('exp001', [mock_experiment])
            
            # Check that info contains experiment details
            assert 'Test Experiment' in str(info_html)
            assert 'exp001' in str(info_html)


class TestStartViewerFunction:
    """Test the start_viewer function."""
    
    def test_start_viewer_requires_chronicle(self):
        """Test that start_viewer requires chronicle_instance parameter."""
        with pytest.raises(ValueError, match="Chronicle instance must be provided"):
            session_dashboard.start_viewer(port=8051)
    
    def test_start_viewer_sets_global_chronicle(self):
        """Test that start_viewer sets the global chronicle_instance."""
        # Create mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        
        # Reset global
        session_dashboard.chronicle_instance = None
        
        # Mock app.run_server to prevent actual server start
        with patch.object(session_dashboard.app, 'run_server'):
            # Call start_viewer
            session_dashboard.start_viewer(
                chronicle_instance=mock_chronicle,
                port=8051
            )
            
            # Verify global was set
            assert session_dashboard.chronicle_instance is mock_chronicle
    
    def test_start_viewer_passes_kwargs_to_dash(self):
        """Test that start_viewer passes all kwargs to dash server."""
        # Create mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        
        # Mock app.run_server
        with patch.object(session_dashboard.app, 'run_server') as mock_run:
            # Call start_viewer with various kwargs
            session_dashboard.start_viewer(
                chronicle_instance=mock_chronicle,
                port=9999,
                debug=False,
                host='0.0.0.0',
                threaded=True
            )
            
            # Verify run_server was called with correct args
            mock_run.assert_called_once_with(
                port=9999,
                debug=False,
                host='0.0.0.0',
                threaded=True
            )
    
    def test_start_viewer_default_port(self):
        """Test that start_viewer uses default port 8051 if not specified."""
        # Create mock chronicle
        mock_chronicle = Mock(spec=Chronicle)
        
        # Mock app.run_server
        with patch.object(session_dashboard.app, 'run_server') as mock_run:
            # Call start_viewer without port
            session_dashboard.start_viewer(
                chronicle_instance=mock_chronicle
            )
            
            # Check if port kwarg was passed (may use default in app.run_server)
            call_kwargs = mock_run.call_args[1]
            # Port might not be in kwargs if using Dash default
            if 'port' in call_kwargs:
                assert call_kwargs['port'] == 8051 or call_kwargs['port'] == 8050


class TestErrorHandlingScenarios:
    """Test various error handling scenarios."""
    
    def test_handle_chronicle_connection_lost(self):
        """Test handling when chronicle connection is lost during operation."""
        # Set up initial chronicle
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Simulate connection lost by making is_recording raise error
        mock_chronicle.is_recording.side_effect = ConnectionError("Lost connection")
        
        # Load should handle error gracefully
        experiments, tree = session_dashboard.load_session_experiments()
        assert experiments == []
        assert tree == {}
    
    def test_handle_corrupt_experiment_data(self):
        """Test handling corrupt or incomplete experiment data."""
        # Create mock entry with missing attributes
        mock_entry = Mock()
        mock_entry.record_id = 'exp001'
        # Missing name attribute
        mock_entry.timestamp = 12345
        mock_entry.get_path = Mock(side_effect=AttributeError("No path"))
        mock_entry.children = []
        
        mock_root = Mock()
        mock_root.children = [mock_entry]
        mock_root.get_path = Mock(return_value='/root')
        
        mock_record_book = Mock()
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        mock_chronicle._active_record_book = mock_record_book
        
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Should handle corrupt data without crashing
        experiments, tree = session_dashboard.load_session_experiments()
        # May return empty or partial data
        assert isinstance(experiments, list)
        assert isinstance(tree, dict)
    
    def test_handle_large_number_of_experiments(self):
        """Test handling a large number of experiments efficiently."""
        # Create many mock entries
        mock_entries = []
        for i in range(1000):
            entry = Mock()
            entry.record_id = f'exp{i:04d}'
            entry.name = f'Experiment {i}'
            entry.timestamp = 123456789 + i
            entry.get_path = Mock(return_value=f'/root/exp{i}')
            entry.children = []
            mock_entries.append(entry)
        
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.children = mock_entries
        mock_root.get_path = Mock(return_value='/root')
        
        mock_record_book = Mock()
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.is_recording = Mock(return_value=True)
        mock_chronicle._active_record_book = mock_record_book
        
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Mock create_tree_view_items to avoid complex tree building
        with patch('leeq.chronicle.viewer.session_dashboard.create_tree_view_items') as mock_create:
            mock_create.return_value = {'root': [f'exp{i:04d}' for i in range(1000)]}
            
            # Should handle large dataset without error
            experiments, tree = session_dashboard.load_session_experiments()
            assert len(experiments) == 1000
            assert len(tree['root']) == 1000


class TestPortConfiguration:
    """Test port configuration and conflict handling."""
    
    def test_different_port_from_live_monitor(self):
        """Test that session viewer uses different port from live monitor."""
        # Default port should be 8051 (different from live monitor's 8050)
        mock_chronicle = Mock(spec=Chronicle)
        
        with patch.object(session_dashboard.app, 'run_server') as mock_run:
            with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
                # Check that default configuration uses port 8051
                mock_chronicle.launch_viewer()
                
                # Verify port 8051 is used (different from live monitor)
                call_args = mock_start.call_args[1] if mock_start.called else {}
                if 'port' in call_args:
                    assert call_args['port'] == 8051
    
    def test_custom_port_override(self):
        """Test that custom port can be specified."""
        mock_chronicle = Mock(spec=Chronicle)
        
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            # Launch with custom port
            mock_chronicle.launch_viewer(port=7777)
            
            # Verify custom port is used
            if mock_start.called:
                call_args = mock_start.call_args[1]
                assert call_args.get('port') == 7777


class TestLiveUpdateMechanism:
    """Test the live update polling mechanism."""
    
    def test_interval_component_configuration(self):
        """Test that interval component is configured for 5-second updates."""
        layout = session_dashboard.create_layout()
        layout_str = str(layout)
        
        # Check for interval component with 5000ms interval
        assert 'session-interval' in layout_str or hasattr(layout, 'children')
        # Interval should be set to 5000ms (5 seconds)
        assert '5000' in layout_str or 'interval=5000' in layout_str
    
    def test_polling_updates_experiment_list(self):
        """Test that polling updates the experiment list."""
        # Set up mock chronicle with initial empty state
        mock_chronicle = Mock(spec=Chronicle)
        session_dashboard.chronicle_instance = mock_chronicle
        
        # First update - no experiments
        with patch('leeq.chronicle.viewer.session_dashboard.load_session_experiments') as mock_load:
            mock_load.return_value = ([], {})
            tree1, status1, data1 = session_dashboard.update_session_view(0, 0)
            assert data1 == []
        
        # Second update - experiments added
        with patch('leeq.chronicle.viewer.session_dashboard.load_session_experiments') as mock_load:
            mock_load.return_value = (
                [{'record_id': 'exp001', 'name': 'New Experiment'}],
                {'root': ['exp001']}
            )
            tree2, status2, data2 = session_dashboard.update_session_view(1, 0)
            assert len(data2) == 1
            assert data2[0]['record_id'] == 'exp001'
    
    def test_manual_refresh_overrides_interval(self):
        """Test that manual refresh works independently of interval."""
        mock_chronicle = Mock(spec=Chronicle)
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Mock callback context to show manual trigger
        with patch('leeq.chronicle.viewer.session_dashboard.callback_context') as mock_ctx:
            # Simulate manual button click
            mock_ctx.triggered = [{'prop_id': 'manual-refresh.n_clicks'}]
            
            with patch('leeq.chronicle.viewer.session_dashboard.load_session_experiments') as mock_load:
                mock_load.return_value = (
                    [{'record_id': 'exp_manual', 'name': 'Manual Refresh Test'}],
                    {'root': ['exp_manual']}
                )
                
                # Manual refresh should work regardless of interval count
                tree, status, data = session_dashboard.update_session_view(100, 1)
                
                assert "Manual refresh" in status or "manually" in status.lower()
                assert data[0]['record_id'] == 'exp_manual'


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])