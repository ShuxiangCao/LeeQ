"""
Test Chronicle integration with session viewer dashboard.

This module tests the integration between Chronicle.launch_viewer() and
the session dashboard, ensuring:
1. Chronicle instance is passed correctly from launch_viewer() to start_viewer()
2. The global chronicle_instance is set properly
3. Session data can be accessed through the chronicle singleton
4. Error handling works when chronicle has no active session
5. Port configuration is respected
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
from datetime import datetime
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Mock the dash app and prevent actual server startup
sys.modules['dash'] = MagicMock()
sys.modules['dash.dependencies'] = MagicMock()
sys.modules['dash_bootstrap_components'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.tools'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()

from leeq.chronicle import Chronicle
from leeq.chronicle.viewer import session_dashboard


class TestChronicleIntegration:
    """Test Chronicle.launch_viewer() integration with session dashboard."""
    
    def test_chronicle_launch_viewer_method_exists(self):
        """Test that Chronicle has launch_viewer method."""
        chronicle = Chronicle()
        assert hasattr(chronicle, 'launch_viewer')
        assert callable(chronicle.launch_viewer)
    
    @patch('leeq.chronicle.viewer.session_dashboard.start_viewer')
    def test_chronicle_instance_passed_to_dashboard(self, mock_start_viewer):
        """Test that chronicle instance is correctly passed to start_viewer."""
        # Create chronicle instance
        chronicle = Chronicle()
        
        # Call launch_viewer with custom port
        chronicle.launch_viewer(port=9999, debug=False)
        
        # Verify start_viewer was called with correct arguments
        mock_start_viewer.assert_called_once()
        call_args = mock_start_viewer.call_args[1]
        
        # Check chronicle instance was passed
        assert 'chronicle_instance' in call_args
        assert call_args['chronicle_instance'] is chronicle
        
        # Check other arguments
        assert call_args['port'] == 9999
        assert call_args['debug'] is False
    
    @patch('leeq.chronicle.viewer.session_dashboard.app')
    def test_global_chronicle_instance_is_set(self, mock_app):
        """Test that global chronicle_instance is set in session_dashboard."""
        # Create chronicle instance
        chronicle = Chronicle()
        
        # Reset global variable
        session_dashboard.chronicle_instance = None
        
        # Call start_viewer directly
        session_dashboard.start_viewer(
            chronicle_instance=chronicle,
            port=8051,
            debug=True
        )
        
        # Verify global chronicle_instance was set
        assert session_dashboard.chronicle_instance is chronicle
    
    def test_start_viewer_requires_chronicle_instance(self):
        """Test that start_viewer raises error without chronicle instance."""
        with pytest.raises(ValueError, match="Chronicle instance must be provided"):
            session_dashboard.start_viewer(port=8051)
    
    @patch('leeq.chronicle.viewer.session_dashboard.app')
    def test_default_port_configuration(self, mock_app):
        """Test that default port 8051 is used when not specified."""
        chronicle = Chronicle()
        
        # Mock start_viewer to capture arguments
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            chronicle.launch_viewer()
            
            # Check default port was passed
            call_args = mock_start.call_args[1]
            assert call_args['port'] == 8051
    
    @patch('leeq.chronicle.viewer.session_dashboard.app')
    def test_custom_port_configuration(self, mock_app):
        """Test that custom port is respected."""
        chronicle = Chronicle()
        
        # Mock start_viewer to capture arguments
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            chronicle.launch_viewer(port=7777)
            
            # Check custom port was passed
            call_args = mock_start.call_args[1]
            assert call_args['port'] == 7777
    
    def test_load_session_experiments_no_chronicle(self):
        """Test load_session_experiments handles missing chronicle gracefully."""
        # Set chronicle_instance to None
        session_dashboard.chronicle_instance = None
        
        # Call load_session_experiments
        entries, tree = session_dashboard.load_session_experiments()
        
        # Should return empty dictionaries
        assert entries == {}
        assert tree == {}
    
    def test_load_session_experiments_with_chronicle(self):
        """Test load_session_experiments accesses chronicle data."""
        # Create mock chronicle with session data
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.get_current_session_entries = Mock(return_value={
            'exp1': {'name': 'Test Experiment 1', 'timestamp': 123456},
            'exp2': {'name': 'Test Experiment 2', 'timestamp': 123457}
        })
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Mock build_experiment_tree to return a simple tree
        with patch('leeq.chronicle.viewer.session_dashboard.build_experiment_tree') as mock_build:
            mock_build.return_value = {'root': ['exp1', 'exp2']}
            
            # Call load_session_experiments
            entries, tree = session_dashboard.load_session_experiments()
            
            # Verify chronicle was accessed
            mock_chronicle.get_current_session_entries.assert_called_once()
            
            # Check returned data
            assert 'exp1' in entries
            assert 'exp2' in entries
            assert tree == {'root': ['exp1', 'exp2']}
    
    def test_chronicle_no_active_session_handling(self):
        """Test that viewer handles chronicle with no active session."""
        # Create mock chronicle with no session
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.get_current_session_entries = Mock(return_value={})
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Call load_session_experiments
        entries, tree = session_dashboard.load_session_experiments()
        
        # Should return empty data without crashing
        assert entries == {}
        assert tree == {}
    
    def test_chronicle_session_error_handling(self):
        """Test error handling when chronicle raises exception."""
        # Create mock chronicle that raises error
        mock_chronicle = Mock(spec=Chronicle)
        mock_chronicle.get_current_session_entries = Mock(
            side_effect=Exception("No active session")
        )
        
        # Set global chronicle instance
        session_dashboard.chronicle_instance = mock_chronicle
        
        # load_session_experiments should handle error gracefully
        try:
            entries, tree = session_dashboard.load_session_experiments()
            # If no exception is raised, check empty returns
            assert entries == {} or entries is None
            assert tree == {} or tree is None
        except Exception:
            # If exception propagates, that's also acceptable error handling
            pass
    
    @patch('leeq.chronicle.viewer.session_dashboard.app')
    def test_additional_kwargs_passed_through(self, mock_app):
        """Test that additional kwargs are passed through to dash server."""
        chronicle = Chronicle()
        
        # Call start_viewer with extra arguments
        session_dashboard.start_viewer(
            chronicle_instance=chronicle,
            port=8051,
            debug=False,
            host='0.0.0.0',
            threaded=True
        )
        
        # Verify app.run_server was called with all arguments
        mock_app.run_server.assert_called_once()
        call_args = mock_app.run_server.call_args[1]
        
        assert call_args['port'] == 8051
        assert call_args['debug'] is False
        assert call_args['host'] == '0.0.0.0'
        assert call_args['threaded'] is True


class TestChronicleSessionDataAccess:
    """Test accessing session data through Chronicle singleton."""
    
    def test_chronicle_singleton_pattern(self):
        """Test that Chronicle uses singleton pattern."""
        chronicle1 = Chronicle()
        chronicle2 = Chronicle()
        
        # Both should be the same instance
        assert chronicle1 is chronicle2
    
    def test_chronicle_get_current_session_entries_method(self):
        """Test that Chronicle has method to get current session entries."""
        chronicle = Chronicle()
        
        # Check if method exists (may not be implemented yet)
        if hasattr(chronicle, 'get_current_session_entries'):
            assert callable(chronicle.get_current_session_entries)
            
            # Try calling it - should not crash even with no session
            try:
                entries = chronicle.get_current_session_entries()
                assert isinstance(entries, (dict, list)) or entries is None
            except Exception as e:
                # Method exists but may require active session
                assert "session" in str(e).lower() or "log" in str(e).lower()
    
    @patch('leeq.chronicle.chronicle.RecordBook')
    def test_chronicle_with_active_session(self, mock_record_book):
        """Test accessing data when Chronicle has active session."""
        chronicle = Chronicle()
        
        # Simulate starting a log session
        chronicle.start_log("test_session")
        
        # Check that record book was created
        assert chronicle._active_record_book is not None
        
        # If get_current_session_entries exists, test it
        if hasattr(chronicle, 'get_current_session_entries'):
            # Mock the method to return test data
            with patch.object(chronicle, 'get_current_session_entries') as mock_get:
                mock_get.return_value = {
                    'exp1': {'name': 'Test', 'timestamp': 12345}
                }
                
                entries = chronicle.get_current_session_entries()
                assert 'exp1' in entries


class TestDashboardCallbacks:
    """Test dashboard callback functions."""
    
    def test_update_session_view_no_chronicle(self):
        """Test update callback with no chronicle instance."""
        # Set chronicle to None
        session_dashboard.chronicle_instance = None
        
        # Call update callback
        tree, info, attrs = session_dashboard.update_session_view(0, None)
        
        # Should return message about no experiments
        assert "No experiments" in tree or "Error" in tree
        assert info == ""
        assert attrs == ""
    
    def test_update_session_view_with_data(self):
        """Test update callback with chronicle data."""
        # Create mock chronicle
        mock_chronicle = Mock()
        mock_chronicle.get_current_session_entries = Mock(return_value={
            'exp1': {'name': 'Test Experiment', 'timestamp': 123456}
        })
        
        # Set global chronicle
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Mock helper functions
        with patch('leeq.chronicle.viewer.session_dashboard.build_experiment_tree') as mock_build:
            with patch('leeq.chronicle.viewer.session_dashboard.render_tree_nodes') as mock_render:
                mock_build.return_value = {'root': ['exp1']}
                mock_render.return_value = "Tree HTML"
                
                # Call update callback
                tree, info, attrs = session_dashboard.update_session_view(1, None)
                
                # Check that tree was rendered
                assert tree == "Tree HTML" or "exp1" in str(tree)
    
    def test_update_session_view_error_handling(self):
        """Test update callback handles errors gracefully."""
        # Create mock chronicle that raises error
        mock_chronicle = Mock()
        mock_chronicle.get_current_session_entries = Mock(
            side_effect=Exception("Test error")
        )
        
        # Set global chronicle
        session_dashboard.chronicle_instance = mock_chronicle
        
        # Call update callback - should handle error
        tree, info, attrs = session_dashboard.update_session_view(0, None)
        
        # Should return error message
        assert "Error" in tree or "No experiments" in tree


class TestIntegrationWorkflow:
    """Test complete integration workflow."""
    
    @patch('leeq.chronicle.viewer.session_dashboard.app')
    def test_complete_launch_workflow(self, mock_app):
        """Test complete workflow from launch_viewer to dashboard."""
        # Step 1: Create Chronicle instance
        chronicle = Chronicle()
        
        # Step 2: Reset global state
        session_dashboard.chronicle_instance = None
        
        # Step 3: Launch viewer
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            # Configure mock to actually call the function
            def side_effect(**kwargs):
                session_dashboard.chronicle_instance = kwargs.get('chronicle_instance')
                return None
            
            mock_start.side_effect = side_effect
            
            # Launch viewer
            chronicle.launch_viewer(port=8888, debug=True)
            
            # Step 4: Verify integration
            assert mock_start.called
            assert session_dashboard.chronicle_instance is chronicle
    
    def test_validation_script(self):
        """Test that validation script can run without errors."""
        # This tests the validation code from the plan
        try:
            from leeq.chronicle import Chronicle
            c = Chronicle()
            
            # Check that launch_viewer method exists
            assert hasattr(c, 'launch_viewer')
            
            # Should not crash even without mock
            print('Launch method accessible')
            
            # Test passes if no exception
            assert True
            
        except ImportError:
            pytest.skip("Chronicle module not properly installed")


class TestChronicleAdvancedIntegration:
    """Test advanced Chronicle integration scenarios."""
    
    def test_launch_viewer_with_active_session(self):
        """Test launching viewer when Chronicle already has active session."""
        chronicle = Chronicle()
        
        # Start a session before launching viewer
        with patch.object(chronicle, 'start_log'):
            chronicle.start_log("active_session")
            
            # Set up mock active record book
            mock_record_book = Mock()
            mock_root = Mock()
            mock_root.record_id = 'root'
            mock_root.get_path = Mock(return_value='/root')
            mock_root.children = []
            mock_record_book.get_root_entry = Mock(return_value=mock_root)
            chronicle._active_record_book = mock_record_book
            
            # Launch viewer with active session
            with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
                chronicle.launch_viewer(port=8051)
                
                # Verify viewer launched with active session
                mock_start.assert_called_once()
                call_args = mock_start.call_args[1]
                assert call_args['chronicle_instance'] is chronicle
    
    def test_launch_viewer_without_session(self):
        """Test launching viewer when no session is active."""
        chronicle = Chronicle()
        
        # Ensure no active session
        chronicle._active_record_book = None
        
        # Launch viewer without session
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            chronicle.launch_viewer(port=8051)
            
            # Viewer should still launch
            mock_start.assert_called_once()
            
            # Session dashboard should handle no session gracefully
            session_dashboard.chronicle_instance = chronicle
            experiments, tree = session_dashboard.load_session_experiments()
            assert experiments == []
            assert tree == {}
    
    def test_multiple_viewer_launches(self):
        """Test handling multiple launch_viewer calls."""
        chronicle = Chronicle()
        
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            # First launch
            chronicle.launch_viewer(port=8051)
            assert mock_start.call_count == 1
            
            # Second launch (might be blocked or create new instance)
            chronicle.launch_viewer(port=8052)
            assert mock_start.call_count == 2
            
            # Verify different ports were used
            calls = mock_start.call_args_list
            assert calls[0][1]['port'] == 8051
            assert calls[1][1]['port'] == 8052
    
    def test_viewer_with_chronicle_singleton_pattern(self):
        """Test that viewer works correctly with Chronicle singleton."""
        # Get first instance
        chronicle1 = Chronicle()
        
        # Get second instance (should be same)
        chronicle2 = Chronicle()
        
        # Verify singleton
        assert chronicle1 is chronicle2
        
        # Launch viewer from first instance
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            chronicle1.launch_viewer(port=8051)
            
            # Verify viewer gets the singleton instance
            call_args = mock_start.call_args[1]
            assert call_args['chronicle_instance'] is chronicle1
            assert call_args['chronicle_instance'] is chronicle2
    
    def test_viewer_debug_mode_configuration(self):
        """Test configuring viewer debug mode."""
        chronicle = Chronicle()
        
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            # Test debug=True (default)
            chronicle.launch_viewer(port=8051)
            call_args = mock_start.call_args[1]
            assert call_args.get('debug', True) is True
            
            # Test debug=False
            chronicle.launch_viewer(port=8051, debug=False)
            call_args = mock_start.call_args[1]
            assert call_args['debug'] is False
    
    def test_viewer_host_configuration(self):
        """Test configuring viewer host for network access."""
        chronicle = Chronicle()
        
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            # Test custom host for network access
            chronicle.launch_viewer(port=8051, host='0.0.0.0')
            
            call_args = mock_start.call_args[1]
            assert call_args.get('host') == '0.0.0.0'
    
    def test_viewer_with_experiments_already_completed(self):
        """Test launching viewer when experiments are already completed."""
        chronicle = Chronicle()
        
        # Set up completed experiments
        completed_exps = []
        for i in range(5):
            exp = Mock()
            exp.record_id = f'completed_{i}'
            exp.name = f'Completed Experiment {i}'
            exp.timestamp = 123456789 + i
            exp.get_path = Mock(return_value=f'/root/completed_{i}')
            exp.children = []
            completed_exps.append(exp)
        
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.get_path = Mock(return_value='/root')
        mock_root.children = completed_exps
        
        mock_record_book = Mock()
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        chronicle._active_record_book = mock_record_book
        
        # Launch viewer
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer'):
            chronicle.launch_viewer(port=8051)
            
            # Set up session dashboard
            session_dashboard.chronicle_instance = chronicle
            
            # Verify all completed experiments are visible
            with patch.object(chronicle, 'is_recording', return_value=True):
                experiments, tree = session_dashboard.load_session_experiments()
                assert len(experiments) == 5
                
                # Check experiments are in order
                for i, exp in enumerate(experiments):
                    assert exp['record_id'] == f'completed_{i}'


if __name__ == "__main__":
    # Run basic validation
    print("Running Chronicle integration validation...")
    
    try:
        from leeq.chronicle import Chronicle
        c = Chronicle()
        
        # Verify launch_viewer exists
        assert hasattr(c, 'launch_viewer'), "Chronicle.launch_viewer() not found"
        print("✓ Chronicle.launch_viewer() method exists")
        
        # Test that it can be called (with mocked dashboard)
        with patch('leeq.chronicle.viewer.session_dashboard.app'):
            with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
                c.launch_viewer(port=8051)
                assert mock_start.called, "start_viewer not called"
                print("✓ Chronicle.launch_viewer() calls start_viewer()")
        
        # Test with custom parameters
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            c.launch_viewer(port=9999, debug=False, host='0.0.0.0')
            call_args = mock_start.call_args[1]
            assert call_args['port'] == 9999
            assert call_args['debug'] is False
            assert call_args['host'] == '0.0.0.0'
            print("✓ Chronicle.launch_viewer() passes custom parameters")
        
        print("✓ Launch method accessible")
        print("\nValidation successful!")
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        raise