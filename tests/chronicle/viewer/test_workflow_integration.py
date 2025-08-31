"""
Test end-to-end workflow integration for Chronicle viewer.

This module tests the complete workflow from launching the viewer
during a calibration script to monitoring experiment progress.

Tests include:
1. Complete workflow from script to viewer
2. Integration with generate_chronicle_logs.py
3. Multi-experiment session handling
4. Real-time update verification
5. Concurrent access scenarios
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call, ANY
import sys
import os
import threading
import time
from datetime import datetime
import tempfile
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Mock the dash and plotly modules
sys.modules['dash'] = MagicMock()
sys.modules['dash.dependencies'] = MagicMock()
sys.modules['dash_bootstrap_components'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.tools'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()

# Import Chronicle first
from leeq.chronicle import Chronicle

# Import session_dashboard with mocked dependencies
from leeq.chronicle.viewer import session_dashboard


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow scenarios."""
    
    def test_launch_viewer_during_calibration(self):
        """Test launching viewer during a calibration session."""
        # Simulate calibration script workflow
        chronicle = Chronicle()
        
        # Mock the dashboard start to prevent actual server
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            # Start a log session (simulating calibration start)
            with patch.object(chronicle, 'start_log'):
                chronicle.start_log("test_calibration_session")
                
                # Launch the viewer
                chronicle.launch_viewer(port=8052)
                
                # Verify viewer was launched with chronicle instance
                mock_start.assert_called_once()
                call_args = mock_start.call_args[1]
                assert call_args['chronicle_instance'] is chronicle
                assert call_args['port'] == 8052
    
    def test_viewer_updates_as_experiments_complete(self):
        """Test that viewer updates as experiments are completed."""
        chronicle = Chronicle()
        
        # Create mock record book with evolving experiments
        mock_record_book = Mock()
        experiments_completed = []
        
        def mock_get_root():
            """Return root with growing list of completed experiments."""
            root = Mock()
            root.record_id = 'root'
            root.get_path = Mock(return_value='/root')
            root.children = []
            
            for i, exp_id in enumerate(experiments_completed):
                exp = Mock()
                exp.record_id = exp_id
                exp.name = f'Experiment {i}'
                exp.timestamp = 123456789 + i
                exp.get_path = Mock(return_value=f'/root/{exp_id}')
                exp.children = []
                root.children.append(exp)
            
            return root
        
        mock_record_book.get_root_entry = mock_get_root
        
        # Set up chronicle with mock record book
        with patch.object(chronicle, 'is_recording', return_value=True):
            chronicle._active_record_book = mock_record_book
            session_dashboard.chronicle_instance = chronicle
            
            # Initial state - no experiments
            exps1, tree1 = session_dashboard.load_session_experiments()
            assert len(exps1) == 0
            
            # Add first experiment
            experiments_completed.append('exp001')
            exps2, tree2 = session_dashboard.load_session_experiments()
            assert len(exps2) == 1
            assert exps2[0]['record_id'] == 'exp001'
            
            # Add second experiment
            experiments_completed.append('exp002')
            exps3, tree3 = session_dashboard.load_session_experiments()
            assert len(exps3) == 2
            assert exps3[1]['record_id'] == 'exp002'
    
    def test_concurrent_experiment_execution(self):
        """Test viewer handling concurrent experiment execution."""
        chronicle = Chronicle()
        
        # Simulate multiple experiments running concurrently
        with patch.object(chronicle, 'is_recording', return_value=True):
            # Create mock experiments that complete at different times
            exp_threads = []
            completed_exps = []
            
            def simulate_experiment(exp_id, duration):
                """Simulate an experiment that takes time to complete."""
                time.sleep(duration)
                completed_exps.append(exp_id)
            
            # Start multiple experiments
            for i in range(5):
                thread = threading.Thread(
                    target=simulate_experiment,
                    args=(f'exp{i:03d}', i * 0.1)
                )
                thread.start()
                exp_threads.append(thread)
            
            # Mock the record book to return completed experiments
            mock_record_book = Mock()
            
            def get_completed_experiments():
                root = Mock()
                root.record_id = 'root'
                root.get_path = Mock(return_value='/root')
                root.children = []
                
                for exp_id in completed_exps:
                    exp = Mock()
                    exp.record_id = exp_id
                    exp.name = f'Experiment {exp_id}'
                    exp.timestamp = time.time()
                    exp.get_path = Mock(return_value=f'/root/{exp_id}')
                    exp.children = []
                    root.children.append(exp)
                
                return root
            
            mock_record_book.get_root_entry = get_completed_experiments
            chronicle._active_record_book = mock_record_book
            session_dashboard.chronicle_instance = chronicle
            
            # Poll for updates while experiments complete
            for _ in range(10):
                exps, tree = session_dashboard.load_session_experiments()
                time.sleep(0.1)
            
            # Wait for all threads to complete
            for thread in exp_threads:
                thread.join()
            
            # Final check - all experiments should be visible
            final_exps, final_tree = session_dashboard.load_session_experiments()
            assert len(final_exps) == 5
    
    def test_viewer_persistence_across_sessions(self):
        """Test that viewer can handle session changes."""
        chronicle = Chronicle()
        
        # First session
        with patch.object(chronicle, 'start_log'):
            with patch.object(chronicle, 'stop_log'):
                chronicle.start_log("session1")
                
                # Mock first session data
                mock_record_book1 = Mock()
                mock_root1 = Mock()
                mock_root1.record_id = 'root1'
                mock_root1.get_path = Mock(return_value='/root')
                mock_root1.children = []
                mock_record_book1.get_root_entry = Mock(return_value=mock_root1)
                
                chronicle._active_record_book = mock_record_book1
                session_dashboard.chronicle_instance = chronicle
                
                with patch.object(chronicle, 'is_recording', return_value=True):
                    exps1, tree1 = session_dashboard.load_session_experiments()
                    assert exps1 == []
                
                chronicle.stop_log()
        
        # Second session
        with patch.object(chronicle, 'start_log'):
            chronicle.start_log("session2")
            
            # Mock second session data
            mock_record_book2 = Mock()
            mock_root2 = Mock()
            mock_root2.record_id = 'root2'
            mock_root2.get_path = Mock(return_value='/root')
            
            # Add experiment to second session
            mock_exp = Mock()
            mock_exp.record_id = 'exp_session2'
            mock_exp.name = 'Session 2 Experiment'
            mock_exp.timestamp = 123456789
            mock_exp.get_path = Mock(return_value='/root/exp')
            mock_exp.children = []
            mock_root2.children = [mock_exp]
            
            mock_record_book2.get_root_entry = Mock(return_value=mock_root2)
            chronicle._active_record_book = mock_record_book2
            
            with patch.object(chronicle, 'is_recording', return_value=True):
                exps2, tree2 = session_dashboard.load_session_experiments()
                assert len(exps2) == 1
                assert exps2[0]['record_id'] == 'exp_session2'


class TestCalibrationScriptIntegration:
    """Test integration with calibration scripts like generate_chronicle_logs.py."""
    
    @patch('leeq.chronicle.viewer.session_dashboard.app')
    def test_generate_chronicle_logs_workflow(self, mock_app):
        """Test workflow similar to generate_chronicle_logs.py."""
        # Simulate the workflow from generate_chronicle_logs.py
        
        # 1. Setup phase
        chronicle = Chronicle()
        
        # 2. Launch viewer
        with patch('leeq.chronicle.viewer.session_dashboard.start_viewer') as mock_start:
            chronicle.launch_viewer(port=8051)
            assert mock_start.called
        
        # 3. Start logging session
        with patch.object(chronicle, 'start_log'):
            chronicle.start_log("calibration_session")
        
        # 4. Run experiments (simulated)
        experiments_to_run = [
            "resonator_spectroscopy",
            "qubit_spectroscopy",
            "rabi_oscillation",
            "t1_measurement",
            "t2_measurement"
        ]
        
        # Mock record book for experiment tracking
        mock_record_book = Mock()
        completed_experiments = []
        
        def add_experiment(exp_name):
            """Simulate completing an experiment."""
            exp = Mock()
            exp.record_id = f'{exp_name}_{uuid.uuid4().hex[:8]}'
            exp.name = exp_name
            exp.timestamp = time.time()
            exp.get_path = Mock(return_value=f'/root/{exp_name}')
            exp.children = []
            completed_experiments.append(exp)
        
        def get_root_with_experiments():
            root = Mock()
            root.record_id = 'root'
            root.get_path = Mock(return_value='/root')
            root.children = completed_experiments.copy()
            return root
        
        mock_record_book.get_root_entry = get_root_with_experiments
        chronicle._active_record_book = mock_record_book
        session_dashboard.chronicle_instance = chronicle
        
        # 5. Execute experiments one by one
        with patch.object(chronicle, 'is_recording', return_value=True):
            for exp_name in experiments_to_run:
                # Simulate experiment execution
                add_experiment(exp_name)
                
                # Check viewer would show the experiment
                exps, tree = session_dashboard.load_session_experiments()
                assert len(exps) == len(completed_experiments)
                
                # Verify latest experiment is visible
                latest_exp = exps[-1] if exps else None
                assert latest_exp is not None
                assert exp_name in latest_exp['name']
        
        # 6. Final verification
        final_exps, final_tree = session_dashboard.load_session_experiments()
        assert len(final_exps) == len(experiments_to_run)
    
    def test_pause_for_monitoring_pattern(self):
        """Test the pattern of pausing to check viewer."""
        # This simulates the pattern: run experiment -> pause -> check viewer
        chronicle = Chronicle()
        
        with patch.object(chronicle, 'is_recording', return_value=True):
            # Mock experiments
            mock_record_book = Mock()
            experiments = []
            
            def get_root():
                root = Mock()
                root.record_id = 'root'
                root.get_path = Mock(return_value='/root')
                root.children = experiments.copy()
                return root
            
            mock_record_book.get_root_entry = get_root
            chronicle._active_record_book = mock_record_book
            session_dashboard.chronicle_instance = chronicle
            
            # Simulate workflow with pauses
            exp_stages = [
                ("Resonator Tuning", 3),
                ("Qubit Tuning", 5),
                ("Gate Calibration", 2)
            ]
            
            for stage_name, num_experiments in exp_stages:
                # Run experiments for this stage
                for i in range(num_experiments):
                    exp = Mock()
                    exp.record_id = f'{stage_name}_{i}'
                    exp.name = f'{stage_name} Exp {i}'
                    exp.timestamp = time.time()
                    exp.get_path = Mock(return_value=f'/root/{stage_name}_{i}')
                    exp.children = []
                    experiments.append(exp)
                
                # Check viewer state (simulating pause to monitor)
                exps, tree = session_dashboard.load_session_experiments()
                
                # Verify experiments are visible
                stage_exps = [e for e in exps if stage_name in e['name']]
                assert len(stage_exps) == num_experiments


class TestDataFlowIntegration:
    """Test data flow from Chronicle to UI components."""
    
    def test_chronicle_to_tree_view_flow(self):
        """Test data flow from Chronicle to tree view component."""
        chronicle = Chronicle()
        
        # Set up mock chronicle with hierarchical data
        mock_record_book = Mock()
        
        # Create hierarchical experiment structure
        root = Mock()
        root.record_id = 'root'
        root.get_path = Mock(return_value='/root')
        
        # Parent experiment
        parent = Mock()
        parent.record_id = 'parent_exp'
        parent.name = 'Parent Experiment'
        parent.timestamp = 123456789
        parent.get_path = Mock(return_value='/root/parent')
        
        # Child experiments
        child1 = Mock()
        child1.record_id = 'child1'
        child1.name = 'Child 1'
        child1.timestamp = 123456790
        child1.get_path = Mock(return_value='/root/parent/child1')
        child1.children = []
        
        child2 = Mock()
        child2.record_id = 'child2'
        child2.name = 'Child 2'
        child2.timestamp = 123456791
        child2.get_path = Mock(return_value='/root/parent/child2')
        child2.children = []
        
        parent.children = [child1, child2]
        root.children = [parent]
        
        mock_record_book.get_root_entry = Mock(return_value=root)
        chronicle._active_record_book = mock_record_book
        
        with patch.object(chronicle, 'is_recording', return_value=True):
            session_dashboard.chronicle_instance = chronicle
            
            # Load experiments
            experiments, tree = session_dashboard.load_session_experiments()
            
            # Verify hierarchical structure is preserved
            assert len(experiments) == 3  # parent + 2 children
            
            # Check parent-child relationships in tree
            with patch('leeq.chronicle.viewer.session_dashboard.create_tree_view_items') as mock_create:
                mock_create.return_value = {
                    'parent_exp': ['child1', 'child2']
                }
                
                experiments, tree = session_dashboard.load_session_experiments()
                assert 'parent_exp' in tree
                assert 'child1' in tree['parent_exp']
                assert 'child2' in tree['parent_exp']
    
    def test_experiment_selection_to_display_flow(self):
        """Test data flow from experiment selection to display panels."""
        # Set up chronicle with experiment data
        chronicle = Chronicle()
        mock_exp = Mock()
        mock_exp.record_id = 'test_exp'
        mock_exp.name = 'Test Experiment'
        mock_exp.timestamp = 123456789
        mock_exp.get_path = Mock(return_value='/root/test')
        mock_exp.children = []
        
        mock_root = Mock()
        mock_root.record_id = 'root'
        mock_root.get_path = Mock(return_value='/root')
        mock_root.children = [mock_exp]
        
        mock_record_book = Mock()
        mock_record_book.get_root_entry = Mock(return_value=mock_root)
        chronicle._active_record_book = mock_record_book
        
        with patch.object(chronicle, 'is_recording', return_value=True):
            session_dashboard.chronicle_instance = chronicle
            
            # Load and select experiment
            experiments, tree = session_dashboard.load_session_experiments()
            
            # Simulate selection (would trigger callback in real app)
            selected_id = experiments[0]['record_id']
            
            # Verify data is available for display
            assert selected_id == 'test_exp'
            selected_exp = next(e for e in experiments if e['record_id'] == selected_id)
            assert selected_exp['name'] == 'Test Experiment'
    
    def test_live_update_data_consistency(self):
        """Test data consistency during live updates."""
        chronicle = Chronicle()
        
        # Track data versions
        data_versions = []
        
        def create_snapshot(num_experiments):
            """Create a consistent snapshot of experiments."""
            exps = []
            for i in range(num_experiments):
                exp = Mock()
                exp.record_id = f'exp{i:03d}'
                exp.name = f'Experiment {i}'
                exp.timestamp = 123456789 + i
                exp.get_path = Mock(return_value=f'/root/exp{i}')
                exp.children = []
                exps.append(exp)
            return exps
        
        # Simulate multiple update cycles
        for num_exps in [0, 2, 5, 10]:
            mock_record_book = Mock()
            
            current_snapshot = create_snapshot(num_exps)
            
            def get_root():
                root = Mock()
                root.record_id = 'root'
                root.get_path = Mock(return_value='/root')
                root.children = current_snapshot
                return root
            
            mock_record_book.get_root_entry = get_root
            chronicle._active_record_book = mock_record_book
            
            with patch.object(chronicle, 'is_recording', return_value=True):
                session_dashboard.chronicle_instance = chronicle
                
                # Load data
                experiments, tree = session_dashboard.load_session_experiments()
                
                # Verify consistency
                assert len(experiments) == num_exps
                
                # Check ordering is preserved
                for i, exp in enumerate(experiments):
                    assert exp['record_id'] == f'exp{i:03d}'
                
                data_versions.append(len(experiments))
        
        # Verify progressive updates
        assert data_versions == [0, 2, 5, 10]


class TestErrorRecovery:
    """Test error recovery and resilience."""
    
    def test_recover_from_chronicle_crash(self):
        """Test recovery when Chronicle crashes and restarts."""
        # Initial chronicle instance
        chronicle1 = Chronicle()
        session_dashboard.chronicle_instance = chronicle1
        
        # Simulate chronicle crash
        session_dashboard.chronicle_instance = None
        
        # Viewer should handle gracefully
        experiments, tree = session_dashboard.load_session_experiments()
        assert experiments == []
        assert tree == {}
        
        # New chronicle instance (recovery)
        chronicle2 = Chronicle()
        session_dashboard.chronicle_instance = chronicle2
        
        # Viewer should work with new instance
        with patch.object(chronicle2, 'is_recording', return_value=False):
            experiments, tree = session_dashboard.load_session_experiments()
            assert experiments == []
            assert tree == {}
    
    def test_handle_intermittent_failures(self):
        """Test handling intermittent data access failures."""
        chronicle = Chronicle()
        session_dashboard.chronicle_instance = chronicle
        
        # Create mock that fails intermittently
        call_count = 0
        
        def intermittent_is_recording():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise ConnectionError("Temporary failure")
            return True
        
        with patch.object(chronicle, 'is_recording', side_effect=intermittent_is_recording):
            chronicle._active_record_book = Mock()
            chronicle._active_record_book.get_root_entry = Mock(return_value=Mock(children=[]))
            
            # Multiple attempts should handle intermittent failures
            success_count = 0
            failure_count = 0
            
            for _ in range(10):
                try:
                    experiments, tree = session_dashboard.load_session_experiments()
                    if experiments is not None:
                        success_count += 1
                except:
                    failure_count += 1
            
            # Some calls should succeed despite intermittent failures
            assert success_count > 0
    
    def test_graceful_degradation(self):
        """Test graceful degradation when features are unavailable."""
        chronicle = Chronicle()
        
        # Test with various missing methods
        missing_methods = [
            'is_recording',
            'get_current_session_entries',
            '_active_record_book'
        ]
        
        for method_name in missing_methods:
            # Temporarily remove method/attribute
            original = getattr(chronicle, method_name, None)
            if hasattr(chronicle, method_name):
                delattr(chronicle, method_name)
            
            session_dashboard.chronicle_instance = chronicle
            
            # Should handle missing method gracefully
            experiments, tree = session_dashboard.load_session_experiments()
            assert experiments == [] or experiments is None
            assert tree == {} or tree is None
            
            # Restore method/attribute if it existed
            if original is not None:
                setattr(chronicle, method_name, original)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])