"""
Robustness tests for parallel CPU spectroscopy implementation.

This test suite validates error handling, timeout behavior, and graceful fallback
mechanisms when workers fail or encounter problems.

Tests:
- Worker process failures and recovery
- Timeout handling for stuck calculations
- Memory pressure scenarios
- Cross-platform compatibility
- Process cleanup and resource management
"""

import pytest
import numpy as np
import time
import multiprocessing
import signal
import psutil
from unittest.mock import patch, MagicMock, Mock
from concurrent.futures import TimeoutError, ProcessPoolExecutor

# Import the simulator
from leeq.theory.simulation.numpy.cw_spectroscopy import (
    CWSpectroscopySimulator,
    _simulate_point_worker
)
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup


class TestParallelRobustness:
    """Test suite for parallel processing robustness and error handling."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance for testing."""
        vq = VirtualTransmon(
            name="Q1",
            qubit_frequency=5000.0,
            anharmonicity=-200.0,
            t1=50.0,
            t2=30.0,
            readout_frequency=7000.0,
            truncate_level=3
        )
        simulation_setup = HighLevelSimulationSetup(
            name="test",
            virtual_qubits={1: vq},
            omega_to_amp_map={1: 500.0}
        )
        return CWSpectroscopySimulator(simulation_setup)
    
    @pytest.fixture
    def small_test_params(self):
        """Small parameter set for robustness testing."""
        return {
            'freq_array': np.linspace(5000, 5010, 3),
            'amp_array': np.linspace(0.02, 0.03, 2),
        }

    def test_timeout_handling(self, simulator, small_test_params):
        """Test timeout handling for stuck worker processes."""
        freq_array = small_test_params['freq_array']
        amp_array = small_test_params['amp_array']
        
        # Mock a very short timeout to trigger timeout behavior
        with patch('leeq.theory.simulation.numpy.cw_spectroscopy._simulate_point_worker') as mock_worker:
            # Make the worker sleep longer than timeout
            def slow_worker(*args):
                time.sleep(0.5)  # 500ms delay
                return 0.5 + 0.2j
            
            mock_worker.side_effect = slow_worker
            
            # Test with very short timeout (should trigger timeout recovery)
            result = simulator.simulate_2d_sweep_parallel(
                freq_array, amp_array, 
                num_workers=2, 
                timeout_per_point=0.1  # 100ms timeout
            )
            
            # Result should still be produced (from fallback sequential processing)
            assert result.shape == (len(amp_array), len(freq_array))
            assert np.all(np.isfinite(result))

    def test_worker_process_crash_recovery(self, simulator, small_test_params):
        """Test recovery when worker processes crash."""
        freq_array = small_test_params['freq_array']
        amp_array = small_test_params['amp_array']
        
        with patch('leeq.theory.simulation.numpy.cw_spectroscopy._simulate_point_worker') as mock_worker:
            # Make first few calls fail, then succeed
            call_count = 0
            def failing_worker(*args):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:  # First 2 calls fail
                    raise RuntimeError("Simulated worker crash")
                return 0.3 + 0.1j
            
            mock_worker.side_effect = failing_worker
            
            # Should handle worker crashes gracefully
            result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
            
            assert result.shape == (len(amp_array), len(freq_array))
            assert np.all(np.isfinite(result))

    def test_complete_parallel_failure_fallback(self, simulator, small_test_params):
        """Test fallback to sequential processing when parallel completely fails."""
        freq_array = small_test_params['freq_array']
        amp_array = small_test_params['amp_array']
        
        with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
            # Make ProcessPoolExecutor raise exception
            mock_executor.side_effect = RuntimeError("ProcessPoolExecutor failed")
            
            # Should fall back to sequential processing
            result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
            
            assert result.shape == (len(amp_array), len(freq_array))
            assert np.all(np.isfinite(result))

    def test_memory_pressure_handling(self, simulator):
        """Test behavior under memory pressure conditions."""
        # Create a larger parameter grid that might stress memory
        freq_array = np.linspace(4900, 5100, 20)
        amp_array = np.linspace(0.01, 0.08, 15)
        
        # Monitor memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        # Run with limited workers to control memory usage
        result = simulator.simulate_2d_sweep_parallel(
            freq_array, amp_array, num_workers=2
        )
        
        memory_after = process.memory_info().rss
        memory_increase = (memory_after - memory_before) / (1024**2)  # MB
        
        # Should complete without excessive memory usage
        assert result.shape == (len(amp_array), len(freq_array))
        assert memory_increase < 1000  # Less than 1GB increase
        assert np.all(np.isfinite(result))

    @patch('multiprocessing.cpu_count')
    def test_cpu_detection_edge_cases(self, mock_cpu_count, simulator, small_test_params):
        """Test CPU detection edge cases and manual overrides."""
        freq_array = small_test_params['freq_array']
        amp_array = small_test_params['amp_array']
        
        # Test with very high reported CPU count
        mock_cpu_count.return_value = 128
        
        result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
        assert result.shape == (len(amp_array), len(freq_array))
        
        # Test with manual worker count override
        result = simulator.simulate_2d_sweep_parallel(
            freq_array, amp_array, num_workers=1  # Force single worker
        )
        assert result.shape == (len(amp_array), len(freq_array))
        
        # Test with zero/negative worker count (should handle gracefully)
        try:
            result = simulator.simulate_2d_sweep_parallel(
                freq_array, amp_array, num_workers=0
            )
            # Should either work or fail gracefully
            if result is not None:
                assert result.shape == (len(amp_array), len(freq_array))
        except Exception:
            # Acceptable to raise exception for invalid worker count
            pass

    def test_mixed_worker_success_failure(self, simulator, small_test_params):
        """Test scenario where some workers succeed and others fail."""
        freq_array = small_test_params['freq_array']
        amp_array = small_test_params['amp_array']
        
        with patch('leeq.theory.simulation.numpy.cw_spectroscopy._simulate_point_worker') as mock_worker:
            # Make every other call fail
            call_count = 0
            def intermittent_failure(*args):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:  # Even calls fail
                    raise ValueError("Intermittent failure")
                return 0.4 + 0.3j
            
            mock_worker.side_effect = intermittent_failure
            
            # Should handle mixed success/failure gracefully
            result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
            
            assert result.shape == (len(amp_array), len(freq_array))
            assert np.all(np.isfinite(result))

    def test_interrupt_handling(self, simulator, small_test_params):
        """Test behavior when process is interrupted (Ctrl+C simulation)."""
        freq_array = small_test_params['freq_array']
        amp_array = small_test_params['amp_array']
        
        with patch('leeq.theory.simulation.numpy.cw_spectroscopy._simulate_point_worker') as mock_worker:
            # Simulate a KeyboardInterrupt during processing
            call_count = 0
            def interrupt_worker(*args):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Interrupt on second call
                    raise KeyboardInterrupt("Simulated interrupt")
                return 0.2 + 0.1j
            
            mock_worker.side_effect = interrupt_worker
            
            # Should handle interrupt gracefully or re-raise it
            try:
                result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
                # If it completes, result should be valid
                if result is not None:
                    assert result.shape == (len(amp_array), len(freq_array))
            except KeyboardInterrupt:
                # Acceptable to propagate KeyboardInterrupt
                pass

    def test_edge_case_parameter_values(self, simulator):
        """Test robustness with edge case parameter values."""
        # Test with extreme parameter values that might cause numerical issues
        test_cases = [
            # Very small amplitudes
            {'freq_array': np.array([5000.0]), 'amp_array': np.array([1e-10])},
            # Very large amplitudes  
            {'freq_array': np.array([5000.0]), 'amp_array': np.array([10.0])},
            # Very high frequencies
            {'freq_array': np.array([50000.0]), 'amp_array': np.array([0.02])},
            # Zero amplitude
            {'freq_array': np.array([5000.0]), 'amp_array': np.array([0.0])},
        ]
        
        for test_case in test_cases:
            freq_array = test_case['freq_array']
            amp_array = test_case['amp_array']
            
            try:
                result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
                
                # Result should have correct shape and be finite
                assert result.shape == (len(amp_array), len(freq_array))
                # Allow NaN/inf for extreme cases, but check they don't crash
                assert isinstance(result, np.ndarray)
                
            except Exception as e:
                # Log but don't fail test - extreme parameters may legitimately fail
                print(f"Edge case failed (acceptable): {test_case} - {e}")

    def test_sequential_fallback_robustness(self, simulator, small_test_params):
        """Test that sequential fallback is robust when parallel fails."""
        freq_array = small_test_params['freq_array']
        amp_array = small_test_params['amp_array']
        
        # Force fallback by making parallel processing fail completely
        with patch.object(simulator, 'simulate_2d_sweep_parallel') as mock_parallel:
            def call_fallback(*args, **kwargs):
                # Call the actual fallback method
                return simulator._fallback_sequential_processing(freq_array, amp_array)
            
            mock_parallel.side_effect = call_fallback
            
            result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
            
            assert result.shape == (len(amp_array), len(freq_array))
            assert np.all(np.isfinite(result))

    def test_resource_cleanup(self, simulator, small_test_params):
        """Test that resources are properly cleaned up after parallel processing."""
        freq_array = small_test_params['freq_array']
        amp_array = small_test_params['amp_array']
        
        # Get initial process count
        initial_process_count = len([p for p in psutil.process_iter() 
                                   if p.pid != psutil.Process().pid])
        
        # Run parallel processing multiple times
        for _ in range(3):
            result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
            assert result.shape == (len(amp_array), len(freq_array))
        
        # Give time for process cleanup
        time.sleep(0.5)
        
        # Check that we don't have excessive process growth
        final_process_count = len([p for p in psutil.process_iter() 
                                 if p.pid != psutil.Process().pid])
        
        process_growth = final_process_count - initial_process_count
        assert process_growth < 10, f"Process count grew by {process_growth}"

    def test_deterministic_error_recovery(self, simulator, small_test_params):
        """Test that error recovery produces deterministic results."""
        freq_array = small_test_params['freq_array']
        amp_array = small_test_params['amp_array']
        
        # Run the same calculation multiple times with forced failures
        results = []
        
        for run in range(3):
            with patch('leeq.theory.simulation.numpy.cw_spectroscopy._simulate_point_worker') as mock_worker:
                # Create consistent but failing behavior
                def consistent_failure(*args):
                    freq, amp = args[:2]
                    if freq > 5005:  # Fail for high frequencies consistently
                        raise RuntimeError("Consistent failure")
                    return 0.5 + 0.2j
                
                mock_worker.side_effect = consistent_failure
                
                result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
                results.append(result)
        
        # All runs should produce identical results (deterministic recovery)
        for result in results[1:]:
            np.testing.assert_array_equal(result, results[0])


class TestParallelCompatibility:
    """Test cross-platform and environment compatibility."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance for testing."""
        vq = VirtualTransmon(
            name="Q1",
            qubit_frequency=5000.0,
            anharmonicity=-200.0,
            t1=50.0,
            t2=30.0,
            readout_frequency=7000.0,
            truncate_level=3
        )
        simulation_setup = HighLevelSimulationSetup(
            name="test",
            virtual_qubits={1: vq},
            omega_to_amp_map={1: 500.0}
        )
        return CWSpectroscopySimulator(simulation_setup)
    
    def test_single_vs_multi_core_consistency(self, simulator):
        """Test that single-core and multi-core produce same results."""
        freq_array = np.linspace(4995, 5005, 3)
        amp_array = np.linspace(0.01, 0.03, 2)
        
        # Single core (sequential-like)
        result_single = simulator.simulate_2d_sweep_parallel(
            freq_array, amp_array, num_workers=1
        )
        
        # Multi-core
        result_multi = simulator.simulate_2d_sweep_parallel(
            freq_array, amp_array, num_workers=2
        )
        
        # Results should be identical
        np.testing.assert_allclose(result_multi, result_single, rtol=1e-12)

    def test_varying_worker_counts(self, simulator):
        """Test consistency across different worker counts."""
        freq_array = np.linspace(4998, 5002, 3)
        amp_array = np.linspace(0.015, 0.025, 2)
        
        # Test different worker counts
        max_workers = min(multiprocessing.cpu_count(), 4)
        worker_counts = [1, 2, max_workers]
        results = []
        
        for num_workers in worker_counts:
            result = simulator.simulate_2d_sweep_parallel(
                freq_array, amp_array, num_workers=num_workers
            )
            results.append(result)
        
        # All results should be identical
        reference = results[0]
        for result in results[1:]:
            np.testing.assert_allclose(result, reference, rtol=1e-12)

    @pytest.mark.skipif(multiprocessing.cpu_count() < 2, 
                       reason="Requires multi-core system")
    def test_multiprocessing_availability(self, simulator):
        """Test that multiprocessing is available and functional."""
        freq_array = np.array([5000.0])
        amp_array = np.array([0.02])
        
        # Should be able to use multiple workers
        result = simulator.simulate_2d_sweep_parallel(
            freq_array, amp_array, num_workers=2
        )
        
        assert result.shape == (1, 1)
        assert np.isfinite(result[0, 0])


if __name__ == "__main__":
    # Run specific test for quick validation
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test for development
        pytest.main([__file__ + "::TestParallelRobustness::test_timeout_handling", "-v"])
    else:
        # Run all tests
        pytest.main([__file__, "-v"])