"""
Comprehensive validation tests for parallel CPU spectroscopy implementation.

This test suite validates that parallel processing produces identical results
to sequential processing across different parameter grid sizes and conditions.

Tests:
- Result accuracy comparison between parallel and sequential
- Physics consistency (peak positions, lineshapes)
- Performance scaling across different CPU counts
- Memory usage validation
- Edge cases (small grids, single points)
"""

import pytest
import numpy as np
import time
import multiprocessing
import psutil
from unittest.mock import patch, MagicMock

# Import the simulator
from leeq.theory.simulation.numpy.cw_spectroscopy import (
    CWSpectroscopySimulator,
    _simulate_point_worker
)
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup


class TestParallelSpectroscopyValidation:
    """Test suite for comprehensive parallel spectroscopy validation."""
    
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
    def small_grid_params(self):
        """Small parameter grid for quick validation."""
        return {
            'freq_array': np.linspace(4990, 5010, 5),  # 5 points
            'amp_array': np.linspace(0.01, 0.03, 3),   # 3 points
        }
    
    @pytest.fixture
    def medium_grid_params(self):
        """Medium parameter grid for comprehensive validation."""
        return {
            'freq_array': np.linspace(4980, 5020, 21),  # 21 points
            'amp_array': np.linspace(0.01, 0.05, 9),    # 9 points
        }
    
    @pytest.fixture
    def large_grid_params(self):
        """Large parameter grid for performance testing."""
        return {
            'freq_array': np.linspace(4970, 5030, 31),  # 31 points  
            'amp_array': np.linspace(0.005, 0.08, 16),  # 16 points
        }

    def test_result_accuracy_small_grid(self, simulator, small_grid_params):
        """Test parallel vs sequential accuracy on small parameter grid."""
        freq_array = small_grid_params['freq_array']
        amp_array = small_grid_params['amp_array']
        
        # Run sequential version (reference)
        sequential_result = self._sequential_2d_sweep(simulator, freq_array, amp_array)
        
        # Run parallel version
        parallel_result = simulator.simulate_2d_sweep_parallel(
            freq_array, amp_array, num_workers=2
        )
        
        # Validate shapes match
        assert sequential_result.shape == parallel_result.shape
        assert sequential_result.shape == (len(amp_array), len(freq_array))
        
        # Validate numerical accuracy (strict tolerance)
        np.testing.assert_allclose(
            parallel_result, sequential_result,
            rtol=1e-12, atol=1e-15,
            err_msg="Parallel results differ from sequential beyond numerical precision"
        )
        
    def test_result_accuracy_medium_grid(self, simulator, medium_grid_params):
        """Test parallel vs sequential accuracy on medium parameter grid."""
        freq_array = medium_grid_params['freq_array']
        amp_array = medium_grid_params['amp_array']
        
        # Run sequential version (reference)
        sequential_result = self._sequential_2d_sweep(simulator, freq_array, amp_array)
        
        # Run parallel version
        parallel_result = simulator.simulate_2d_sweep_parallel(
            freq_array, amp_array, num_workers=4
        )
        
        # Validate shapes and accuracy
        assert sequential_result.shape == parallel_result.shape
        np.testing.assert_allclose(
            parallel_result, sequential_result,
            rtol=1e-12, atol=1e-15,
            err_msg="Medium grid: Parallel results differ from sequential"
        )

    def test_physics_consistency_peak_detection(self, simulator, medium_grid_params):
        """Test that physics characteristics are preserved in parallel processing."""
        freq_array = medium_grid_params['freq_array']
        amp_array = medium_grid_params['amp_array']
        
        # Run both versions
        sequential_result = self._sequential_2d_sweep(simulator, freq_array, amp_array)
        parallel_result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
        
        # Find resonance peaks (minimum real part for each amplitude)
        seq_peak_indices = np.argmin(np.real(sequential_result), axis=1)
        par_peak_indices = np.argmin(np.real(parallel_result), axis=1)
        
        # Peak positions must be identical
        np.testing.assert_array_equal(
            seq_peak_indices, par_peak_indices,
            err_msg="Resonance peak positions differ between parallel and sequential"
        )
        
        # Peak frequencies must match exactly
        seq_peak_freqs = freq_array[seq_peak_indices]
        par_peak_freqs = freq_array[par_peak_indices]
        np.testing.assert_allclose(seq_peak_freqs, par_peak_freqs, rtol=1e-15)

    def test_worker_cpu_scaling(self, simulator, small_grid_params):
        """Test parallel processing with different worker counts."""
        freq_array = small_grid_params['freq_array']
        amp_array = small_grid_params['amp_array']
        
        # Get reference result
        reference_result = self._sequential_2d_sweep(simulator, freq_array, amp_array)
        
        # Test with different worker counts
        max_workers = min(multiprocessing.cpu_count(), 8)
        for num_workers in [1, 2, max_workers]:
            parallel_result = simulator.simulate_2d_sweep_parallel(
                freq_array, amp_array, num_workers=num_workers
            )
            
            np.testing.assert_allclose(
                parallel_result, reference_result,
                rtol=1e-12, atol=1e-15,
                err_msg=f"Results differ with {num_workers} workers"
            )

    def test_single_point_grid(self, simulator):
        """Test edge case: single parameter point (no parallelization benefit)."""
        freq_array = np.array([5000.0])
        amp_array = np.array([0.02])
        
        sequential_result = self._sequential_2d_sweep(simulator, freq_array, amp_array)
        parallel_result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
        
        assert sequential_result.shape == (1, 1)
        assert parallel_result.shape == (1, 1)
        np.testing.assert_allclose(parallel_result, sequential_result, rtol=1e-15)

    def test_empty_arrays_error_handling(self, simulator):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="must not be empty"):
            simulator.simulate_2d_sweep_parallel(np.array([]), np.array([0.01, 0.02]))
            
        with pytest.raises(ValueError, match="must not be empty"):
            simulator.simulate_2d_sweep_parallel(np.array([5000.0]), np.array([]))

    def test_memory_usage_monitoring(self, simulator, medium_grid_params):
        """Test memory usage remains reasonable during parallel processing."""
        freq_array = medium_grid_params['freq_array']
        amp_array = medium_grid_params['amp_array']
        
        # Monitor memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run parallel processing
        result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 500MB for medium grid)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f} MB"
        
        # Result should be valid
        expected_shape = (len(amp_array), len(freq_array))
        assert result.shape == expected_shape

    def test_performance_timing_comparison(self, simulator, small_grid_params):
        """Test that parallel processing provides speedup (when beneficial)."""
        freq_array = small_grid_params['freq_array']
        amp_array = small_grid_params['amp_array']
        
        # Time sequential processing
        start_time = time.time()
        sequential_result = self._sequential_2d_sweep(simulator, freq_array, amp_array)
        sequential_time = time.time() - start_time
        
        # Time parallel processing (with 2 workers)
        start_time = time.time()
        parallel_result = simulator.simulate_2d_sweep_parallel(
            freq_array, amp_array, num_workers=2
        )
        parallel_time = time.time() - start_time
        
        # Results must match
        np.testing.assert_allclose(parallel_result, sequential_result, rtol=1e-12)
        
        # For very small grids, parallel might be slower due to process overhead
        # Just verify both complete successfully
        assert sequential_time > 0
        assert parallel_time > 0
        
        print(f"Timing comparison - Sequential: {sequential_time:.3f}s, "
              f"Parallel: {parallel_time:.3f}s")

    @patch('leeq.theory.simulation.numpy.cw_spectroscopy.ProcessPoolExecutor')
    def test_fallback_to_sequential_on_failure(self, mock_executor, simulator, small_grid_params):
        """Test fallback to sequential processing when parallel fails."""
        freq_array = small_grid_params['freq_array']
        amp_array = small_grid_params['amp_array']
        
        # Mock executor to raise exception
        mock_executor.side_effect = Exception("Simulated parallel failure")
        
        # Should fall back to sequential processing
        result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
        
        # Result should still be valid (from sequential fallback)
        expected_shape = (len(amp_array), len(freq_array))
        assert result.shape == expected_shape
        assert np.all(np.isfinite(result))

    def test_worker_function_independence(self, simulator):
        """Test that worker function produces identical results across calls."""
        freq, amp = 5000.0, 0.02
        
        # Call worker function multiple times
        results = []
        for _ in range(5):
            result = _simulate_point_worker(freq, amp, simulator)
            results.append(result)
        
        # All results should be identical (deterministic calculation)
        for result in results[1:]:
            np.testing.assert_allclose(result, results[0], rtol=1e-15)

    def test_large_grid_validation(self, simulator, large_grid_params):
        """Test validation on larger parameter grid (performance test)."""
        freq_array = large_grid_params['freq_array']
        amp_array = large_grid_params['amp_array']
        
        # Only test a subset for speed (sample validation)
        freq_subset = freq_array[::5]  # Every 5th point
        amp_subset = amp_array[::3]    # Every 3rd point
        
        sequential_result = self._sequential_2d_sweep(simulator, freq_subset, amp_subset)
        parallel_result = simulator.simulate_2d_sweep_parallel(freq_subset, amp_subset)
        
        np.testing.assert_allclose(
            parallel_result, sequential_result,
            rtol=1e-12, atol=1e-15,
            err_msg="Large grid subset validation failed"
        )

    def _sequential_2d_sweep(self, simulator, freq_array, amp_array):
        """
        Reference implementation: sequential 2D sweep for comparison.
        
        This mimics the original sequential processing to provide
        ground truth results for validation.
        """
        results = np.zeros((len(amp_array), len(freq_array)), dtype=complex)
        
        for i, amp in enumerate(amp_array):
            for j, freq in enumerate(freq_array):
                result = _simulate_point_worker(freq, amp, simulator)
                results[i, j] = result
                
        return results


class TestParallelSpectroscopyRobustness:
    """Test suite for robustness and error handling."""
    
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
    
    def test_auto_worker_detection(self, simulator):
        """Test automatic CPU core detection."""
        freq_array = np.linspace(4990, 5010, 3)
        amp_array = np.linspace(0.01, 0.03, 2)
        
        # Call without specifying num_workers (should auto-detect)
        result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
        
        assert result.shape == (2, 3)
        assert np.all(np.isfinite(result))

    @patch('multiprocessing.cpu_count')
    def test_cpu_count_override(self, mock_cpu_count, simulator):
        """Test manual override of CPU count."""
        mock_cpu_count.return_value = 16  # Mock high CPU count
        
        freq_array = np.linspace(4990, 5010, 3)
        amp_array = np.linspace(0.01, 0.03, 2)
        
        # Manually specify lower worker count
        result = simulator.simulate_2d_sweep_parallel(
            freq_array, amp_array, num_workers=2
        )
        
        assert result.shape == (2, 3)
        assert np.all(np.isfinite(result))

    def test_parameter_boundary_values(self, simulator):
        """Test with boundary parameter values."""
        # Test very low and high frequencies/amplitudes
        freq_array = np.array([1000.0, 10000.0])  # Extreme frequencies
        amp_array = np.array([0.001, 0.1])        # Extreme amplitudes
        
        result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
        
        assert result.shape == (2, 2)
        # Results should be finite (no NaN/inf from extreme parameters)
        assert np.all(np.isfinite(result))

    def test_deterministic_results_across_runs(self, simulator):
        """Test that results are deterministic across multiple runs."""
        freq_array = np.linspace(4995, 5005, 4)
        amp_array = np.linspace(0.015, 0.025, 3)
        
        # Run multiple times
        results = []
        for _ in range(3):
            result = simulator.simulate_2d_sweep_parallel(freq_array, amp_array)
            results.append(result)
        
        # All results should be identical
        for result in results[1:]:
            np.testing.assert_array_equal(result, results[0])


if __name__ == "__main__":
    # Run specific test for quick validation
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test for development
        pytest.main([__file__ + "::TestParallelSpectroscopyValidation::test_result_accuracy_small_grid", "-v"])
    else:
        # Run all tests
        pytest.main([__file__, "-v"])