"""
Integration tests for end-to-end parallel QubitSpectroscopyAmplitudeFrequency execution.

This test suite validates that the parallel processing integration works correctly
in the complete experiment workflow, from parameter setup through result generation.

Tests:
- End-to-end parallel experiment execution
- Parameter passing and validation  
- Result format and content validation
- Performance metrics collection
- Comparison with sequential execution
- Large parameter grid handling
"""

import pytest
import numpy as np
import time
import multiprocessing
from unittest.mock import patch, MagicMock

from leeq.experiments.builtin.basic.calibrations.qubit_spectroscopy import (
    QubitSpectroscopyAmplitudeFrequency
)


class TestParallelIntegration:
    """Integration tests for parallel experiment execution."""
    
    @pytest.fixture
    def mock_qubit(self):
        """Create mock qubit for testing."""
        mock_qubit = MagicMock()
        
        # Mock measurement primitive
        mock_mp = MagicMock()
        mock_mp.clone.return_value = mock_mp
        mock_mp.update_pulse_args.return_value = None
        mock_mp.set_transform_function.return_value = None
        
        # Add __name__ attribute for the sweeper
        mock_mp.update_freq = MagicMock()
        mock_mp.update_freq.__name__ = 'update_freq'
        
        mock_qubit.get_default_measurement_prim_int.return_value = mock_mp
        
        # Mock measurement primitive list
        mock_mplist = MagicMock()
        mock_qubit.get_default_measurement_prim_intlist.return_value = mock_mplist
        
        return mock_qubit
    
    @pytest.fixture
    def small_params(self):
        """Small parameter set for integration testing."""
        return {
            'start': 4990.0,
            'stop': 5010.0,
            'step': 10.0,  # 3 frequency points
            'qubit_amp_start': 0.01,
            'qubit_amp_stop': 0.03,
            'qubit_amp_step': 0.01,  # 3 amplitude points
            'num_avs': 100,
            'rep_rate': 10000,
            'disable_noise': True
        }
    
    @pytest.fixture
    def medium_params(self):
        """Medium parameter set for comprehensive testing."""
        return {
            'start': 4980.0,
            'stop': 5020.0,
            'step': 5.0,  # 9 frequency points
            'qubit_amp_start': 0.005,
            'qubit_amp_stop': 0.055,
            'qubit_amp_step': 0.005,  # 11 amplitude points
            'num_avs': 500,
            'rep_rate': 10000,
            'disable_noise': True
        }

    def test_parallel_experiment_basic_execution(self, mock_qubit, small_params):
        """Test basic parallel experiment execution with small parameter grid."""
        # Create and run parallel experiment
        exp_parallel = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=mock_qubit,
            use_parallel=True,
            num_workers=2,
            **small_params
        )
        
        # Validate experiment completed
        assert hasattr(exp_parallel, 'result')
        assert exp_parallel.result is not None
        
        # Validate result structure
        result = exp_parallel.result
        assert 'Magnitude' in result
        assert 'Phase' in result
        assert 'I' in result
        assert 'Q' in result
        
        # Validate result dimensions
        expected_freq_points = int((small_params['stop'] - small_params['start']) / small_params['step']) + 1
        expected_amp_points = int((small_params['qubit_amp_stop'] - small_params['qubit_amp_start']) / small_params['qubit_amp_step']) + 1
        
        assert result['Magnitude'].shape == (expected_amp_points, expected_freq_points)
        assert result['Phase'].shape == (expected_amp_points, expected_freq_points)
        
        # Validate performance metrics were collected
        assert hasattr(exp_parallel, 'performance_metrics')
        metrics = exp_parallel.performance_metrics
        assert metrics['parallel_enabled'] is True
        assert metrics['num_workers'] == 2
        assert metrics['execution_time'] > 0
        assert metrics['grid_size'] == (expected_amp_points, expected_freq_points)

    def test_parallel_vs_sequential_consistency(self, mock_qubit, small_params):
        """Test that parallel and sequential experiments produce identical results."""
        # Run sequential experiment
        exp_sequential = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=mock_qubit,
            use_parallel=False,
            **small_params
        )
        
        # Run parallel experiment
        exp_parallel = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=mock_qubit,
            use_parallel=True,
            num_workers=2,
            **small_params
        )
        
        # Compare results (allow small numerical differences)
        seq_result = exp_sequential.result
        par_result = exp_parallel.result
        
        # Check magnitude consistency (main physics result)
        np.testing.assert_allclose(
            par_result['Magnitude'], seq_result['Magnitude'],
            rtol=1e-10, atol=1e-12,
            err_msg="Parallel and sequential magnitude results differ"
        )
        
        # Check phase consistency  
        np.testing.assert_allclose(
            par_result['Phase'], seq_result['Phase'],
            rtol=1e-10, atol=1e-12,
            err_msg="Parallel and sequential phase results differ"
        )
        
        # Performance comparison
        seq_time = exp_sequential.performance_metrics['execution_time']
        par_time = exp_parallel.performance_metrics['execution_time']
        
        print(f"Timing comparison - Sequential: {seq_time:.3f}s, Parallel: {par_time:.3f}s")
        
        # For small grids, parallel might be slower due to overhead
        # Just ensure both completed successfully
        assert seq_time > 0
        assert par_time > 0

    def test_parallel_performance_scaling(self, mock_qubit, medium_params):
        """Test parallel performance with different worker counts."""
        max_workers = min(multiprocessing.cpu_count(), 4)
        worker_counts = [1, 2, max_workers] if max_workers >= 2 else [1, 2]
        
        results = {}
        timings = {}
        
        for num_workers in worker_counts:
            exp = QubitSpectroscopyAmplitudeFrequency(
                dut_qubit=mock_qubit,
                use_parallel=(num_workers > 1),
                num_workers=num_workers,
                **medium_params
            )
            
            results[num_workers] = exp.result
            timings[num_workers] = exp.performance_metrics['execution_time']
            
            # Validate each result
            assert exp.result['Magnitude'].shape[0] > 0
            assert exp.result['Magnitude'].shape[1] > 0
            assert np.all(np.isfinite(exp.result['Magnitude']))
        
        # Compare results for consistency
        reference_result = results[worker_counts[0]]
        for num_workers in worker_counts[1:]:
            np.testing.assert_allclose(
                results[num_workers]['Magnitude'], 
                reference_result['Magnitude'],
                rtol=1e-10, atol=1e-12,
                err_msg=f"Results differ with {num_workers} workers"
            )
        
        # Report performance scaling
        print(f"Performance scaling results:")
        for num_workers in worker_counts:
            time_val = timings[num_workers]
            speedup = timings[worker_counts[0]] / time_val if num_workers > worker_counts[0] else 1.0
            print(f"  {num_workers} worker{'s' if num_workers != 1 else ''}: {time_val:.3f}s (speedup: {speedup:.2f}x)")

    def test_parallel_parameter_validation(self, mock_qubit):
        """Test parameter validation in parallel mode."""
        base_params = {
            'start': 5000.0,
            'stop': 5010.0,
            'step': 5.0,
            'qubit_amp_start': 0.01,
            'qubit_amp_stop': 0.02,
            'qubit_amp_step': 0.01,
            'num_avs': 100,
            'disable_noise': True
        }
        
        # Test valid parallel parameters
        exp = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=mock_qubit,
            use_parallel=True,
            num_workers=2,
            **base_params
        )
        assert exp.performance_metrics['parallel_enabled'] is True
        assert exp.performance_metrics['num_workers'] == 2
        
        # Test auto-detection of workers (num_workers=None)
        exp = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=mock_qubit,
            use_parallel=True,
            num_workers=None,  # Should auto-detect
            **base_params
        )
        assert exp.performance_metrics['parallel_enabled'] is True
        assert exp.performance_metrics['num_workers'] > 0

    def test_parallel_with_different_grid_sizes(self, mock_qubit):
        """Test parallel processing with various parameter grid sizes."""
        grid_configs = [
            # Small grid
            {'freq_points': 3, 'amp_points': 2},
            # Medium grid  
            {'freq_points': 10, 'amp_points': 5},
            # Larger grid
            {'freq_points': 15, 'amp_points': 8},
        ]
        
        for config in grid_configs:
            freq_step = 20.0 / (config['freq_points'] - 1)
            amp_step = 0.02 / (config['amp_points'] - 1) if config['amp_points'] > 1 else 0.01
            
            params = {
                'start': 4990.0,
                'stop': 5010.0,
                'step': freq_step,
                'qubit_amp_start': 0.01,
                'qubit_amp_stop': 0.03,
                'qubit_amp_step': amp_step,
                'num_avs': 100,
                'disable_noise': True
            }
            
            exp = QubitSpectroscopyAmplitudeFrequency(
                dut_qubit=mock_qubit,
                use_parallel=True,
                num_workers=2,
                **params
            )
            
            # Validate grid dimensions
            expected_shape = (config['amp_points'], config['freq_points'])
            actual_shape = exp.result['Magnitude'].shape
            
            # Allow for small differences due to floating point step calculations
            assert abs(actual_shape[0] - expected_shape[0]) <= 1
            assert abs(actual_shape[1] - expected_shape[1]) <= 1
            
            # Validate result quality
            assert np.all(np.isfinite(exp.result['Magnitude']))
            assert exp.performance_metrics['total_points'] > 0

    def test_parallel_error_recovery_integration(self, mock_qubit, small_params):
        """Test error recovery in full experiment integration."""
        with patch('leeq.theory.simulation.numpy.cw_spectroscopy._simulate_point_worker') as mock_worker:
            # Make some workers fail, others succeed
            call_count = 0
            def intermittent_worker(*args):
                nonlocal call_count
                call_count += 1
                if call_count % 3 == 0:  # Every 3rd call fails
                    raise RuntimeError("Simulated worker failure")
                return 0.5 + 0.3j
            
            mock_worker.side_effect = intermittent_worker
            
            # Should complete despite worker failures
            exp = QubitSpectroscopyAmplitudeFrequency(
                dut_qubit=mock_qubit,
                use_parallel=True,
                num_workers=2,
                **small_params
            )
            
            # Experiment should complete successfully
            assert exp.result is not None
            assert 'Magnitude' in exp.result
            assert np.all(np.isfinite(exp.result['Magnitude']))

    def test_parallel_memory_efficiency_integration(self, mock_qubit):
        """Test memory efficiency in integrated parallel experiment."""
        # Create a reasonably large parameter grid
        params = {
            'start': 4950.0,
            'stop': 5050.0,
            'step': 2.0,  # 51 frequency points
            'qubit_amp_start': 0.005,
            'qubit_amp_stop': 0.075,
            'qubit_amp_step': 0.005,  # 15 amplitude points
            'num_avs': 500,
            'disable_noise': True
        }
        
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # Run parallel experiment
        exp = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=mock_qubit,
            use_parallel=True,
            num_workers=4,
            **params
        )
        
        memory_after = process.memory_info().rss / (1024**2)  # MB
        memory_increase = memory_after - memory_before
        
        # Validate results
        total_points = exp.performance_metrics['total_points']
        assert total_points > 500  # Reasonably large grid
        assert np.all(np.isfinite(exp.result['Magnitude']))
        
        # Memory usage should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
        print(f"Memory usage for {total_points} points: {memory_increase:.1f} MB")

    def test_parallel_deterministic_results(self, mock_qubit, small_params):
        """Test that parallel experiments produce deterministic results."""
        results = []
        
        # Run same experiment multiple times
        for _ in range(3):
            exp = QubitSpectroscopyAmplitudeFrequency(
                dut_qubit=mock_qubit,
                use_parallel=True,
                num_workers=2,
                **small_params
            )
            results.append(exp.result['Magnitude'])
        
        # All results should be identical (deterministic)
        reference = results[0]
        for result in results[1:]:
            np.testing.assert_array_equal(
                result, reference,
                err_msg="Parallel experiment results are not deterministic"
            )

    def test_parallel_vs_sequential_performance_metrics(self, mock_qubit, medium_params):
        """Test that performance metrics are correctly collected and compared."""
        # Run sequential
        exp_seq = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=mock_qubit,
            use_parallel=False,
            **medium_params
        )
        
        # Run parallel
        exp_par = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=mock_qubit,
            use_parallel=True,
            num_workers=4,
            **medium_params
        )
        
        # Compare metrics
        seq_metrics = exp_seq.performance_metrics
        par_metrics = exp_par.performance_metrics
        
        # Basic metric validation
        assert seq_metrics['parallel_enabled'] is False
        assert par_metrics['parallel_enabled'] is True
        assert seq_metrics['num_workers'] == 1
        assert par_metrics['num_workers'] == 4
        
        # Grid size should be identical
        assert seq_metrics['grid_size'] == par_metrics['grid_size']
        assert seq_metrics['total_points'] == par_metrics['total_points']
        
        # Both should have execution times
        assert seq_metrics['execution_time'] > 0
        assert par_metrics['execution_time'] > 0
        
        # Memory usage should be recorded
        assert 'memory_used_mb' in seq_metrics
        assert 'memory_used_mb' in par_metrics
        
        # Calculate and report speedup
        if par_metrics['execution_time'] < seq_metrics['execution_time']:
            speedup = seq_metrics['execution_time'] / par_metrics['execution_time']
            print(f"Achieved speedup: {speedup:.2f}x with {par_metrics['num_workers']} workers")
        else:
            print("No speedup observed (possibly due to small grid or overhead)")


class TestParallelIntegrationEdgeCases:
    """Test edge cases in parallel integration."""
    
    @pytest.fixture
    def mock_qubit(self):
        """Create mock qubit for testing."""
        mock_qubit = MagicMock()
        
        # Mock measurement primitive
        mock_mp = MagicMock()
        mock_mp.clone.return_value = mock_mp
        mock_mp.update_pulse_args.return_value = None
        mock_mp.set_transform_function.return_value = None
        
        # Add __name__ attribute for the sweeper
        mock_mp.update_freq = MagicMock()
        mock_mp.update_freq.__name__ = 'update_freq'
        
        mock_qubit.get_default_measurement_prim_int.return_value = mock_mp
        
        # Mock measurement primitive list
        mock_mplist = MagicMock()
        mock_qubit.get_default_measurement_prim_intlist.return_value = mock_mplist
        
        return mock_qubit
    
    def test_single_point_parallel_experiment(self, mock_qubit):
        """Test parallel experiment with single parameter point."""
        params = {
            'start': 5000.0,
            'stop': 5000.0,  # Single frequency point
            'step': 1.0,
            'qubit_amp_start': 0.02,
            'qubit_amp_stop': 0.02,  # Single amplitude point
            'qubit_amp_step': 0.01,
            'num_avs': 100,
            'disable_noise': True
        }
        
        exp = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=mock_qubit,
            use_parallel=True,
            **params
        )
        
        # Should handle single point gracefully
        assert exp.result['Magnitude'].shape == (1, 1)
        assert np.isfinite(exp.result['Magnitude'][0, 0])

    def test_parallel_experiment_with_noise_disabled(self, mock_qubit):
        """Test parallel experiment behavior with noise disabled."""
        params = {
            'start': 4995.0,
            'stop': 5005.0,
            'step': 5.0,
            'qubit_amp_start': 0.015,
            'qubit_amp_stop': 0.025,
            'qubit_amp_step': 0.005,
            'num_avs': 200,
            'disable_noise': True  # Key parameter
        }
        
        exp = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=mock_qubit,
            use_parallel=True,
            num_workers=2,
            **params
        )
        
        # With noise disabled, results should be perfectly deterministic
        # Run again and compare
        exp2 = QubitSpectroscopyAmplitudeFrequency(
            dut_qubit=mock_qubit,
            use_parallel=True,
            num_workers=2,
            **params
        )
        
        # Results should be identical
        np.testing.assert_array_equal(
            exp.result['Magnitude'],
            exp2.result['Magnitude'],
            err_msg="Results with disabled noise should be identical"
        )

    def test_parallel_experiment_fallback_behavior(self, mock_qubit):
        """Test experiment behavior when parallel processing fails."""
        params = {
            'start': 4990.0,
            'stop': 5010.0,
            'step': 10.0,
            'qubit_amp_start': 0.01,
            'qubit_amp_stop': 0.03,
            'qubit_amp_step': 0.01,
            'num_avs': 100,
            'disable_noise': True
        }
        
        with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
            # Force ProcessPoolExecutor to fail
            mock_executor.side_effect = RuntimeError("Forced parallel failure")
            
            # Should fall back to sequential processing
            exp = QubitSpectroscopyAmplitudeFrequency(
                dut_qubit=mock_qubit,
                use_parallel=True,  # Request parallel
                **params
            )
            
            # Experiment should complete via fallback
            assert exp.result is not None
            assert 'Magnitude' in exp.result
            assert exp.result['Magnitude'].shape == (3, 3)  # Expected grid size


if __name__ == "__main__":
    # Run specific test for quick validation
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test for development
        pytest.main([__file__ + "::TestParallelIntegration::test_parallel_experiment_basic_execution", "-v"])
    else:
        # Run all tests
        pytest.main([__file__, "-v"])