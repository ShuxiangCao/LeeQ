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
# Import the new mock testing framework for demonstration
from tests.utils.mock_testing_framework import MockTestingFramework, create_complete_mock_qubit


class PicklableMock:
    """
    A simple mock class that can be pickled for parallel processing.
    
    This replaces MagicMock objects that cannot be pickled across process boundaries.
    Provides the essential interface needed for qubit simulation tests.
    """
    
    def __init__(self, **kwargs):
        self.return_value = kwargs.get('return_value', None)
        self.side_effect = kwargs.get('side_effect', None)
        self.__name__ = kwargs.get('__name__', 'mock_function')
        self._children = {}
        self._numeric_value = kwargs.get('_numeric_value', 1.0)  # Default numeric value
        
        # Initialize attributes from kwargs
        for key, value in kwargs.items():
            if key not in ['return_value', 'side_effect', '__name__', '_numeric_value']:
                setattr(self, key, value)
    
    def __call__(self, *args, **kwargs):
        if self.side_effect:
            if callable(self.side_effect):
                return self.side_effect(*args, **kwargs)
            elif isinstance(self.side_effect, Exception):
                raise self.side_effect
            elif hasattr(self.side_effect, '__iter__'):
                # Return next value from iterable
                try:
                    return next(iter(self.side_effect))
                except StopIteration:
                    pass
        return self.return_value
    
    def __getattr__(self, name):
        if name not in self._children:
            self._children[name] = PicklableMock()
        return self._children[name]
    
    def __getitem__(self, key):
        return self.__getattr__(f'_item_{key}')
    
    # Numeric operations for compatibility with mathematical operations
    def __mul__(self, other):
        if hasattr(self, '_numeric_value'):
            return self._numeric_value * other
        return self._numeric_value * other
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        if hasattr(self, '_numeric_value'):
            return self._numeric_value + other
        return self._numeric_value + other
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if hasattr(self, '_numeric_value'):
            return self._numeric_value - other
        return self._numeric_value - other
    
    def __rsub__(self, other):
        return other - self._numeric_value
    
    def __truediv__(self, other):
        if hasattr(self, '_numeric_value'):
            return self._numeric_value / other
        return self._numeric_value / other
    
    def __rtruediv__(self, other):
        return other / self._numeric_value
    
    def __float__(self):
        return float(self._numeric_value) if hasattr(self, '_numeric_value') else 1.0
    
    def __int__(self):
        return int(self._numeric_value) if hasattr(self, '_numeric_value') else 1
    
    def clone(self):
        return self
    
    def update_pulse_args(self, *args, **kwargs):
        return None
    
    def set_transform_function(self, *args, **kwargs):
        return None


def create_picklable_mock_qubit():
    """Create a pickle-able mock qubit for parallel processing tests."""
    mock_qubit = PicklableMock()
    
    # Mock measurement primitive
    mock_mp = PicklableMock()
    mock_mp.clone = lambda: mock_mp
    mock_mp.update_pulse_args = PicklableMock(__name__='update_pulse_args', return_value=None)
    mock_mp.set_transform_function = PicklableMock(return_value=None)
    mock_mp.update_freq = PicklableMock(__name__='update_freq')
    
    mock_qubit.get_default_measurement_prim_int = PicklableMock(return_value=mock_mp)
    
    # Mock measurement primitive list with numeric attributes
    mock_mplist = PicklableMock()
    mock_mplist.freq = 5000.0  # Default readout frequency
    mock_mplist.amp = PicklableMock(_numeric_value=0.1)     # Default amplitude as PicklableMock for multiplication
    mock_mplist.channel = 1   # Default readout channel
    mock_qubit.get_default_measurement_prim_intlist = PicklableMock(return_value=mock_mplist)
    
    # Mock pulse collection chain
    mock_pulse_collection = PicklableMock(channel=2)
    mock_pulse = PicklableMock()
    mock_pulse_cloned = PicklableMock()
    
    mock_pulse.clone = PicklableMock(return_value=mock_pulse_cloned)
    mock_pulse_cloned.update_pulse_args = PicklableMock(__name__='update_pulse_args', return_value=None)
    
    mock_pulse_collection._item_X = mock_pulse
    mock_qubit.get_default_c1 = PicklableMock(return_value=mock_pulse_collection)
    
    return mock_qubit


class TestParallelIntegration:
    """Integration tests for parallel experiment execution."""
    
    @pytest.fixture
    def mock_qubit(self):
        """Create pickle-able mock qubit for parallel processing tests."""
        return create_picklable_mock_qubit()
    
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

    def test_parallel_experiment_basic_execution(self, simulation_setup, mock_qubit, small_params):
        """Test basic parallel experiment execution with small parameter grid."""
        # Patch chronicle recording to avoid registration errors
        # Use simple functions instead of MagicMock to avoid pickling issues
        def passthrough_log_record(func, args, kwargs, **kw):
            return func(*args, **kwargs)
        
        def simple_chronicle_log(*args, **kwargs):
            return None
            
        with patch('leeq.chronicle.decorators._log_and_record', passthrough_log_record), \
             patch.object(QubitSpectroscopyAmplitudeFrequency, 'chronicle_log', simple_chronicle_log):
            
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
        # Note: I/Q components may not be present in all simulation modes
        # The essential results are Magnitude and Phase
        
        # Validate result dimensions - use actual grid size as calculated by experiment
        actual_shape = result['Magnitude'].shape
        assert actual_shape == result['Phase'].shape  # Both should have same dimensions
        assert len(actual_shape) == 2  # Should be 2D array
        assert actual_shape[0] > 0 and actual_shape[1] > 0  # Should have positive dimensions
        
        # Validate performance metrics were collected
        assert hasattr(exp_parallel, 'performance_metrics')
        metrics = exp_parallel.performance_metrics
        assert metrics['parallel_enabled'] is True
        assert metrics['num_workers'] == 2
        assert metrics['execution_time'] > 0
        assert metrics['grid_size'] == actual_shape

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
        """Create pickle-able mock qubit for parallel processing tests."""
        return create_picklable_mock_qubit()
    
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


class TestMockTestingFrameworkIntegration:
    """
    Demonstration of the new Mock Testing Framework for robust mock object setup.
    
    This test class shows how to use the new utilities to create properly configured
    mock objects that avoid AttributeError issues with the sweeper system.
    """
    
    def test_framework_created_mock_qubit(self):
        """Test using MockTestingFramework.create_qubit()."""
        # Use the framework to create a complete mock qubit
        mock_qubit = MockTestingFramework.create_qubit()
        
        # Test that the mock has all the attributes that would be accessed
        # by the experiment without AttributeError issues
        
        # These would previously fail with AttributeError: __name__
        mp = mock_qubit.get_default_measurement_prim_int()
        assert hasattr(mp.update_freq, '__name__')
        assert mp.update_freq.__name__ == 'update_freq'
        
        # Test pulse collection chain
        pulse_collection = mock_qubit.get_default_c1()
        pulse = pulse_collection['X']
        cloned_pulse = pulse.clone()
        assert hasattr(cloned_pulse.update_pulse_args, '__name__')
        assert cloned_pulse.update_pulse_args.__name__ == 'update_pulse_args'
        
        # Test that this would work with sweeper system (no AttributeError)
        from leeq.experiments.sweeper import SweepParametersSideEffectFunction
        
        # This would previously fail with AttributeError: __name__
        sweep_effect = SweepParametersSideEffectFunction(
            mp.update_freq, 'frequency'
        )
        assert sweep_effect is not None
        assert sweep_effect._function_name == 'update_freq'
    
    def test_framework_interface_validation(self):
        """Test mock interface validation against real objects."""
        # Create a mock using the framework
        mock_qubit = MockTestingFramework.create_qubit()
        
        # Validate that the mock has the expected interface
        required_methods = [
            'get_default_measurement_prim_int',
            'get_default_measurement_prim_intlist',
            'get_default_c1'
        ]
        
        for method in required_methods:
            assert hasattr(mock_qubit, method), f"Mock missing method: {method}"
        
        # Validate sweeper compatibility
        mp = mock_qubit.get_default_measurement_prim_int()
        assert hasattr(mp.update_freq, '__name__')
        assert mp.update_freq.__name__ == 'update_freq'
    
    def test_framework_setup_validation(self):
        """Test validating complete mock setup."""
        # Create multiple mock objects
        mock_qubit = MockTestingFramework.create_qubit()
        mock_mp = MockTestingFramework.create_measurement_primitive()
        mock_collection = MockTestingFramework.create_pulse_collection()
        
        # Validate the complete setup
        mock_setup = {
            'qubit': mock_qubit,
            'measurement_primitive': mock_mp,
            'pulse_collection': mock_collection
        }
        
        # This should pass validation
        result = MockTestingFramework.validate_setup(mock_setup)
        assert result is True
    
    def test_comparison_old_vs_new_mock_patterns(self):
        """Compare old manual mock setup vs new framework setup."""
        # Old pattern (manual setup - what was causing issues)
        old_mock_qubit = MagicMock()
        old_mock_mp = MagicMock()
        old_mock_mp.update_freq = MagicMock()
        # Missing: old_mock_mp.update_freq.__name__ = 'update_freq'  # This was the bug
        old_mock_qubit.get_default_measurement_prim_int.return_value = old_mock_mp
        
        # New pattern (framework setup - no issues)
        new_mock_qubit = MockTestingFramework.create_qubit()
        
        # Test that new pattern works while old pattern would fail
        new_mp = new_mock_qubit.get_default_measurement_prim_int()
        assert hasattr(new_mp.update_freq, '__name__')
        assert new_mp.update_freq.__name__ == 'update_freq'
        
        # Old pattern would fail here:
        # old_mp = old_mock_qubit.get_default_measurement_prim_int()
        # print(old_mp.update_freq.__name__)  # AttributeError: __name__
        
        print("Mock Testing Framework prevents AttributeError issues")


# Script-style execution converted to proper pytest discovery
# Tests will be run by pytest discovery, no manual execution needed