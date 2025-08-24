"""
Backward compatibility tests for Kerr nonlinearity integration.

This module tests that existing dispersive readout simulations continue to work
unchanged after adding Kerr nonlinearity features. The tests verify that:
1. Default behavior is identical to original implementation
2. Existing API calls work without modification
3. No performance regression occurs
4. All existing method signatures remain functional
"""

import pytest
import numpy as np
import time

from leeq.theory.simulation.numpy.dispersive_readout.simulator import (
    DispersiveReadoutSimulatorSyntheticData
)
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon


class TestDefaultBehaviorUnchanged:
    """Test that default behavior matches original implementation."""
    
    def test_default_constructor_unchanged(self):
        """Test that default constructor behavior is identical to original."""
        # Default constructor should not enable Kerr effects
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=1.0
        )
        
        # Verify Kerr is disabled by default
        assert sim.use_kerr_nonlinearity is False
        assert sim.kerr_coefficient is None
        assert not hasattr(sim, 'kerr_calculator') or sim.kerr_calculator is None
        assert not hasattr(sim, 'power_sweep_manager') or sim.power_sweep_manager is None
        
        # Should use default legacy chi values
        assert sim.chis is not None
        assert len(sim.chis) == 4  # Default number of levels
        assert sim.chis[0] == 0  # Ground state chi is always 0
    
    def test_legacy_chi_array_behavior(self):
        """Test that providing explicit chi array works exactly as before."""
        legacy_chis = [0, -0.25, -0.5, -0.75]
        
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=0.5,
            chis=legacy_chis
        )
        
        # Should use the provided chi values exactly
        assert sim.chis == legacy_chis
        assert sim.use_kerr_nonlinearity is False
        
        # Simulation should work as before
        trace = sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
        assert isinstance(trace, np.ndarray)
        assert len(trace) > 0
        assert np.iscomplexobj(trace)
    
    def test_physics_model_backward_compatibility(self):
        """Test that existing physics model usage is unaffected."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=0.5,
            use_physics_model=True,
            anharmonicity=-250,
            coupling_strength=50,
            f_q=5500
        )
        
        # Should enable physics model but not Kerr effects
        assert sim.use_physics_model is True
        assert sim.use_kerr_nonlinearity is False
        assert sim.kerr_coefficient is None
        
        # Should calculate chi values using physics
        assert len(sim.chis) == 4  # Default num_levels
        assert sim.chis[0] == 0  # Ground state reference
        
        # Simulation should work
        trace = sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
        assert isinstance(trace, np.ndarray)


class TestAPICompatibility:
    """Test that all existing API method signatures work unchanged."""
    
    def test_simulate_trace_signature(self):
        """Test _simulate_trace method signature remains compatible."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=1.0,
            chis=[0, -0.25, -0.5, -0.75]
        )
        
        # Test all existing parameter combinations
        # Basic call
        trace1 = sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
        assert isinstance(trace1, np.ndarray)
        
        # With noise
        trace2 = sim._simulate_trace(state=2, f_prob=7000, noise_std=0.1)
        assert isinstance(trace2, np.ndarray)
        
        # With state sequence
        state_seq = [(0, 2), (2, 1), (4, 0)]
        trace3 = sim._simulate_trace(state=state_seq, f_prob=7000, noise_std=0)
        assert isinstance(trace3, np.ndarray)
        
        # With return states over time
        result = sim._simulate_trace(
            state=1, f_prob=7000, noise_std=0, return_states_over_time=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        trace, states = result
        assert isinstance(trace, np.ndarray)
        assert isinstance(states, np.ndarray)
    
    def test_simulate_trace_with_decay_signature(self):
        """Test simulate_trace_with_decay method signature unchanged."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=0.5,
            chis=[0, -0.25, -0.5],
            t1s=[100.0, 50.0, 33.3],
            width=10
        )
        
        # Basic decay simulation - needs explicit noise_std=0 
        trace1 = sim.simulate_trace_with_decay(state=1, f_prob=7000, noise_std=0)
        assert isinstance(trace1, np.ndarray)
        
        # With noise
        trace2 = sim.simulate_trace_with_decay(state=2, f_prob=7000, noise_std=0.1)
        assert isinstance(trace2, np.ndarray)
        
        # With states over time
        result = sim.simulate_trace_with_decay(
            state=1, f_prob=7000, noise_std=0.05, return_states_over_time=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_virtual_transmon_get_resonator_response(self):
        """Test VirtualTransmon.get_resonator_response works without power parameter."""
        vt = VirtualTransmon(
            name="test_transmon",
            readout_frequency=7000,
            readout_linewith=1.0,  # Note: typo in original codebase
            qubit_frequency=5000,
            anharmonicity=-250,
            t1=100.0,
            t2=50.0,
            coupling_strength=50
        )
        
        # Test original signature without power parameter
        response1 = vt.get_resonator_response(f=7000)
        assert isinstance(response1, np.ndarray)
        
        # Test with amp and baseline (original parameters)
        response2 = vt.get_resonator_response(f=7000, amp=1.5, baseline=0.1)
        assert isinstance(response2, np.ndarray)
        
        # Verify Kerr is disabled by default
        assert vt.use_kerr_nonlinearity is False
        assert vt.kerr_coefficient is None
    
    def test_factory_method_compatibility(self):
        """Test that factory method still works with original parameters."""
        sim = DispersiveReadoutSimulatorSyntheticData.from_physics_model(
            f_r=7000,
            f_q=5000,
            anharmonicity=-200,
            coupling_strength=100,
            kappa=1.0,
            num_levels=3
        )
        
        # Should work with original behavior
        assert sim.use_physics_model is True
        assert sim.use_kerr_nonlinearity is False  # Kerr not enabled by default
        assert len(sim.chis) == 3
        
        # Should simulate correctly
        trace = sim._simulate_trace(state=2, f_prob=7000, noise_std=0)
        assert isinstance(trace, np.ndarray)


class TestFeatureFlagIsolation:
    """Test that feature flag properly isolates new functionality."""
    
    def test_kerr_flag_default_false(self):
        """Test that use_kerr_nonlinearity defaults to False."""
        # Test all constructor paths
        sim1 = DispersiveReadoutSimulatorSyntheticData(f_r=7000, kappa=1.0)
        assert sim1.use_kerr_nonlinearity is False
        
        sim2 = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000, kappa=1.0, chis=[0, -0.25]
        )
        assert sim2.use_kerr_nonlinearity is False
        
        sim3 = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000, kappa=1.0, use_physics_model=True,
            anharmonicity=-250, coupling_strength=50
        )
        assert sim3.use_kerr_nonlinearity is False
    
    def test_kerr_flag_explicit_false(self):
        """Test that explicitly setting use_kerr_nonlinearity=False works."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=1.0,
            use_kerr_nonlinearity=False  # Explicit
        )
        
        assert sim.use_kerr_nonlinearity is False
        assert sim.kerr_coefficient is None
        assert not hasattr(sim, 'kerr_calculator') or sim.kerr_calculator is None
        
        # Should work normally
        trace = sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
        assert isinstance(trace, np.ndarray)
    
    def test_kerr_coefficient_ignored_when_flag_false(self):
        """Test that kerr_coefficient is ignored when use_kerr_nonlinearity=False."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=1.0,
            use_kerr_nonlinearity=False,
            kerr_coefficient=-0.01  # Should be ignored
        )
        
        assert sim.use_kerr_nonlinearity is False
        assert sim.kerr_coefficient is None  # Should be None, not -0.01
        
        # Behavior should be identical to original
        trace = sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
        assert isinstance(trace, np.ndarray)
    
    def test_virtual_transmon_kerr_flag_isolation(self):
        """Test VirtualTransmon Kerr flag isolation."""
        vt = VirtualTransmon(
            name="test_transmon",
            readout_frequency=7000,
            readout_linewith=1.0,  # Note: typo in original codebase
            qubit_frequency=5000,
            anharmonicity=-250,
            t1=100.0,
            t2=50.0,
            coupling_strength=50
            # use_kerr_nonlinearity not specified, should default to False
        )
        
        assert vt.use_kerr_nonlinearity is False
        
        # Power parameter should be ignored when Kerr disabled
        response1 = vt.get_resonator_response(f=7000)
        response2 = vt.get_resonator_response(f=7000, power=1.0)
        
        # Responses should be identical when Kerr disabled
        np.testing.assert_array_almost_equal(response1, response2, decimal=10)


class TestPerformanceRegression:
    """Test that no significant performance degradation occurred."""
    
    @pytest.fixture
    def benchmark_sim(self):
        """Create simulator for performance benchmarking."""
        return DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=1.0,
            chis=[0, -0.25, -0.5, -0.75],
            width=10,
            sampling_rate=500  # Reduced for faster testing
        )
    
    def test_single_trace_performance(self, benchmark_sim):
        """Test that single trace simulation remains fast."""
        # Warm up
        _ = benchmark_sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            _ = benchmark_sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        # Should complete in less than 50ms per trace (generous threshold)
        assert avg_time < 0.05, f"Single trace took {avg_time:.3f}s, expected <0.05s"
    
    def test_basic_simulation_performance(self, benchmark_sim):
        """Test that basic simulation performance is maintained (non-decay)."""
        # Focus on non-decay simulation which is more stable
        start_time = time.time()
        for _ in range(20):
            _ = benchmark_sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 20
        # Should complete in less than 50ms per basic simulation
        assert avg_time < 0.05, f"Basic simulation took {avg_time:.3f}s, expected <0.05s"
    
    def test_frequency_sweep_performance(self, benchmark_sim):
        """Test frequency sweep performance."""
        frequencies = np.linspace(6990, 7010, 21)  # 21 points for testing
        
        start_time = time.time()
        traces = []
        for f in frequencies:
            trace = benchmark_sim._simulate_trace(state=1, f_prob=f, noise_std=0)
            traces.append(trace)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_point = total_time / len(frequencies)
        
        # Should complete in less than 25ms per frequency point
        assert avg_time_per_point < 0.025, (
            f"Frequency sweep took {avg_time_per_point:.3f}s per point, expected <0.025s"
        )


class TestNumericalConsistency:
    """Test that numerical results are consistent with original implementation."""
    
    def test_deterministic_output(self):
        """Test that output is deterministic when noise is disabled."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=1.0,
            chis=[0, -0.25, -0.5, -0.75],
            amp=1.0,
            baseline=0.1
        )
        
        # Generate same trace multiple times
        traces = []
        for _ in range(5):
            trace = sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
            traces.append(trace)
        
        # All traces should be identical (no random seed variation)
        for i in range(1, len(traces)):
            np.testing.assert_array_almost_equal(
                traces[0], traces[i], decimal=10,
                err_msg="Traces should be identical when noise_std=0"
            )
    
    def test_frequency_response_shape(self):
        """Test that frequency response maintains expected shape."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=1.0,
            chis=[0, -0.5]  # Simple two-level system
        )
        
        # Test frequencies around resonance
        frequencies = np.linspace(6995, 7005, 11)
        
        # Get responses for both states
        responses_0 = []
        responses_1 = []
        for f in frequencies:
            resp_0 = sim._simulate_trace(state=0, f_prob=f, noise_std=0)
            resp_1 = sim._simulate_trace(state=1, f_prob=f, noise_std=0)
            # Take magnitude for analysis
            responses_0.append(np.abs(resp_0).mean())
            responses_1.append(np.abs(resp_1).mean())
        
        responses_0 = np.array(responses_0)
        responses_1 = np.array(responses_1)
        
        # Verify basic properties
        assert len(responses_0) == len(frequencies)
        assert len(responses_1) == len(frequencies)
        
        # State 1 should have different response than state 0
        # (exact values depend on implementation details)
        assert not np.allclose(responses_0, responses_1, rtol=0.01)
    
    def test_state_sequence_consistency(self):
        """Test that state sequence simulation works consistently."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=1.0,
            chis=[0, -0.25, -0.5],
            width=5,  # Shorter pulse for testing
            sampling_rate=200
        )
        
        # Test state sequence
        state_sequence = [(0, 0), (1, 1), (3, 2), (4, 1)]  # (time, state)
        
        trace = sim._simulate_trace(
            state=state_sequence, f_prob=7000, noise_std=0
        )
        
        assert isinstance(trace, np.ndarray)
        assert np.iscomplexobj(trace)
        assert len(trace) > 0
        
        # Should produce consistent results
        trace2 = sim._simulate_trace(
            state=state_sequence, f_prob=7000, noise_std=0
        )
        np.testing.assert_array_almost_equal(trace, trace2, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])