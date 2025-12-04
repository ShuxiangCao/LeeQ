"""
Test backward compatibility for dispersive readout simulator integration.

This module tests that the existing simulator API continues to work after
integrating the physics-based chi shift calculation.
"""

import pytest
import numpy as np

from leeq.theory.simulation.numpy.dispersive_readout.simulator import (
    DispersiveReadoutSimulatorSyntheticData
)


class TestBackwardCompatibility:
    """Test backward compatibility of the simulator."""
    
    def test_legacy_chi_array_still_works(self):
        """Test that providing explicit chi array still works as before."""
        # Legacy usage with explicit chi array
        legacy_chis = [0, -0.25, -0.5, -0.75]
        
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=0.5,
            chis=legacy_chis,
            use_physics_model=False  # Explicitly disable physics model
        )
        
        # Should use the provided chi values exactly
        assert sim.chis == legacy_chis
        assert not sim.use_physics_model
        
        # Should be able to simulate traces
        trace = sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
        assert isinstance(trace, np.ndarray)
        assert len(trace) > 0
    
    def test_default_behavior_unchanged(self):
        """Test that default behavior without physics model is unchanged."""
        # Create simulator without explicit chi values and physics model off
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=6000,
            kappa=1.0,
            use_physics_model=False
        )
        
        # Should use default legacy chi values
        expected_default = [0, -0.25, -0.5, -0.75]
        assert sim.chis == expected_default
        
        # Test that simulation works
        trace = sim._simulate_trace(state=2, f_prob=6000, noise_std=0.1)
        assert isinstance(trace, np.ndarray)
        assert len(trace) > 0
    
    def test_physics_model_can_be_enabled(self):
        """Test that physics model can be enabled with new parameters."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=0.5,
            use_physics_model=True,
            anharmonicity=-250,
            coupling_strength=50,
            f_q=5500
        )
        
        # Should use physics model
        assert sim.use_physics_model
        assert sim.anharmonicity == -250
        assert sim.coupling_strength == 50
        assert sim.f_q == 5500
        
        # Chi values should be calculated by physics model
        assert len(sim.chis) == 4  # Default num_levels
        assert sim.chis[0] == 0  # Ground state chi is always 0
        
        # Should be able to simulate
        trace = sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
        assert isinstance(trace, np.ndarray)
    
    def test_factory_method_works(self):
        """Test the new factory method for physics-based initialization."""
        sim = DispersiveReadoutSimulatorSyntheticData.from_physics_model(
            f_r=7000,
            f_q=5000,
            anharmonicity=-200,
            coupling_strength=100,
            kappa=1.0,
            num_levels=3
        )
        
        assert sim.use_physics_model
        assert sim.f_q == 5000
        assert sim.anharmonicity == -200
        assert sim.coupling_strength == 100
        assert len(sim.chis) == 3
        
        # Should simulate correctly
        trace = sim._simulate_trace(state=2, f_prob=7000, noise_std=0)
        assert isinstance(trace, np.ndarray)
    
    def test_explicit_chi_overrides_physics(self):
        """Test that explicit chi values override physics model."""
        custom_chis = [0, -1.0, -2.0, -3.0]
        
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=0.5,
            chis=custom_chis,  # Explicit chi values
            use_physics_model=True,  # Physics model enabled but should be ignored
            anharmonicity=-250,
            coupling_strength=50
        )
        
        # Should use explicit chi values, not physics calculation
        assert sim.chis == custom_chis
        assert sim.use_physics_model  # Flag is set but not used
    
    def test_default_f_q_calculation(self):
        """Test default f_q calculation when not provided."""
        f_r = 8000
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=f_r,
            kappa=0.5,
            use_physics_model=True,
            anharmonicity=-250,
            coupling_strength=50
            # f_q not provided
        )
        
        # Should default to f_r - 1000
        expected_f_q = f_r - 1000
        assert sim.f_q == expected_f_q
    
    def test_num_levels_extension(self):
        """Test that chi array is extended if num_levels is larger."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=0.5,
            chis=[0, -0.25],  # Only 2 levels provided
            num_levels=5,  # But need 5 levels
            use_physics_model=False
        )
        
        # Should extend chi array to 5 levels
        assert len(sim.chis) == 5
        assert sim.chis[0] == 0
        assert sim.chis[1] == -0.25
        # Should continue with linear scaling
        assert len(sim.chis) == 5
    
    def test_parameter_storage(self):
        """Test that all parameters are properly stored."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=0.5,
            use_physics_model=True,
            anharmonicity=-300,
            coupling_strength=75,
            f_q=5200,
            num_levels=6
        )
        
        # Check all parameters are stored
        assert sim.f_r == 7000
        assert sim.kappa == 0.5
        assert sim.use_physics_model
        assert sim.anharmonicity == -300
        assert sim.coupling_strength == 75
        assert sim.f_q == 5200
        assert sim.num_levels == 6
        assert len(sim.chis) == 6


class TestSimulationConsistency:
    """Test that simulations produce consistent results."""
    
    def test_simulation_output_format(self):
        """Test that simulation output format is unchanged."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=0.5,
            chis=[0, -0.25, -0.5, -0.75],
            width=5,
            sampling_rate=100
        )
        
        # Test basic simulation
        trace = sim._simulate_trace(state=1, f_prob=7000, noise_std=0)
        assert isinstance(trace, np.ndarray)
        assert np.iscomplexobj(trace)  # Should be complex I/Q data
        
        # Test with state sequence
        state_sequence = [(0, 2), (2, 1), (4, 0)]
        trace_seq = sim._simulate_trace(state=state_sequence, f_prob=7000, noise_std=0)
        assert isinstance(trace_seq, np.ndarray)
        assert np.iscomplexobj(trace_seq)
        
        # Test with states over time return
        trace_with_states = sim._simulate_trace(
            state=1, f_prob=7000, noise_std=0, return_states_over_time=True
        )
        assert isinstance(trace_with_states, tuple)
        assert len(trace_with_states) == 2
        trace, states = trace_with_states
        assert isinstance(trace, np.ndarray)
        assert isinstance(states, np.ndarray)
    
    def test_decay_simulation_compatibility(self):
        """Test that decay simulation still works."""
        sim = DispersiveReadoutSimulatorSyntheticData(
            f_r=7000,
            kappa=0.5,
            chis=[0, -0.25, -0.5, -0.75],
            t1s=[100.0, 50.0, 33.3],
            width=10
        )
        
        # Test decay simulation
        trace = sim.simulate_trace_with_decay(state=2, f_prob=7000, noise_std=0.1)
        assert isinstance(trace, np.ndarray)
        assert np.iscomplexobj(trace)
        
        # Test with states over time
        result = sim.simulate_trace_with_decay(
            state=3, f_prob=7000, noise_std=0.1, return_states_over_time=True
        )
        assert isinstance(result, tuple)
        trace, states = result
        assert isinstance(trace, np.ndarray)
        assert isinstance(states, np.ndarray)


# Script-style execution converted to proper pytest discovery
# Tests will be run by pytest discovery, no manual execution needed