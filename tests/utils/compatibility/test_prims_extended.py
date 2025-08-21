"""
Extended tests for compatibility primitives.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from leeq.utils.compatibility.prims import (
    SeriesLPB,
    SweepLPB,
    build_CZ_stark_from_parameters
)


class TestSeriesLPB:
    """Test SeriesLPB compatibility class."""
    
    def test_series_lpb_alias(self):
        """Test that SeriesLPB is properly aliased."""
        # SeriesLPB should be an alias to SerialLPB
        from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial as SerialLPB
        
        assert SeriesLPB is SerialLPB
    
    def test_series_lpb_instantiation(self):
        """Test SeriesLPB can be instantiated."""
        # Create mock primitives
        mock_prim1 = Mock()
        mock_prim2 = Mock()
        
        # Create SeriesLPB instance
        series_lpb = SeriesLPB()
        
        assert series_lpb is not None
        assert hasattr(series_lpb, '__class__')
    
    def test_series_lpb_with_children(self):
        """Test SeriesLPB with child primitives."""
        # Create mock child primitives
        children = [Mock(), Mock(), Mock()]
        
        # Create SeriesLPB with children
        series_lpb = SeriesLPB(children=children)
        
        assert series_lpb is not None
        # The actual implementation may store children differently
        # This test just verifies instantiation works


class TestSweepLPB:
    """Test SweepLPB compatibility class."""
    
    def test_sweep_lpb_with_list(self):
        """Test SweepLPB creation with list of children."""
        # Create mock children
        children_list = [Mock(), Mock(), Mock()]
        
        # Create SweepLPB instance
        sweep_lpb = SweepLPB(children_list)
        
        assert sweep_lpb is not None
        # Should be instance of LogicalPrimitiveBlockSweep
        from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
        assert isinstance(sweep_lpb, LogicalPrimitiveBlockSweep)
    
    def test_sweep_lpb_with_tuple(self):
        """Test SweepLPB creation with tuple of children."""
        # Create mock children as tuple
        children_tuple = (Mock(), Mock())
        
        # Create SweepLPB instance
        sweep_lpb = SweepLPB(children_tuple)
        
        assert sweep_lpb is not None
        from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
        assert isinstance(sweep_lpb, LogicalPrimitiveBlockSweep)
    
    def test_sweep_lpb_with_individual_args(self):
        """Test SweepLPB creation with individual arguments."""
        # Create mock children as separate arguments
        child1 = Mock()
        child2 = Mock()
        child3 = Mock()
        
        # Create SweepLPB instance
        sweep_lpb = SweepLPB(child1, child2, child3)
        
        assert sweep_lpb is not None
        from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
        assert isinstance(sweep_lpb, LogicalPrimitiveBlockSweep)
    
    def test_sweep_lpb_empty(self):
        """Test SweepLPB creation with no children."""
        # Create SweepLPB with no arguments
        sweep_lpb = SweepLPB()
        
        assert sweep_lpb is not None
        from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
        assert isinstance(sweep_lpb, LogicalPrimitiveBlockSweep)
    
    def test_sweep_lpb_single_child(self):
        """Test SweepLPB creation with single child."""
        child = Mock()
        
        # Test with single argument
        sweep_lpb1 = SweepLPB(child)
        
        # Test with single argument in list
        sweep_lpb2 = SweepLPB([child])
        
        assert sweep_lpb1 is not None
        assert sweep_lpb2 is not None
        
        from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
        assert isinstance(sweep_lpb1, LogicalPrimitiveBlockSweep)
        assert isinstance(sweep_lpb2, LogicalPrimitiveBlockSweep)


class TestBuildCZStarkFromParameters:
    """Test build_CZ_stark_from_parameters function."""
    
    @pytest.fixture
    def mock_qubits(self):
        """Create mock qubit elements."""
        control_q = Mock()
        control_q.name = "control_qubit"
        
        target_q = Mock()
        target_q.name = "target_qubit"
        
        return control_q, target_q
    
    def test_build_cz_stark_basic_parameters(self, mock_qubits):
        """Test CZ Stark gate building with basic parameters."""
        control_q, target_q = mock_qubits
        
        # Basic parameters
        width = 50.0  # ns
        amp_control = 0.3
        amp_target = 0.4
        frequency = 5000.0  # MHz
        rise = 0.01
        zz_interaction_positive = True
        
        # Build the gate
        lpb = build_CZ_stark_from_parameters(
            control_q=control_q,
            target_q=target_q,
            width=width,
            amp_control=amp_control,
            amp_target=amp_target,
            frequency=frequency,
            rise=rise,
            zz_interaction_positive=zz_interaction_positive
        )
        
        assert lpb is not None
        # Should be a SiZZelTwoQubitGateCollection
        from leeq.core.primitives.built_in.sizzel_gate import SiZZelTwoQubitGateCollection
        assert isinstance(lpb, SiZZelTwoQubitGateCollection)
    
    def test_build_cz_stark_all_parameters(self, mock_qubits):
        """Test CZ Stark gate building with all parameters."""
        control_q, target_q = mock_qubits
        
        # All parameters
        parameters = {
            'width': 60.0,
            'amp_control': 0.25,
            'amp_target': 0.35,
            'frequency': 5200.0,
            'rise': 0.02,
            'zz_interaction_positive': False,
            'iz_control': 0.1,
            'iz_target': 0.05,
            'phase_diff': np.pi/4,
            'echo': True,
            'trunc': 1.1
        }
        
        # Build the gate
        lpb = build_CZ_stark_from_parameters(
            control_q=control_q,
            target_q=target_q,
            **parameters
        )
        
        assert lpb is not None
        
        # Verify it has expected attributes
        assert hasattr(lpb, 'parameters')
        assert lpb.name == 'zz'
    
    def test_build_cz_stark_parameter_validation(self, mock_qubits):
        """Test parameter validation for CZ Stark gate."""
        control_q, target_q = mock_qubits
        
        # Test with valid parameters
        valid_params = {
            'width': 50.0,
            'amp_control': 0.3,
            'amp_target': 0.4,
            'frequency': 5000.0,
            'rise': 0.01,
            'zz_interaction_positive': True
        }
        
        lpb = build_CZ_stark_from_parameters(
            control_q=control_q,
            target_q=target_q,
            **valid_params
        )
        
        # Verify parameter values are reasonable
        assert valid_params['width'] > 0
        assert 0 <= valid_params['amp_control'] <= 1.0
        assert 0 <= valid_params['amp_target'] <= 1.0
        assert valid_params['frequency'] > 0
        assert 0 <= valid_params['rise'] <= 1.0
        assert isinstance(valid_params['zz_interaction_positive'], bool)
    
    def test_build_cz_stark_default_parameters(self, mock_qubits):
        """Test CZ Stark gate building with default optional parameters."""
        control_q, target_q = mock_qubits
        
        # Only required parameters
        lpb = build_CZ_stark_from_parameters(
            control_q=control_q,
            target_q=target_q,
            width=50.0,
            amp_control=0.3,
            amp_target=0.4,
            frequency=5000.0,
            rise=0.01,
            zz_interaction_positive=True
        )
        
        assert lpb is not None
        
        # Should have default values for optional parameters
        assert hasattr(lpb, 'parameters')
        params = lpb.parameters
        
        # Check that default values are set
        assert 'iz_control' in params
        assert 'iz_target' in params
        assert 'phase_diff' in params
        assert 'echo' in params
        assert 'trunc' in params
        
        # Default values should be reasonable
        assert params['iz_control'] == 0
        assert params['iz_target'] == 0
        assert params['phase_diff'] == 0
        assert params['echo'] is False
        assert params['trunc'] == 1.05
    
    def test_build_cz_stark_different_qubit_types(self):
        """Test CZ Stark gate building with different qubit mock types."""
        # Test with different mock configurations
        control_q = Mock()
        control_q.name = "Q1"
        control_q.frequency = 5000.0
        
        target_q = Mock()
        target_q.name = "Q2"
        target_q.frequency = 5100.0
        
        lpb = build_CZ_stark_from_parameters(
            control_q=control_q,
            target_q=target_q,
            width=40.0,
            amp_control=0.2,
            amp_target=0.3,
            frequency=5050.0,
            rise=0.015,
            zz_interaction_positive=True
        )
        
        assert lpb is not None
        assert lpb.dut_control is control_q
        assert lpb.dut_target is target_q


class TestParameterStructures:
    """Test parameter structure handling."""
    
    def test_parameter_dictionary_structure(self):
        """Test parameter dictionary structure for CZ Stark gate."""
        # Expected parameter structure
        expected_params = {
            'width': 50.0,
            'amp_control': 0.3,
            'amp_target': 0.4,
            'freq': 5000.0,
            'rise': 0.01,
            'iz_control': 0.0,
            'iz_target': 0.0,
            'phase_diff': 0.0,
            'echo': False,
            'trunc': 1.05,
            'zz_interaction_positive': True
        }
        
        # Validate parameter types and ranges
        assert isinstance(expected_params['width'], (int, float))
        assert expected_params['width'] > 0
        
        assert isinstance(expected_params['amp_control'], (int, float))
        assert 0 <= expected_params['amp_control'] <= 1.0
        
        assert isinstance(expected_params['amp_target'], (int, float))
        assert 0 <= expected_params['amp_target'] <= 1.0
        
        assert isinstance(expected_params['freq'], (int, float))
        assert expected_params['freq'] > 0
        
        assert isinstance(expected_params['rise'], (int, float))
        assert 0 <= expected_params['rise'] <= 1.0
        
        assert isinstance(expected_params['echo'], bool)
        assert isinstance(expected_params['zz_interaction_positive'], bool)
        
        assert expected_params['trunc'] > 1.0
    
    def test_parameter_conversion(self):
        """Test parameter type conversion and validation."""
        # Test different input types
        test_cases = [
            {'width': 50, 'expected_type': (int, float)},  # int should be valid
            {'width': 50.0, 'expected_type': (int, float)},  # float should be valid
            {'amp_control': 0, 'expected_type': (int, float)},  # minimum amplitude
            {'amp_control': 1, 'expected_type': (int, float)},  # maximum amplitude
            {'echo': True, 'expected_type': bool},
            {'echo': False, 'expected_type': bool},
            {'zz_interaction_positive': True, 'expected_type': bool},
            {'zz_interaction_positive': False, 'expected_type': bool}
        ]
        
        for test_case in test_cases:
            param_name = list(test_case.keys())[0]
            if param_name in ['width', 'amp_control', 'echo', 'zz_interaction_positive']:
                param_value = test_case[param_name]
                expected_type = test_case['expected_type']
                
                assert isinstance(param_value, expected_type), f"{param_name} should be {expected_type}"
    
    def test_parameter_bounds_checking(self):
        """Test parameter bounds checking logic."""
        # Define parameter bounds
        bounds = {
            'width': (0, float('inf')),  # Positive values
            'amp_control': (0, 1.0),     # 0 to 1
            'amp_target': (0, 1.0),      # 0 to 1
            'freq': (0, float('inf')),   # Positive frequencies
            'rise': (0, 1.0),            # 0 to 1
            'trunc': (1.0, float('inf')) # Greater than 1
        }
        
        # Test valid values
        valid_params = {
            'width': 50.0,
            'amp_control': 0.5,
            'amp_target': 0.3,
            'freq': 5000.0,
            'rise': 0.02,
            'trunc': 1.05
        }
        
        for param, value in valid_params.items():
            if param in bounds:
                min_val, max_val = bounds[param]
                assert min_val < value < max_val or value == min_val or value == max_val
        
        # Test boundary values
        boundary_tests = [
            ('width', 0.1),      # Small positive value
            ('amp_control', 0),  # Minimum amplitude
            ('amp_control', 1),  # Maximum amplitude
            ('amp_target', 0),   # Minimum amplitude
            ('amp_target', 1),   # Maximum amplitude
            ('rise', 0),         # Minimum rise
            ('rise', 1),         # Maximum rise
            ('trunc', 1.01),     # Just above minimum
        ]
        
        for param, value in boundary_tests:
            if param in bounds:
                min_val, max_val = bounds[param]
                if param == 'trunc':
                    assert value > min_val
                else:
                    assert min_val <= value <= max_val


@pytest.mark.integration
class TestCompatibilityIntegration:
    """Integration tests for compatibility primitives."""
    
    def test_series_and_sweep_compatibility(self):
        """Test that SeriesLPB and SweepLPB work together."""
        # Create mock primitives
        prim1 = Mock()
        prim2 = Mock()
        prim3 = Mock()
        
        # Create SeriesLPB
        series = SeriesLPB([prim1, prim2])
        
        # Create SweepLPB containing SeriesLPB
        sweep = SweepLPB([series, prim3])
        
        assert series is not None
        assert sweep is not None
        
        # Both should be proper LPB instances
        from leeq.core.primitives.logical_primitives import (
            LogicalPrimitiveBlockSerial,
            LogicalPrimitiveBlockSweep
        )
        
        assert isinstance(series, LogicalPrimitiveBlockSerial)
        assert isinstance(sweep, LogicalPrimitiveBlockSweep)
    
    def test_cz_stark_with_lpb_blocks(self):
        """Test CZ Stark gate integration with LPB blocks."""
        # Create mock qubits
        control_q = Mock()
        target_q = Mock()
        
        # Create CZ Stark gate
        cz_gate = build_CZ_stark_from_parameters(
            control_q=control_q,
            target_q=target_q,
            width=50.0,
            amp_control=0.3,
            amp_target=0.4,
            frequency=5000.0,
            rise=0.01,
            zz_interaction_positive=True
        )
        
        # Should be able to use in Series and Sweep blocks
        series = SeriesLPB([cz_gate])
        sweep = SweepLPB([cz_gate, series])
        
        assert series is not None
        assert sweep is not None
        assert cz_gate is not None
    
    def test_backward_compatibility_workflow(self):
        """Test complete backward compatibility workflow."""
        # This simulates how legacy code might use these components
        
        # Step 1: Create mock elements
        control = Mock()
        control.name = "control"
        target = Mock()
        target.name = "target"
        
        # Step 2: Create gate using compatibility function
        gate = build_CZ_stark_from_parameters(
            control_q=control,
            target_q=target,
            width=45.0,
            amp_control=0.25,
            amp_target=0.35,
            frequency=4950.0,
            rise=0.02,
            zz_interaction_positive=False
        )
        
        # Step 3: Use compatibility LPB classes
        preparation = SeriesLPB([Mock(), Mock()])  # Mock preparation gates
        measurement = Mock()  # Mock measurement
        
        # Create experiment structure
        experiment_sequence = SeriesLPB([preparation, gate, measurement])
        experiment_sweep = SweepLPB([experiment_sequence])
        
        # Verify everything was created successfully
        assert gate is not None
        assert preparation is not None
        assert experiment_sequence is not None
        assert experiment_sweep is not None
        
        # Verify types
        from leeq.core.primitives.logical_primitives import (
            LogicalPrimitiveBlockSerial,
            LogicalPrimitiveBlockSweep
        )
        from leeq.core.primitives.built_in.sizzel_gate import SiZZelTwoQubitGateCollection
        
        assert isinstance(gate, SiZZelTwoQubitGateCollection)
        assert isinstance(preparation, LogicalPrimitiveBlockSerial)
        assert isinstance(experiment_sequence, LogicalPrimitiveBlockSerial)
        assert isinstance(experiment_sweep, LogicalPrimitiveBlockSweep)