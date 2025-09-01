"""
Parameter extraction test for ResonatorSweepTransmissionWithExtraInitialLPB.

This module validates the parameter extraction method that converts HighLevelSimulationSetup
configurations into parameters suitable for MultiQubitDispersiveReadoutSimulator.

Task 1.2: Create parameter extraction helper method structure
- Test that _extract_params returns valid parameter dictionaries with expected keys
- Validate coupling matrix construction from dispersive shifts
- Ensure channel map is properly constructed for multiplexed readout
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from leeq.experiments.builtin.basic.calibrations.resonator_spectroscopy import ResonatorSweepTransmissionWithExtraInitialLPB


@pytest.fixture
def mock_virtual_qubit_single():
    """Create mock VirtualTransmon for single-qubit system."""
    vq = Mock()
    vq.qubit_frequency = 5000.0
    vq.readout_frequency = 7000.0
    vq.anharmonicity = -200.0
    vq.readout_linewidth = 1.0
    vq.readout_dipsersive_shift = 1.0  # Note: typo preserved for compatibility
    return vq

@pytest.fixture
def mock_virtual_qubits_multi():
    """Create mock VirtualTransmons for multi-qubit system."""
    vq1 = Mock()
    vq1.qubit_frequency = 5000.0
    vq1.readout_frequency = 7000.0
    vq1.anharmonicity = -200.0
    vq1.readout_linewidth = 1.0
    vq1.readout_dipsersive_shift = 1.0
    
    vq2 = Mock()
    vq2.qubit_frequency = 5200.0
    vq2.readout_frequency = 7500.0
    vq2.anharmonicity = -180.0
    vq2.readout_linewidth = 1.5
    vq2.readout_dipsersive_shift = 0.8
    
    return [vq1, vq2]

@pytest.fixture
def mock_setup_single(mock_virtual_qubit_single):
    """Create mock HighLevelSimulationSetup for single qubit."""
    setup = Mock()
    setup._virtual_qubits = {'Q1': mock_virtual_qubit_single}
    setup.get_coupling_strength_by_qubit.return_value = 0  # No couplings
    return setup

@pytest.fixture
def mock_setup_multi(mock_virtual_qubits_multi):
    """Create mock HighLevelSimulationSetup for multi-qubit system."""
    setup = Mock()
    vq1, vq2 = mock_virtual_qubits_multi
    setup._virtual_qubits = {'Q1': vq1, 'Q2': vq2}
    
    # Mock coupling strength method
    def get_coupling(vq_a, vq_b):
        if (vq_a == vq1 and vq_b == vq2) or (vq_a == vq2 and vq_b == vq1):
            return 5.0  # 5 MHz coupling
        return 0
        
    setup.get_coupling_strength_by_qubit.side_effect = get_coupling
    return setup

@pytest.fixture
def mock_dut_qubit():
    """Create mock DUT qubit element."""
    dut = Mock()
    dut.name = "test_qubit"
    return dut

@pytest.fixture
def experiment():
    """Create experiment instance for testing."""
    # Create instance without calling constructor (which tries to run)
    exp = ResonatorSweepTransmissionWithExtraInitialLPB.__new__(ResonatorSweepTransmissionWithExtraInitialLPB)
    return exp


class TestParameterExtraction:
    """Test parameter extraction from HighLevelSimulationSetup."""
    
    def test_extract_params_structure(self, experiment, mock_setup_single, mock_dut_qubit):
        """
        Test that _extract_params returns correct dictionary structure.
        
        This test validates the basic structure and keys of the returned
        parameter dictionary and channel map.
        """
        params_dict, channel_map, string_to_int_channel_map = experiment._extract_params(mock_setup_single, mock_dut_qubit)
        
        # Validate params_dict structure
        assert isinstance(params_dict, dict)
        required_param_keys = [
            'qubit_frequencies', 'qubit_anharmonicities', 'resonator_frequencies',
            'resonator_kappas', 'coupling_matrix', 'n_qubits', 'n_resonators'
        ]
        
        for key in required_param_keys:
            assert key in params_dict, f"Missing required parameter key: {key}"
        
        # Validate channel_map structure
        assert isinstance(channel_map, dict)
        assert len(channel_map) > 0, "Channel map should not be empty"
        
        # Each channel should map to a list of resonator indices
        for channel_id, resonator_indices in channel_map.items():
            assert isinstance(resonator_indices, list)
            assert len(resonator_indices) > 0
            assert all(isinstance(idx, int) for idx in resonator_indices)
        
        print("✅ Parameter extraction returns correct structure")
        # Test completed successfully
    
    def test_single_qubit_parameter_extraction(self, experiment, mock_setup_single, mock_dut_qubit):
        """
        Test parameter extraction for single-qubit system.
        
        Validates that single-qubit parameters are extracted correctly
        and coupling matrix contains only qubit-resonator coupling.
        """
        params_dict, channel_map, string_to_int_channel_map = experiment._extract_params(mock_setup_single, mock_dut_qubit)
        
        # Single qubit system checks
        assert params_dict['n_qubits'] == 1
        assert params_dict['n_resonators'] == 1
        
        # Parameter arrays should have length 1
        assert len(params_dict['qubit_frequencies']) == 1
        assert len(params_dict['qubit_anharmonicities']) == 1
        assert len(params_dict['resonator_frequencies']) == 1
        assert len(params_dict['resonator_kappas']) == 1
        
        # Validate parameter values
        assert params_dict['qubit_frequencies'][0] == 5000.0
        assert params_dict['qubit_anharmonicities'][0] == -200.0
        assert params_dict['resonator_frequencies'][0] == 7000.0
        assert params_dict['resonator_kappas'][0] == 1.0
        
        # Coupling matrix should have one qubit-resonator coupling
        coupling_matrix = params_dict['coupling_matrix']
        assert isinstance(coupling_matrix, dict)
        assert ('Q0', 'R0') in coupling_matrix
        
        # Calculate expected coupling from dispersive shift
        chi = 1.0
        delta = 7000.0 - 5000.0  # readout_freq - qubit_freq
        expected_g = (abs(chi * delta)) ** 0.5
        actual_g = coupling_matrix[('Q0', 'R0')]
        assert abs(actual_g - expected_g) < 0.01 * expected_g
        
        # Channel map should have 1:1 mapping with integer keys
        assert len(channel_map) == 1
        channel_ids = list(channel_map.keys())
        assert channel_ids[0] == 0  # Integer channel ID
        assert channel_map[channel_ids[0]] == [0]  # Maps to resonator 0
        
        # String to integer mapping should exist
        assert len(string_to_int_channel_map) == 1
        assert string_to_int_channel_map['Q1'] == 0
        
        print(f"✅ Single-qubit parameters extracted: g={actual_g:.1f}")
        # Test completed successfully
    
    def test_multi_qubit_parameter_extraction(self, experiment, mock_setup_multi, mock_dut_qubit):
        """
        Test parameter extraction for multi-qubit system.
        
        Validates that multi-qubit parameters are extracted correctly
        including qubit-qubit couplings.
        """
        params_dict, channel_map, string_to_int_channel_map = experiment._extract_params(mock_setup_multi, mock_dut_qubit)
        
        # Multi-qubit system checks
        assert params_dict['n_qubits'] == 2
        assert params_dict['n_resonators'] == 2
        
        # Parameter arrays should have length 2
        assert len(params_dict['qubit_frequencies']) == 2
        assert len(params_dict['qubit_anharmonicities']) == 2
        assert len(params_dict['resonator_frequencies']) == 2
        assert len(params_dict['resonator_kappas']) == 2
        
        # Validate parameter values (sorted by virtual qubit keys)
        expected_qubit_freqs = [5000.0, 5200.0]
        expected_resonator_freqs = [7000.0, 7500.0]
        expected_anharmonicities = [-200.0, -180.0]
        expected_kappas = [1.0, 1.5]
        
        assert params_dict['qubit_frequencies'] == expected_qubit_freqs
        assert params_dict['resonator_frequencies'] == expected_resonator_freqs
        assert params_dict['qubit_anharmonicities'] == expected_anharmonicities
        assert params_dict['resonator_kappas'] == expected_kappas
        
        # Coupling matrix should have qubit-resonator and qubit-qubit couplings
        coupling_matrix = params_dict['coupling_matrix']
        assert isinstance(coupling_matrix, dict)
        
        # Check qubit-resonator couplings
        assert ('Q0', 'R0') in coupling_matrix
        assert ('Q1', 'R1') in coupling_matrix
        
        # Check qubit-qubit coupling
        assert ('Q0', 'Q1') in coupling_matrix
        assert coupling_matrix[('Q0', 'Q1')] == 5.0
        
        # Channel map should have 2 integer channels, each mapping to one resonator
        assert len(channel_map) == 2
        sorted_channels = sorted(channel_map.keys())
        assert sorted_channels == [0, 1]  # Integer channel IDs
        assert channel_map[0] == [0]
        assert channel_map[1] == [1]
        
        # String to integer mapping should exist for both channels
        assert len(string_to_int_channel_map) == 2
        assert string_to_int_channel_map['Q1'] == 0
        assert string_to_int_channel_map['Q2'] == 1
        
        print("✅ Multi-qubit parameters extracted with couplings")
        # Test completed successfully
    
    def test_coupling_matrix_physics(self, experiment, mock_setup_single, mock_dut_qubit):
        """
        Test that coupling matrix follows correct dispersive shift physics.
        
        Validates that g = sqrt(|chi * delta|) relationship is implemented correctly.
        """
        params_dict, _, string_to_int_channel_map = experiment._extract_params(mock_setup_single, mock_dut_qubit)
        coupling_matrix = params_dict['coupling_matrix']
        
        # Extract coupling strength
        g = coupling_matrix[('Q0', 'R0')]
        
        # Recalculate from physics
        chi = 1.0  # From mock
        delta = 7000.0 - 5000.0  # readout_freq - qubit_freq
        expected_g = (abs(chi * delta)) ** 0.5
        
        # Validate physics relationship
        assert abs(g - expected_g) < 1e-10, f"g={g}, expected={expected_g}"
        
        # Check that coupling is positive and reasonable
        assert g > 0, "Coupling strength should be positive"
        assert 1 < g < 1000, "Coupling strength should be in reasonable range (1-1000 MHz)"
        
        print(f"✅ Coupling matrix physics validated: g={g:.1f} MHz")
        # Test completed successfully
    
    def test_parameter_extraction_with_missing_attributes(self, experiment, mock_dut_qubit):
        """
        Test parameter extraction handles missing VirtualTransmon attributes gracefully.
        
        Validates that default values are used when attributes are missing.
        """
        # Create a simple class that only has the required attributes
        class MinimalVirtualTransmon:
            def __init__(self):
                self.qubit_frequency = 5000.0
                self.readout_frequency = 7000.0
                # Missing: anharmonicity, readout_linewidth, readout_dipsersive_shift
        
        vq = MinimalVirtualTransmon()
        
        setup = Mock()
        setup._virtual_qubits = {'Q1': vq}
        setup.get_coupling_strength_by_qubit.return_value = 0
        
        # Should not raise exception
        params_dict, channel_map, string_to_int_channel_map = experiment._extract_params(setup, mock_dut_qubit)
        
        # Check default values were applied
        assert len(params_dict['qubit_anharmonicities']) == 1
        assert params_dict['qubit_anharmonicities'][0] == -200.0  # Default
        
        assert len(params_dict['resonator_kappas']) == 1
        assert params_dict['resonator_kappas'][0] == 1.0  # Default
        
        coupling_matrix = params_dict['coupling_matrix']
        assert ('Q0', 'R0') in coupling_matrix
        # Should use default dispersive shift of 1.0
        
        print("✅ Missing attributes handled with defaults")
        # Test completed successfully
    
    def test_channel_map_construction(self, experiment, mock_setup_multi, mock_dut_qubit):
        """
        Test channel map construction for multiplexed readout.
        
        Validates that channels are properly mapped to resonator indices.
        """
        params_dict, channel_map, string_to_int_channel_map = experiment._extract_params(mock_setup_multi, mock_dut_qubit)
        
        # Channel map validation - now uses integer channel IDs
        assert isinstance(channel_map, dict)
        assert len(channel_map) == 2
        
        # Each channel should map to exactly one resonator (1:1 mapping)
        for channel_id, resonator_indices in channel_map.items():
            assert isinstance(channel_id, int)  # Channel IDs should be integers
            assert len(resonator_indices) == 1
            assert isinstance(resonator_indices[0], int)
            assert 0 <= resonator_indices[0] < params_dict['n_resonators']
        
        # All resonators should be mapped
        mapped_resonators = set()
        for resonator_indices in channel_map.values():
            mapped_resonators.update(resonator_indices)
        
        expected_resonators = set(range(params_dict['n_resonators']))
        assert mapped_resonators == expected_resonators
        
        # String to integer channel mapping should exist
        assert isinstance(string_to_int_channel_map, dict)
        assert len(string_to_int_channel_map) == 2
        
        print("✅ Channel map constructed correctly")
        # Test completed successfully
    
    def test_extract_params_empty_setup(self, experiment, mock_dut_qubit):
        """
        Test parameter extraction with empty setup.
        
        Should handle edge case of no virtual qubits gracefully.
        """
        setup = Mock()
        setup._virtual_qubits = {}
        setup.get_coupling_strength_by_qubit.return_value = 0
        
        params_dict, channel_map, string_to_int_channel_map = experiment._extract_params(setup, mock_dut_qubit)
        
        # Should handle empty case
        assert params_dict['n_qubits'] == 0
        assert params_dict['n_resonators'] == 0
        assert len(params_dict['qubit_frequencies']) == 0
        assert len(params_dict['resonator_frequencies']) == 0
        assert params_dict['coupling_matrix'] == {}
        assert channel_map == {}
        
        print("✅ Empty setup handled gracefully")
        # Test completed successfully
    
    def test_parameter_types_and_formats(self, experiment, mock_setup_single, mock_dut_qubit):
        """
        Test that extracted parameters have correct types and formats.
        
        Validates data types for MultiQubitDispersiveReadoutSimulator compatibility.
        """
        params_dict, channel_map, string_to_int_channel_map = experiment._extract_params(mock_setup_single, mock_dut_qubit)
        
        # Parameter type validation
        assert isinstance(params_dict['qubit_frequencies'], list)
        assert isinstance(params_dict['qubit_anharmonicities'], list)
        assert isinstance(params_dict['resonator_frequencies'], list) 
        assert isinstance(params_dict['resonator_kappas'], list)
        assert isinstance(params_dict['coupling_matrix'], dict)
        assert isinstance(params_dict['n_qubits'], int)
        assert isinstance(params_dict['n_resonators'], int)
        
        # Numeric type validation
        for freq in params_dict['qubit_frequencies']:
            assert isinstance(freq, (int, float))
            assert freq > 0
            
        for anharm in params_dict['qubit_anharmonicities']:
            assert isinstance(anharm, (int, float))
            assert anharm < 0  # Anharmonicity should be negative
            
        for freq in params_dict['resonator_frequencies']:
            assert isinstance(freq, (int, float))
            assert freq > 0
            
        for kappa in params_dict['resonator_kappas']:
            assert isinstance(kappa, (int, float))
            assert kappa > 0
            
        # Coupling matrix validation
        for key, coupling in params_dict['coupling_matrix'].items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(coupling, (int, float))
            assert coupling >= 0  # Couplings should be non-negative
        
        # Channel map validation - integer keys
        for channel_id, resonator_list in channel_map.items():
            assert isinstance(channel_id, int)  # Channel IDs are now integers
            assert isinstance(resonator_list, list)
            for idx in resonator_list:
                assert isinstance(idx, int)
                assert idx >= 0
        
        # String to integer channel map validation
        for channel_name, channel_id in string_to_int_channel_map.items():
            assert isinstance(channel_name, str)
            assert isinstance(channel_id, int)
        
        print("✅ Parameter types and formats validated")
        # Test completed successfully


class TestParameterExtractionIntegration:
    """Integration tests for parameter extraction in realistic scenarios."""
    
    def test_parameters_compatible_with_simulator(self, experiment, mock_setup_single, mock_dut_qubit):
        """
        Test that extracted parameters are compatible with MultiQubitDispersiveReadoutSimulator.
        
        This integration test validates that the parameter format can be used
        to initialize the simulator (without actually running it).
        """
        params_dict, channel_map, string_to_int_channel_map = experiment._extract_params(mock_setup_single, mock_dut_qubit)
        
        # Test that parameters can be unpacked for simulator initialization
        try:
            # This would be the actual call:
            # simulator = MultiQubitDispersiveReadoutSimulator(**params_dict)
            # But we just test the parameter format here
            
            # Validate required parameters for simulator
            simulator_required_params = [
                'qubit_frequencies', 'qubit_anharmonicities', 
                'resonator_frequencies', 'resonator_kappas',
                'coupling_matrix', 'n_qubits', 'n_resonators'
            ]
            
            for param in simulator_required_params:
                assert param in params_dict
                
            # Validate parameter consistency
            n_qubits = params_dict['n_qubits']
            n_resonators = params_dict['n_resonators']
            
            assert len(params_dict['qubit_frequencies']) == n_qubits
            assert len(params_dict['qubit_anharmonicities']) == n_qubits
            assert len(params_dict['resonator_frequencies']) == n_resonators
            assert len(params_dict['resonator_kappas']) == n_resonators
            
        except Exception as e:
            pytest.fail(f"Parameters not compatible with simulator: {e}")
        
        print("✅ Parameters compatible with MultiQubitDispersiveReadoutSimulator")
        # Test completed successfully
    
    def test_realistic_multi_qubit_system(self):
        """
        Test parameter extraction for a realistic 3-qubit system.
        
        This test uses more realistic parameters and validates the
        complete extraction process.
        """
        # Create realistic 3-qubit system
        qubit_params = [
            {'freq': 4900, 'readout': 6800, 'anharm': -220, 'kappa': 0.8, 'chi': 1.2},
            {'freq': 5100, 'readout': 7200, 'anharm': -200, 'kappa': 1.0, 'chi': 1.0},
            {'freq': 5300, 'readout': 7600, 'anharm': -180, 'kappa': 1.2, 'chi': 0.9}
        ]
        
        # Create mock virtual qubits
        virtual_qubits = {}
        for i, params in enumerate(qubit_params):
            vq = Mock()
            vq.qubit_frequency = params['freq']
            vq.readout_frequency = params['readout']
            vq.anharmonicity = params['anharm']
            vq.readout_linewidth = params['kappa']
            vq.readout_dipsersive_shift = params['chi']
            virtual_qubits[f'Q{i+1}'] = vq
        
        # Create mock setup with nearest-neighbor couplings
        setup = Mock()
        setup._virtual_qubits = virtual_qubits
        
        def get_coupling(vq_a, vq_b):
            # Nearest neighbor coupling of 10 MHz
            qubit_list = list(virtual_qubits.values())
            idx_a = qubit_list.index(vq_a)
            idx_b = qubit_list.index(vq_b)
            if abs(idx_a - idx_b) == 1:  # Nearest neighbors
                return 10.0
            return 0
            
        setup.get_coupling_strength_by_qubit.side_effect = get_coupling
        
        # Test parameter extraction using class method directly
        experiment = ResonatorSweepTransmissionWithExtraInitialLPB.__new__(ResonatorSweepTransmissionWithExtraInitialLPB)
        dut_qubit = Mock()
        
        params_dict, channel_map, string_to_int_channel_map = experiment._extract_params(setup, dut_qubit)
        
        # Validate 3-qubit system
        assert params_dict['n_qubits'] == 3
        assert params_dict['n_resonators'] == 3
        
        # Check all parameters extracted
        assert len(params_dict['qubit_frequencies']) == 3
        assert len(params_dict['resonator_frequencies']) == 3
        
        # Check coupling matrix has nearest-neighbor couplings
        coupling_matrix = params_dict['coupling_matrix']
        assert ('Q0', 'Q1') in coupling_matrix
        assert ('Q1', 'Q2') in coupling_matrix
        assert coupling_matrix[('Q0', 'Q1')] == 10.0
        assert coupling_matrix[('Q1', 'Q2')] == 10.0
        
        # Should not have Q0-Q2 coupling (not nearest neighbors)
        assert ('Q0', 'Q2') not in coupling_matrix
        
        # Check channel map - integer keys
        assert len(channel_map) == 3
        expected_integer_channels = [0, 1, 2]
        assert sorted(channel_map.keys()) == expected_integer_channels
        
        # Check string to integer mapping
        assert len(string_to_int_channel_map) == 3
        expected_string_channels = ['Q1', 'Q2', 'Q3']
        assert sorted(string_to_int_channel_map.keys()) == expected_string_channels
        
        print("✅ Realistic 3-qubit system parameters extracted successfully")
        # Test completed successfully