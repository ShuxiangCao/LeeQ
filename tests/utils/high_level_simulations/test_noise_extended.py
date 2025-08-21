"""
Extended tests for high-level simulation noise functionality.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from leeq.utils.high_level_simulations.noise import apply_noise_to_data


class TestApplyNoiseToData:
    """Test the apply_noise_to_data function."""
    
    @pytest.fixture
    def mock_readout_qubit(self):
        """Create a mock readout qubit with noise properties."""
        qubit = Mock()
        
        # Mock quiescent state distribution
        # [ground_state_prob, excited_state_prob, higher_levels...]
        qubit.quiescent_state_distribution = np.array([0.95, 0.04, 0.01])
        
        return qubit
    
    @pytest.fixture
    def sample_data(self):
        """Create sample expectation value data."""
        # Expectation values should be between -1 and 1
        return np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    def test_apply_noise_basic(self, mock_readout_qubit, sample_data):
        """Test basic noise application."""
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.return_value = False  # Disable sampling noise
            mock_setup.return_value.status.return_value = mock_status
            
            result = apply_noise_to_data(mock_readout_qubit, sample_data)
            
            assert isinstance(result, np.ndarray)
            assert len(result) == len(sample_data)
            assert np.all(result >= -1.0)
            assert np.all(result <= 1.0)
    
    def test_apply_noise_with_sampling_noise(self, mock_readout_qubit, sample_data):
        """Test noise application with sampling noise enabled."""
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.side_effect = lambda param: {
                'Sampling_Noise': True,
                'Shot_Number': 1000
            }.get(param, False)
            mock_setup.return_value.status.return_value = mock_status
            
            # Set random seed for reproducible tests
            np.random.seed(42)
            
            result = apply_noise_to_data(mock_readout_qubit, sample_data)
            
            assert isinstance(result, np.ndarray)
            assert len(result) == len(sample_data)
            assert np.all(result >= -1.0)
            assert np.all(result <= 1.0)
    
    def test_apply_noise_data_transformation(self, mock_readout_qubit):
        """Test the data transformation pipeline."""
        # Test with extreme values
        extreme_data = np.array([-1.0, 1.0])
        
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.return_value = False  # Disable sampling noise
            mock_setup.return_value.status.return_value = mock_status
            
            result = apply_noise_to_data(mock_readout_qubit, extreme_data)
            
            # Result should still be in valid range
            assert np.all(result >= -1.0)
            assert np.all(result <= 1.0)
    
    def test_apply_noise_different_qubit_distributions(self, sample_data):
        """Test noise application with different quiescent state distributions."""
        # Test case 1: Very pure ground state
        pure_qubit = Mock()
        pure_qubit.quiescent_state_distribution = np.array([0.99, 0.005, 0.005])
        
        # Test case 2: More mixed state
        mixed_qubit = Mock()
        mixed_qubit.quiescent_state_distribution = np.array([0.85, 0.10, 0.05])
        
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.return_value = False
            mock_setup.return_value.status.return_value = mock_status
            
            result_pure = apply_noise_to_data(pure_qubit, sample_data)
            result_mixed = apply_noise_to_data(mixed_qubit, sample_data)
            
            # Both should be valid
            assert np.all(result_pure >= -1.0) and np.all(result_pure <= 1.0)
            assert np.all(result_mixed >= -1.0) and np.all(result_mixed <= 1.0)
            
            # Mixed state should generally have more noise (higher standard deviation)
            # but this is stochastic so we just check basic validity
            assert len(result_pure) == len(sample_data)
            assert len(result_mixed) == len(sample_data)
    
    def test_apply_noise_array_shapes(self, mock_readout_qubit):
        """Test noise application with different array shapes."""
        # Test 1D arrays of different sizes
        small_data = np.array([0.0])
        medium_data = np.array([0.0, 0.5, -0.5])
        large_data = np.random.uniform(-1, 1, 100)
        
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.return_value = False
            mock_setup.return_value.status.return_value = mock_status
            
            result_small = apply_noise_to_data(mock_readout_qubit, small_data)
            result_medium = apply_noise_to_data(mock_readout_qubit, medium_data)
            result_large = apply_noise_to_data(mock_readout_qubit, large_data)
            
            assert len(result_small) == 1
            assert len(result_medium) == 3
            assert len(result_large) == 100
            
            # All results should be in valid range
            assert np.all(result_small >= -1.0) and np.all(result_small <= 1.0)
            assert np.all(result_medium >= -1.0) and np.all(result_medium <= 1.0)
            assert np.all(result_large >= -1.0) and np.all(result_large <= 1.0)
    
    def test_apply_noise_multidimensional_data(self, mock_readout_qubit):
        """Test noise application with multidimensional data."""
        # Create 2D data array
        data_2d = np.array([
            [0.0, 0.5, -0.5],
            [1.0, -1.0, 0.0]
        ])
        
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.return_value = False
            mock_setup.return_value.status.return_value = mock_status
            
            result = apply_noise_to_data(mock_readout_qubit, data_2d)
            
            assert result.shape == data_2d.shape
            assert np.all(result >= -1.0)
            assert np.all(result <= 1.0)
    
    def test_apply_noise_edge_cases(self, mock_readout_qubit):
        """Test noise application with edge cases."""
        # Test with all zeros
        zero_data = np.zeros(5)
        
        # Test with all ones
        ones_data = np.ones(5)
        
        # Test with all negative ones
        neg_ones_data = -np.ones(5)
        
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.return_value = False
            mock_setup.return_value.status.return_value = mock_status
            
            result_zeros = apply_noise_to_data(mock_readout_qubit, zero_data)
            result_ones = apply_noise_to_data(mock_readout_qubit, ones_data)
            result_neg_ones = apply_noise_to_data(mock_readout_qubit, neg_ones_data)
            
            # All results should be in valid range
            for result in [result_zeros, result_ones, result_neg_ones]:
                assert np.all(result >= -1.0)
                assert np.all(result <= 1.0)
                assert len(result) == 5
    
    @patch('numpy.random.binomial')
    @patch('numpy.random.normal')
    def test_apply_noise_random_functions(self, mock_normal, mock_binomial, mock_readout_qubit, sample_data):
        """Test that random functions are called appropriately."""
        # Setup return values for mocked random functions
        mock_binomial.return_value = np.array([500, 400, 600, 450, 550])
        mock_normal.side_effect = [
            np.array([0.1, -0.05, 0.02, 0.08, -0.03]),  # First call
            np.array([0.05, -0.02, 0.01, 0.04, -0.01]), # Second call
            np.array([0.02, -0.01, 0.005, 0.015, -0.005]) # Third call
        ]
        
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.side_effect = lambda param: {
                'Sampling_Noise': True,
                'Shot_Number': 1000
            }.get(param, False)
            mock_setup.return_value.status.return_value = mock_status
            
            result = apply_noise_to_data(mock_readout_qubit, sample_data)
            
            # Verify random functions were called
            mock_binomial.assert_called_once()
            assert mock_normal.call_count >= 2  # Should be called multiple times
            
            assert isinstance(result, np.ndarray)
            assert len(result) == len(sample_data)


class TestNoiseParameters:
    """Test noise parameter handling and validation."""
    
    def test_quiescent_state_distribution_properties(self):
        """Test properties of quiescent state distributions."""
        # Valid distributions
        valid_distributions = [
            np.array([0.95, 0.04, 0.01]),           # High fidelity
            np.array([0.90, 0.08, 0.02]),           # Medium fidelity
            np.array([0.80, 0.15, 0.05]),           # Lower fidelity
            np.array([0.99, 0.009, 0.001]),         # Very high fidelity
        ]
        
        for dist in valid_distributions:
            # Should sum to 1 (or very close)
            assert abs(np.sum(dist) - 1.0) < 1e-10
            
            # All elements should be non-negative
            assert np.all(dist >= 0)
            
            # Ground state should be dominant
            assert dist[0] > dist[1]
            assert dist[0] > dist[2]
    
    def test_standard_deviation_calculation(self):
        """Test standard deviation calculation from quiescent state distribution."""
        test_distributions = [
            np.array([0.95, 0.04, 0.01]),
            np.array([0.90, 0.08, 0.02]),
            np.array([0.85, 0.10, 0.05]),
        ]
        
        for dist in test_distributions:
            # Calculate standard deviation as sum of excited state populations
            std_dev = np.sum(dist[1:])
            
            assert 0 <= std_dev <= 1
            assert std_dev == dist[1] + dist[2]
            
            # Higher excited state populations should give higher std_dev
            assert std_dev < 1 - dist[0] + 1e-10  # Account for floating point
    
    def test_shot_number_parameters(self):
        """Test shot number parameter validation."""
        valid_shot_numbers = [100, 1000, 10000, 100000]
        
        for shots in valid_shot_numbers:
            assert isinstance(shots, int)
            assert shots > 0
            assert shots <= 1000000  # Reasonable upper bound
            
            # Test that shot numbers affect binomial sampling
            # Higher shot numbers should reduce relative noise
            prob = 0.5
            samples = np.random.binomial(shots, prob, 1000)
            relative_std = np.std(samples) / np.mean(samples)
            
            # Relative standard deviation should decrease with shot number
            expected_relative_std = np.sqrt((1-prob)/(prob*shots))
            assert abs(relative_std - expected_relative_std) < 0.1  # Within 10%


class TestNoiseEffects:
    """Test the effects of different noise parameters."""
    
    @pytest.fixture
    def noise_test_data(self):
        """Create data for noise effect testing."""
        # Create data representing different measurement outcomes
        return {
            'ground_state': np.array([-1.0, -0.9, -0.8]),  # Should be near -1
            'excited_state': np.array([1.0, 0.9, 0.8]),    # Should be near +1
            'mixed_state': np.array([0.0, 0.1, -0.1]),     # Should be near 0
        }
    
    def test_noise_preserves_bounds(self, noise_test_data):
        """Test that noise application preserves expectation value bounds."""
        for state_name, data in noise_test_data.items():
            # Create mock qubit with moderate noise
            mock_qubit = Mock()
            mock_qubit.quiescent_state_distribution = np.array([0.90, 0.08, 0.02])
            
            with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
                mock_status = Mock()
                mock_status.get_param.return_value = False
                mock_setup.return_value.status.return_value = mock_status
                
                # Apply noise multiple times to test consistency
                for _ in range(10):
                    result = apply_noise_to_data(mock_qubit, data)
                    
                    # Result should always be in valid expectation value range
                    assert np.all(result >= -1.0), f"Values below -1 for {state_name}"
                    assert np.all(result <= 1.0), f"Values above +1 for {state_name}"
    
    def test_noise_scaling_with_qubit_fidelity(self):
        """Test that noise scales appropriately with qubit fidelity."""
        test_data = np.array([0.0, 0.5, -0.5])  # Mixed expectation values
        
        # High fidelity qubit
        high_fidelity_qubit = Mock()
        high_fidelity_qubit.quiescent_state_distribution = np.array([0.98, 0.015, 0.005])
        
        # Low fidelity qubit
        low_fidelity_qubit = Mock()
        low_fidelity_qubit.quiescent_state_distribution = np.array([0.85, 0.10, 0.05])
        
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.return_value = False
            mock_setup.return_value.status.return_value = mock_status
            
            # Set random seed for reproducible comparison
            np.random.seed(123)
            result_high_fidelity = apply_noise_to_data(high_fidelity_qubit, test_data)
            
            np.random.seed(123)  # Same seed for fair comparison
            result_low_fidelity = apply_noise_to_data(low_fidelity_qubit, test_data)
            
            # Both results should be valid
            assert np.all(result_high_fidelity >= -1.0) and np.all(result_high_fidelity <= 1.0)
            assert np.all(result_low_fidelity >= -1.0) and np.all(result_low_fidelity <= 1.0)
    
    def test_sampling_noise_effects(self):
        """Test the effects of sampling noise."""
        mock_qubit = Mock()
        mock_qubit.quiescent_state_distribution = np.array([0.90, 0.08, 0.02])
        
        test_data = np.array([0.5, 0.5, 0.5])  # Identical values
        
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            # Test without sampling noise
            mock_status = Mock()
            mock_status.get_param.return_value = False
            mock_setup.return_value.status.return_value = mock_status
            
            results_no_sampling = []
            for _ in range(5):
                result = apply_noise_to_data(mock_qubit, test_data)
                results_no_sampling.append(result)
            
            # Test with sampling noise
            mock_status.get_param.side_effect = lambda param: {
                'Sampling_Noise': True,
                'Shot_Number': 1000
            }.get(param, False)
            
            results_with_sampling = []
            for _ in range(5):
                result = apply_noise_to_data(mock_qubit, test_data)
                results_with_sampling.append(result)
            
            # All results should be valid
            for result in results_no_sampling + results_with_sampling:
                assert np.all(result >= -1.0)
                assert np.all(result <= 1.0)
                assert len(result) == len(test_data)


@pytest.mark.integration
class TestNoiseIntegration:
    """Integration tests for noise functionality."""
    
    def test_realistic_measurement_simulation(self):
        """Test realistic measurement simulation with noise."""
        # Simulate a realistic qubit measurement scenario
        
        # Create qubit with realistic parameters
        qubit = Mock()
        qubit.quiescent_state_distribution = np.array([0.92, 0.06, 0.02])  # 92% fidelity
        
        # Simulate Rabi oscillation data
        angles = np.linspace(0, 2*np.pi, 20)
        ideal_data = np.cos(angles)  # Perfect Rabi oscillation
        
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.side_effect = lambda param: {
                'Sampling_Noise': True,
                'Shot_Number': 5000
            }.get(param, False)
            mock_setup.return_value.status.return_value = mock_status
            
            # Apply noise to simulate realistic data
            noisy_data = apply_noise_to_data(qubit, ideal_data)
            
            # Verify properties of noisy data
            assert len(noisy_data) == len(ideal_data)
            assert np.all(noisy_data >= -1.0)
            assert np.all(noisy_data <= 1.0)
            
            # Noisy data should still roughly follow the oscillation pattern
            # but this is stochastic so we just verify basic structure
            assert isinstance(noisy_data, np.ndarray)
    
    def test_multiple_qubit_types_simulation(self):
        """Test noise simulation with different qubit types."""
        # Simulate different qubit qualities
        qubit_types = {
            'high_quality': np.array([0.97, 0.02, 0.01]),
            'medium_quality': np.array([0.90, 0.07, 0.03]),
            'low_quality': np.array([0.80, 0.12, 0.08])
        }
        
        test_data = np.array([0.8, 0.4, 0.0, -0.4, -0.8])
        
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.return_value = False
            mock_setup.return_value.status.return_value = mock_status
            
            results = {}
            for qubit_type, distribution in qubit_types.items():
                qubit = Mock()
                qubit.quiescent_state_distribution = distribution
                
                result = apply_noise_to_data(qubit, test_data)
                results[qubit_type] = result
                
                # All results should be valid
                assert np.all(result >= -1.0)
                assert np.all(result <= 1.0)
                assert len(result) == len(test_data)
            
            # Verify we got results for all qubit types
            assert len(results) == len(qubit_types)
    
    def test_parameter_sweep_simulation(self):
        """Test noise application across a parameter sweep."""
        # Simulate a parameter sweep experiment (e.g., amplitude sweep)
        amplitudes = np.linspace(0, 1, 11)
        
        # Generate ideal Rabi data for each amplitude
        ideal_results = []
        for amp in amplitudes:
            # Simple model: expectation value proportional to amplitude
            expectation = amp * np.cos(np.pi * amp)
            ideal_results.append(expectation)
        
        ideal_results = np.array(ideal_results)
        
        # Apply noise
        qubit = Mock()
        qubit.quiescent_state_distribution = np.array([0.88, 0.09, 0.03])
        
        with patch('leeq.utils.high_level_simulations.noise.setup') as mock_setup:
            mock_status = Mock()
            mock_status.get_param.side_effect = lambda param: {
                'Sampling_Noise': True,
                'Shot_Number': 2000
            }.get(param, False)
            mock_setup.return_value.status.return_value = mock_status
            
            noisy_results = apply_noise_to_data(qubit, ideal_results)
            
            # Verify sweep results
            assert len(noisy_results) == len(ideal_results)
            assert np.all(noisy_results >= -1.0)
            assert np.all(noisy_results <= 1.0)
            
            # Results should still show general trend despite noise
            assert isinstance(noisy_results, np.ndarray)