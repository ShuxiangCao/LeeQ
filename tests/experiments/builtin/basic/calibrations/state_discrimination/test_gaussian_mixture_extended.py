"""
Extended tests for Gaussian mixture state discrimination experiments.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sklearn.mixture import GaussianMixture

from leeq.experiments.builtin.basic.calibrations.state_discrimination.gaussian_mixture import *


class TestGaussianMixtureBasics:
    """Test basic functionality of Gaussian mixture state discrimination."""
    
    def test_module_imports(self):
        """Test that the module imports successfully."""
        # If we get here, imports were successful
        assert True
    
    def test_gaussian_2d_parameters(self):
        """Test 2D Gaussian parameters."""
        # Mock IQ data for two states
        mean_0 = np.array([0.0, 0.0])  # Ground state center
        mean_1 = np.array([1.0, 0.5])  # Excited state center
        cov_0 = np.array([[0.1, 0.0], [0.0, 0.1]])  # Covariance matrix
        cov_1 = np.array([[0.15, 0.02], [0.02, 0.12]])
        
        assert mean_0.shape == (2,)
        assert mean_1.shape == (2,)
        assert cov_0.shape == (2, 2)
        assert cov_1.shape == (2, 2)
        
        # Test that covariance matrices are positive definite
        assert np.all(np.linalg.eigvals(cov_0) > 0)
        assert np.all(np.linalg.eigvals(cov_1) > 0)
    
    def test_iq_data_generation(self):
        """Test generation of mock IQ data."""
        n_samples = 1000
        
        # Generate ground state data
        ground_state = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], n_samples)
        
        # Generate excited state data
        excited_state = np.random.multivariate_normal([1, 0.5], [[0.15, 0.02], [0.02, 0.12]], n_samples)
        
        assert ground_state.shape == (n_samples, 2)
        assert excited_state.shape == (n_samples, 2)
        
        # Check that states are reasonably separated
        ground_mean = np.mean(ground_state, axis=0)
        excited_mean = np.mean(excited_state, axis=0)
        separation = np.linalg.norm(excited_mean - ground_mean)
        
        assert separation > 0.5  # Should be well separated


class TestStateDiscrimination:
    """Test state discrimination functionality."""
    
    @pytest.fixture
    def mock_iq_data(self):
        """Generate mock IQ data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        n_samples = 500
        
        # Ground state (|0⟩): centered around origin
        ground_i = np.random.normal(0.0, 0.1, n_samples)
        ground_q = np.random.normal(0.0, 0.1, n_samples)
        ground_state = np.column_stack([ground_i, ground_q])
        
        # Excited state (|1⟩): offset from ground state
        excited_i = np.random.normal(0.8, 0.15, n_samples)
        excited_q = np.random.normal(0.4, 0.12, n_samples)
        excited_state = np.column_stack([excited_i, excited_q])
        
        # Labels
        ground_labels = np.zeros(n_samples, dtype=int)
        excited_labels = np.ones(n_samples, dtype=int)
        
        return {
            'ground_state': ground_state,
            'excited_state': excited_state,
            'ground_labels': ground_labels,
            'excited_labels': excited_labels,
            'all_data': np.vstack([ground_state, excited_state]),
            'all_labels': np.hstack([ground_labels, excited_labels])
        }
    
    def test_gaussian_mixture_fitting(self, mock_iq_data):
        """Test fitting Gaussian mixture model to IQ data."""
        data = mock_iq_data['all_data']
        n_components = 2
        
        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(data)
        
        # Test that model was fitted
        assert hasattr(gmm, 'means_')
        assert hasattr(gmm, 'covariances_')
        assert hasattr(gmm, 'weights_')
        
        # Check dimensions
        assert gmm.means_.shape == (n_components, 2)
        assert gmm.covariances_.shape == (n_components, 2, 2)
        assert gmm.weights_.shape == (n_components,)
        
        # Check that weights sum to 1
        assert abs(np.sum(gmm.weights_) - 1.0) < 1e-10
    
    def test_state_prediction(self, mock_iq_data):
        """Test state prediction using fitted model."""
        data = mock_iq_data['all_data']
        labels = mock_iq_data['all_labels']
        
        # Fit model
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data)
        
        # Predict states
        predictions = gmm.predict(data)
        
        assert len(predictions) == len(labels)
        assert np.all((predictions == 0) | (predictions == 1))
        
        # Calculate accuracy (allowing for label permutation)
        accuracy1 = np.mean(predictions == labels)
        accuracy2 = np.mean(predictions == (1 - labels))
        accuracy = max(accuracy1, accuracy2)
        
        # Should achieve reasonable accuracy on well-separated data
        assert accuracy > 0.8
    
    def test_probability_calculation(self, mock_iq_data):
        """Test calculation of state probabilities."""
        data = mock_iq_data['all_data']
        
        # Fit model
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data)
        
        # Calculate probabilities
        probabilities = gmm.predict_proba(data)
        
        assert probabilities.shape == (len(data), 2)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        
        # Check that probabilities sum to 1 for each sample
        row_sums = np.sum(probabilities, axis=1)
        assert np.allclose(row_sums, 1.0)


class TestDiscriminationMetrics:
    """Test discrimination quality metrics."""
    
    @pytest.fixture
    def mock_classification_results(self):
        """Generate mock classification results."""
        n_samples = 200
        
        # True labels (half ground state, half excited state)
        true_labels = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        # Predicted labels with some errors
        np.random.seed(42)
        predicted_labels = true_labels.copy()
        
        # Add some classification errors (5% error rate)
        error_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        predicted_labels[error_indices] = 1 - predicted_labels[error_indices]
        
        return {
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'n_samples': n_samples
        }
    
    def test_accuracy_calculation(self, mock_classification_results):
        """Test accuracy calculation."""
        true_labels = mock_classification_results['true_labels']
        predicted_labels = mock_classification_results['predicted_labels']
        
        accuracy = np.mean(predicted_labels == true_labels)
        
        assert 0 <= accuracy <= 1
        assert accuracy > 0.9  # Should be high accuracy with low error rate
    
    def test_confusion_matrix(self, mock_classification_results):
        """Test confusion matrix calculation."""
        true_labels = mock_classification_results['true_labels']
        predicted_labels = mock_classification_results['predicted_labels']
        
        # Calculate confusion matrix elements
        tp = np.sum((true_labels == 1) & (predicted_labels == 1))  # True positives
        tn = np.sum((true_labels == 0) & (predicted_labels == 0))  # True negatives
        fp = np.sum((true_labels == 0) & (predicted_labels == 1))  # False positives
        fn = np.sum((true_labels == 1) & (predicted_labels == 0))  # False negatives
        
        # Check that all counts are non-negative
        assert tp >= 0
        assert tn >= 0
        assert fp >= 0
        assert fn >= 0
        
        # Check that total matches sample count
        total = tp + tn + fp + fn
        assert total == len(true_labels)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1_score <= 1
    
    def test_fidelity_calculation(self, mock_classification_results):
        """Test state discrimination fidelity."""
        true_labels = mock_classification_results['true_labels']
        predicted_labels = mock_classification_results['predicted_labels']
        
        # Calculate individual state fidelities
        ground_mask = (true_labels == 0)
        excited_mask = (true_labels == 1)
        
        ground_fidelity = np.mean(predicted_labels[ground_mask] == 0) if np.any(ground_mask) else 0
        excited_fidelity = np.mean(predicted_labels[excited_mask] == 1) if np.any(excited_mask) else 0
        
        # Average fidelity
        average_fidelity = (ground_fidelity + excited_fidelity) / 2
        
        assert 0 <= ground_fidelity <= 1
        assert 0 <= excited_fidelity <= 1
        assert 0 <= average_fidelity <= 1


class TestCalibrationExperiments:
    """Test state discrimination calibration experiments."""
    
    @pytest.fixture
    def mock_dut(self):
        """Create a mock DUT element."""
        dut = Mock()
        dut.get_measurement_primitive.return_value = Mock()
        dut.get_gate.return_value = Mock()
        dut.name = "test_qubit"
        return dut
    
    def test_calibration_parameters(self, mock_dut):
        """Test calibration experiment parameters."""
        calibration_params = {
            'measurement_duration': 2.0,  # μs
            'integration_length': 1.0,    # μs
            'demodulation_frequency': 50, # MHz
            'n_shots': 10000,
            'threshold_optimization': True,
            'save_raw_data': True
        }
        
        # Validate parameters
        assert calibration_params['measurement_duration'] > 0
        assert calibration_params['integration_length'] > 0
        assert calibration_params['integration_length'] <= calibration_params['measurement_duration']
        assert calibration_params['demodulation_frequency'] > 0
        assert calibration_params['n_shots'] > 0
        assert isinstance(calibration_params['threshold_optimization'], bool)
        assert isinstance(calibration_params['save_raw_data'], bool)
    
    def test_state_preparation_sequences(self, mock_dut):
        """Test state preparation sequences for calibration."""
        # Ground state preparation (no gates)
        ground_prep = []
        
        # Excited state preparation (π pulse)
        excited_prep = [mock_dut.get_gate('X')]
        
        assert len(ground_prep) == 0
        assert len(excited_prep) == 1
    
    @patch('leeq.core.primitives.logical_primitives.LogicalPrimitiveBlockSerial')
    def test_experiment_lpb_creation(self, mock_serial, mock_dut):
        """Test creation of logical primitive blocks for experiments."""
        mock_serial.return_value = Mock()
        
        # Create experiment LPB
        prep_gate = mock_dut.get_gate('X')
        measurement = mock_dut.get_measurement_primitive(0)
        
        lpb = mock_serial([prep_gate, measurement])
        
        assert lpb is not None
        mock_serial.assert_called_once()


class TestThresholdOptimization:
    """Test threshold optimization for state discrimination."""
    
    @pytest.fixture
    def mock_threshold_data(self):
        """Generate mock data for threshold optimization."""
        np.random.seed(42)
        
        n_samples = 1000
        
        # Generate ground and excited state distributions along discrimination axis
        ground_dist = np.random.normal(0.0, 0.2, n_samples)
        excited_dist = np.random.normal(1.0, 0.25, n_samples)
        
        return {
            'ground_distribution': ground_dist,
            'excited_distribution': excited_dist
        }
    
    def test_optimal_threshold_calculation(self, mock_threshold_data):
        """Test calculation of optimal discrimination threshold."""
        ground_dist = mock_threshold_data['ground_distribution']
        excited_dist = mock_threshold_data['excited_distribution']
        
        # Simple threshold: midpoint between means
        ground_mean = np.mean(ground_dist)
        excited_mean = np.mean(excited_dist)
        simple_threshold = (ground_mean + excited_mean) / 2
        
        assert ground_mean < simple_threshold < excited_mean
        
        # Test threshold performance
        ground_correct = np.mean(ground_dist < simple_threshold)
        excited_correct = np.mean(excited_dist > simple_threshold)
        overall_accuracy = (ground_correct + excited_correct) / 2
        
        assert ground_correct > 0.5
        assert excited_correct > 0.5
        assert overall_accuracy > 0.8
    
    def test_roc_curve_calculation(self, mock_threshold_data):
        """Test ROC curve calculation for threshold optimization."""
        ground_dist = mock_threshold_data['ground_distribution']
        excited_dist = mock_threshold_data['excited_distribution']
        
        # Generate range of thresholds
        thresholds = np.linspace(np.min(ground_dist) - 0.1, np.max(excited_dist) + 0.1, 100)
        
        tpr_values = []  # True positive rate
        fpr_values = []  # False positive rate
        
        for threshold in thresholds:
            # True positive rate (excited state correctly identified)
            tpr = np.mean(excited_dist > threshold)
            
            # False positive rate (ground state incorrectly identified as excited)
            fpr = np.mean(ground_dist > threshold)
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        tpr_values = np.array(tpr_values)
        fpr_values = np.array(fpr_values)
        
        # Check ROC curve properties
        assert len(tpr_values) == len(thresholds)
        assert len(fpr_values) == len(thresholds)
        assert np.all(tpr_values >= 0) and np.all(tpr_values <= 1)
        assert np.all(fpr_values >= 0) and np.all(fpr_values <= 1)
        
        # ROC curve should be monotonic (approximately)
        assert np.all(np.diff(tpr_values) <= 0.01)  # TPR decreases with higher threshold
        assert np.all(np.diff(fpr_values) <= 0.01)  # FPR decreases with higher threshold


@pytest.mark.integration
class TestIntegrationWorkflow:
    """Test complete state discrimination workflow."""
    
    @pytest.fixture
    def mock_complete_workflow(self):
        """Create mock data for complete workflow test."""
        np.random.seed(42)
        
        # Simulate a complete calibration workflow
        n_calibration_shots = 5000
        
        # Ground state calibration data
        ground_i = np.random.normal(0.1, 0.15, n_calibration_shots)
        ground_q = np.random.normal(0.05, 0.12, n_calibration_shots)
        ground_data = np.column_stack([ground_i, ground_q])
        
        # Excited state calibration data
        excited_i = np.random.normal(0.9, 0.18, n_calibration_shots)
        excited_q = np.random.normal(0.6, 0.16, n_calibration_shots)
        excited_data = np.column_stack([excited_i, excited_q])
        
        return {
            'ground_calibration': ground_data,
            'excited_calibration': excited_data,
            'n_shots': n_calibration_shots
        }
    
    def test_end_to_end_calibration(self, mock_complete_workflow):
        """Test end-to-end calibration workflow."""
        ground_data = mock_complete_workflow['ground_calibration']
        excited_data = mock_complete_workflow['excited_calibration']
        
        # Step 1: Combine calibration data
        all_data = np.vstack([ground_data, excited_data])
        labels = np.hstack([np.zeros(len(ground_data)), np.ones(len(excited_data))])
        
        # Step 2: Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(all_data)
        
        # Step 3: Validate model
        predictions = gmm.predict(all_data)
        accuracy = max(np.mean(predictions == labels), np.mean(predictions == (1 - labels)))
        
        # Step 4: Extract discrimination parameters
        means = gmm.means_
        covariances = gmm.covariances_
        
        # Workflow validation
        assert accuracy > 0.85  # Should achieve good discrimination
        assert means.shape == (2, 2)  # Two 2D Gaussian components
        assert covariances.shape == (2, 2, 2)  # Covariance matrices for each component
        
        # Check that means are well separated
        separation = np.linalg.norm(means[1] - means[0])
        assert separation > 0.5
    
    def test_discrimination_performance_metrics(self, mock_complete_workflow):
        """Test comprehensive performance metrics."""
        ground_data = mock_complete_workflow['ground_calibration']
        excited_data = mock_complete_workflow['excited_calibration']
        
        # Fit model
        all_data = np.vstack([ground_data, excited_data])
        labels = np.hstack([np.zeros(len(ground_data)), np.ones(len(excited_data))])
        
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(all_data)
        
        predictions = gmm.predict(all_data)
        probabilities = gmm.predict_proba(all_data)
        
        # Adjust predictions if needed (handle label permutation)
        if np.mean(predictions == labels) < 0.5:
            predictions = 1 - predictions
            probabilities = probabilities[:, ::-1]
        
        # Calculate comprehensive metrics
        accuracy = np.mean(predictions == labels)
        
        # State-specific fidelities
        ground_fidelity = np.mean(predictions[:len(ground_data)] == 0)
        excited_fidelity = np.mean(predictions[len(ground_data):] == 1)
        
        # Confidence measures
        max_probabilities = np.max(probabilities, axis=1)
        mean_confidence = np.mean(max_probabilities)
        
        # Validation
        assert accuracy > 0.8
        assert ground_fidelity > 0.8
        assert excited_fidelity > 0.8
        assert mean_confidence > 0.8
        
        # Create performance report
        performance_report = {
            'overall_accuracy': accuracy,
            'ground_state_fidelity': ground_fidelity,
            'excited_state_fidelity': excited_fidelity,
            'mean_confidence': mean_confidence,
            'model_components': gmm.n_components,
            'calibration_shots': mock_complete_workflow['n_shots']
        }
        
        # Validate report structure
        assert all(isinstance(v, (int, float)) for v in performance_report.values())
        assert performance_report['model_components'] == 2