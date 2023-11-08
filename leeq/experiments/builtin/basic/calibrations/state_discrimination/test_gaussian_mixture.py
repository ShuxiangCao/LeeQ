import pytest
from sklearn.mixture import GaussianMixture
import pytest
import numpy as np
from leeq.experiments.builtin import (fit_gmm_model, measurement_transform_gmm,
                                      find_output_map, calculate_signal_to_noise_ratio)
from sklearn.pipeline import Pipeline


@pytest.fixture
def sample_data():
    # Generating some random sample data for testing purposes
    np.random.seed(0)
    data = np.random.rand(100, 2)

    return data


@pytest.fixture
def sample_complex_data_simple(sample_data):
    return sample_data[:, 0] + 1j * sample_data[:, 1]


@pytest.fixture
def sample_complex_data():
    # Generating some random sample complex data for testing purposes
    np.random.seed(0)
    data_1 = (np.random.standard_normal(100) + 1j * np.random.standard_normal(100)) + (1 + 2.j)
    data_2 = (np.random.standard_normal(100) + 1j * np.random.standard_normal(100)) + (5 + 6.j)

    # draw a dataset that is 30% from data_1 and 70% from data_2
    data_a = np.random.choice([0, 1], size=100, p=[0.3, 0.7])
    data_a = np.concatenate([data_1[data_a == 0], data_2[data_a == 1]])

    # draw a dataset that is 70% from data_1 and 30% from data_2
    data_b = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    data_b = np.concatenate([data_1[data_b == 0], data_2[data_b == 1]])

    # stack the two datasets
    data = np.stack((data_a, data_b), axis=1)

    return data


@pytest.fixture
def clf(sample_data):
    # Create a GaussianMixture and fit it to the sample data
    gmm = GaussianMixture(n_components=2, covariance_type='full')
    clf = Pipeline([('gmm', gmm)])
    clf.fit(sample_data)
    return clf


def test_fit_gmm_model(sample_complex_data_simple):
    # Test successful model fitting
    n_components = 2
    pipeline = fit_gmm_model(sample_complex_data_simple, n_components)
    assert isinstance(pipeline, Pipeline)
    assert pipeline['gmm'].n_components == n_components

    # Value error for invalid data type
    with pytest.raises(ValueError):
        fit_gmm_model(sample_complex_data_simple.flatten().real, n_components)

    # Test ValueError for invalid n_components
    with pytest.raises(ValueError):
        fit_gmm_model(sample_complex_data_simple, 0)


def test_measurement_transform_gmm(sample_complex_data, clf):
    output_map = {0: 0, 1: 1}

    transformed_data = measurement_transform_gmm(sample_complex_data, '<zs>', clf, output_map)
    assert transformed_data.shape == sample_complex_data.shape

    # Test for different basis
    result = measurement_transform_gmm(sample_complex_data, 'bin', clf, output_map)
    assert isinstance(result, np.ndarray)

    # Test for RuntimeError with unknown basis
    with pytest.raises(RuntimeError):
        measurement_transform_gmm(sample_complex_data, 'unknown_basis', clf, output_map)


def test_find_output_map(sample_data, sample_complex_data, clf):
    n_components = 2
    outcome_map = find_output_map(sample_complex_data, clf)
    assert len(outcome_map) == n_components

    # Test for RuntimeError
    clf_bad = Pipeline([('gmm', GaussianMixture(n_components=n_components + 1))])
    clf_bad.fit(sample_data)
    with pytest.raises(RuntimeError):
        find_output_map(sample_complex_data, clf_bad)


def test_calculate_signal_to_noise_ratio(clf):
    outcome_map = [0, 1]
    snr = calculate_signal_to_noise_ratio(clf, outcome_map)
    assert isinstance(snr, dict)

    # Test for ValueError
    bad_clf = Pipeline([('no_gmm', 123)])
    with pytest.raises(ValueError):
        calculate_signal_to_noise_ratio(bad_clf, outcome_map)
