import numpy as np
import pytest

from leeq.theory.fits.multilevel_decay import normalize_gamma, single_photon_gamma_encode, single_photon_gamma_decode, \
    simulate_decay, fit_decay


def test_normalize_gamma():
    gamma = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = np.array([[0, 2, 3], [4, 0, 6], [7, 8, 0]])
    expected[0, 0] = -(2 + 3)
    expected[1, 1] = -(4 + 6)
    expected[2, 2] = -(7 + 8)
    normalized = normalize_gamma(gamma)
    np.testing.assert_array_equal(normalized, expected)


def test_single_photon_gamma_encode():
    gamma = np.array([[0, 1], [2, 0]])
    encoded = single_photon_gamma_encode(gamma)
    expected = np.array([1, 2])
    np.testing.assert_array_equal(encoded, expected)


def test_single_photon_gamma_decode():
    encoded = np.array([1, 2])
    decoded = single_photon_gamma_decode(encoded)
    expected = np.array([[0, 1], [2, 0]])
    expected[0, 0] = -1
    expected[1, 1] = -2
    np.testing.assert_array_equal(decoded, expected)


def test_gamma_encode_decode():
    np.random.seed(42)  # For reproducible tests
    size = 5  # Define the size of the gamma matrix
    random_gamma = np.random.rand(size, size)

    # Remove off-diagonal elements more than one line away from the diagonal
    for i in range(size):
        for j in range(size):
            if abs(i - j) > 1:
                random_gamma[i, j] = 0

    gamma = normalize_gamma(np.copy(random_gamma))

    encoded = single_photon_gamma_encode(gamma)
    decoded_gamma = single_photon_gamma_decode(encoded)

    np.testing.assert_array_almost_equal(decoded_gamma, gamma, decimal=5)


def test_simulate_decay():
    initial_distribution = np.array([1, 0, 0, 0])
    gamma = np.array([[0, 0.1, 0.2, 0.3], [0.1, 0, 0.1, 0.2], [0.2, 0.1, 0, 0.1], [0.3, 0.2, 0.1, 0]])
    time_resolution = 1.0
    time_length = 2.0
    results = simulate_decay(initial_distribution, gamma, time_resolution, time_length)
    assert results.shape == (2, 4)  # Checking if the results shape matches the expected number of steps and state size
