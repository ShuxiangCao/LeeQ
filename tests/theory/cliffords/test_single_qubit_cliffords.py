# test_clifford_gates.py
import numpy as np
import pytest
from leeq.theory.cliffords.single_qubit_cliffords import \
    create_clifford_gates, find_inverse_C1, append_inverse_C1, pI, \
    pX, pY, pZ


def test_create_clifford_gates():
    # Test for X gate with pi phase
    expected_X = np.array([[0, -1j], [-1j, 0]])
    np.testing.assert_almost_equal(create_clifford_gates(pX, 1.0), expected_X, decimal=5)

    # Test for Y gate with pi/2 phase
    expected_Y = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                           [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    np.testing.assert_almost_equal(create_clifford_gates(pY, 0.5), expected_Y, decimal=5)

    # Test for Z gate with -pi/2 phase
    expected_Z = np.array([[np.exp(1j * np.pi / 4), 0],
                           [0, np.exp(-1j * np.pi / 4)]])
    np.testing.assert_almost_equal(create_clifford_gates(pZ, -0.5), expected_Z, decimal=5)


def test_find_inverse_C1():
    # Test finding the inverse of identity (should be identity)
    assert find_inverse_C1(pI, 'XY') == 0

    # Test finding the inverse of a non-existent gate
    x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
    assert find_inverse_C1(x_gate, 'XY') == 1


def test_append_inverse_C1():
    # Assuming that the index 1 corresponds to a gate that is its own inverse
    initial_sequence = [1]
    extended_sequence = append_inverse_C1(initial_sequence, clifford_set='XY')
    assert extended_sequence == [1, 1]  # Should append the inverse index

    # Test appending inverse to a sequence that results in identity
    identity_sequence = [0]  # Assuming index 0 corresponds to the identity gate
    extended_sequence = append_inverse_C1(identity_sequence, clifford_set='XY')
    assert extended_sequence == [0, 0]

    # Test with non-existent gate index, should return the original sequence with None
    invalid_sequence = [999]  # Assuming 999 is an invalid gate index
    with pytest.raises(IndexError):
        extended_sequence = append_inverse_C1(invalid_sequence, clifford_set='XY')

    # Make sure the extended sequence is equivalent to identity

    for i in range(10):
        random_sequence = [np.random.randint(0, 24) for _ in range(10)]
        extended_sequence = append_inverse_C1(random_sequence, clifford_set='XY')
        extended_sequence_2 = append_inverse_C1(extended_sequence, clifford_set='XY')
        assert extended_sequence_2[-1] == 0  # The last gate should be identity
