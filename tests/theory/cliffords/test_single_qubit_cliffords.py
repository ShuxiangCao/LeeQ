# test_clifford_gates.py
import numpy as np
import pytest
from leeq.theory.cliffords.single_qubit_cliffords import \
    create_clifford_gates, find_inverse_C1, append_inverse_C1, pI, \
    pX, pY, pZ, get_c1, get_clifford_from_id, C1, NC1, TOLERANCE


def test_create_clifford_gates():
    # Test for X gate with pi phase
    expected_X = np.array([[0, -1j], [-1j, 0]])
    np.testing.assert_almost_equal(
        create_clifford_gates(
            pX, 1.0), expected_X, decimal=5)

    # Test for Y gate with pi/2 phase
    expected_Y = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                           [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    np.testing.assert_almost_equal(
        create_clifford_gates(
            pY, 0.5), expected_Y, decimal=5)

    # Test for Z gate with -pi/2 phase
    expected_Z = np.array([[np.exp(1j * np.pi / 4), 0],
                           [0, np.exp(-1j * np.pi / 4)]])
    np.testing.assert_almost_equal(
        create_clifford_gates(
            pZ, -0.5), expected_Z, decimal=5)


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
    # Assuming index 0 corresponds to the identity gate
    identity_sequence = [0]
    extended_sequence = append_inverse_C1(identity_sequence, clifford_set='XY')
    assert extended_sequence == [0, 0]

    # Test with non-existent gate index, should return the original sequence
    # with None
    invalid_sequence = [999]  # Assuming 999 is an invalid gate index
    with pytest.raises(IndexError):
        extended_sequence = append_inverse_C1(
            invalid_sequence, clifford_set='XY')

    # Make sure the extended sequence is equivalent to identity

    for i in range(10):
        random_sequence = [np.random.randint(0, 24) for _ in range(10)]
        extended_sequence = append_inverse_C1(
            random_sequence, clifford_set='XY')
        extended_sequence_2 = append_inverse_C1(
            extended_sequence, clifford_set='XY')
        assert extended_sequence_2[-1] == 0  # The last gate should be identity


class TestCliffordGateRetrieval:
    """Test suite for Clifford gate retrieval functions."""

    def test_get_c1_valid_indices(self):
        """Test retrieving Clifford gates with valid indices."""
        for i in range(NC1):
            gate = get_c1(i)
            assert gate.shape == (2, 2)
            assert gate.dtype == complex

    def test_get_c1_boundary_cases(self):
        """Test retrieving gates at boundary indices."""
        # Test first gate (identity)
        first_gate = get_c1(0)
        np.testing.assert_array_almost_equal(first_gate, pI)

        # Test last gate
        last_gate = get_c1(NC1 - 1)
        assert last_gate.shape == (2, 2)

    def test_get_c1_invalid_index(self):
        """Test error handling for invalid indices."""
        with pytest.raises(IndexError):
            get_c1(NC1)  # Index too large

        # Note: Python allows negative indices, so -1 is valid (refers to last element)


class TestCliffordMapping:
    """Test suite for Clifford mapping functions."""

    def test_get_clifford_from_id_all_valid(self):
        """Test getting Clifford sequences for all valid IDs."""
        for clifford_id in range(24):  # All 24 single-qubit Cliffords
            sequence = get_clifford_from_id(clifford_id)
            assert isinstance(sequence, list)
            assert len(sequence) >= 1
            assert all(isinstance(gate, str) for gate in sequence)

    def test_get_clifford_from_id_specific_cases(self):
        """Test specific known Clifford mappings."""
        # Identity should map to ['I']
        assert get_clifford_from_id(0) == ['I']

        # X gate should map to ['X']
        assert get_clifford_from_id(1) == ['X']

        # Y gate should map to ['Y']
        assert get_clifford_from_id(2) == ['Y']

    def test_get_clifford_from_id_invalid_index(self):
        """Test error handling for invalid Clifford IDs."""
        with pytest.raises(AssertionError):
            get_clifford_from_id(24)  # Too large

        with pytest.raises(AssertionError):
            get_clifford_from_id(-1)  # Negative

        with pytest.raises(AssertionError):
            get_clifford_from_id(100)  # Way too large


class TestErrorHandling:
    """Test suite for error handling in Clifford functions."""

    def test_find_inverse_C1_invalid_clifford_set(self):
        """Test error handling for invalid Clifford set."""
        with pytest.raises(ValueError, match="Invalid Clifford set"):
            find_inverse_C1(pI, 'INVALID')

    def test_find_inverse_C1_not_implemented_VZX(self):
        """Test ValueError for VZX Clifford set (not implemented yet)."""
        with pytest.raises(ValueError, match="Invalid Clifford set"):
            find_inverse_C1(pI, 'VZX')

    def test_append_inverse_C1_invalid_clifford_set(self):
        """Test error handling for invalid Clifford set in append_inverse_C1."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            append_inverse_C1([0], clifford_set='VZX')

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            append_inverse_C1([0], clifford_set='INVALID')


class TestCliffordGateProperties:
    """Test mathematical properties of Clifford gates."""

    def test_all_gates_are_unitary(self):
        """Test that all Clifford gates are unitary."""
        for i in range(NC1):
            gate = get_c1(i)
            # Check if U† @ U = I
            identity_check = gate.conj().T @ gate
            np.testing.assert_array_almost_equal(identity_check, pI, decimal=10)

    def test_all_gates_have_unit_determinant(self):
        """Test that all Clifford gates have unit determinant."""
        for i in range(NC1):
            gate = get_c1(i)
            det = np.linalg.det(gate)
            assert abs(abs(det) - 1.0) < TOLERANCE

    def test_pauli_matrices_properties(self):
        """Test properties of the Pauli matrices."""
        # Test Pauli matrix squares
        np.testing.assert_array_almost_equal(pX @ pX, pI)
        np.testing.assert_array_almost_equal(pY @ pY, pI)
        np.testing.assert_array_almost_equal(pZ @ pZ, pI)

        # Test commutation relations
        np.testing.assert_array_almost_equal(pX @ pY, 1j * pZ)
        np.testing.assert_array_almost_equal(pY @ pZ, 1j * pX)
        np.testing.assert_array_almost_equal(pZ @ pX, 1j * pY)


class TestInverseFinding:
    """Test suite for inverse finding functionality."""

    def test_find_inverse_for_all_cliffords(self):
        """Test finding inverses for all Clifford gates."""
        for i in range(NC1):
            gate = get_c1(i)
            inverse_index = find_inverse_C1(gate, 'XY')
            inverse_gate = get_c1(inverse_index)

            # Check that gate @ inverse = identity
            product = gate @ inverse_gate

            # Normalize by first element if it's not zero
            if not np.isclose(product[0, 0], 0):
                product /= product[0, 0]

            np.testing.assert_array_almost_equal(product, pI, decimal=8)

    def test_find_inverse_identity(self):
        """Test that identity is its own inverse."""
        inverse_index = find_inverse_C1(pI, 'XY')
        assert inverse_index == 0

    def test_find_inverse_pauli_gates(self):
        """Test inverses of basic Pauli gates."""
        # X is its own inverse
        x_inverse = find_inverse_C1(pX, 'XY')
        x_gate = get_c1(x_inverse)
        product_x = pX @ x_gate
        # Normalize phase if needed
        if not np.isclose(product_x[0, 0], 0):
            product_x /= product_x[0, 0]
        np.testing.assert_array_almost_equal(product_x, pI, decimal=8)

        # Y is its own inverse
        y_inverse = find_inverse_C1(pY, 'XY')
        y_gate = get_c1(y_inverse)
        product_y = pY @ y_gate
        # Normalize phase if needed
        if not np.isclose(product_y[0, 0], 0):
            product_y /= product_y[0, 0]
        np.testing.assert_array_almost_equal(product_y, pI, decimal=8)

    def test_find_inverse_tolerance(self):
        """Test inverse finding with matrices close to Cliffords."""
        # Create a matrix very close to identity
        almost_identity = pI + 1e-12 * np.random.random((2, 2))
        inverse_index = find_inverse_C1(almost_identity, 'XY')
        assert inverse_index == 0  # Should still find identity as closest

    def test_find_inverse_failure_case(self):
        """Test inverse finding failure for non-Clifford matrix."""
        # Create a matrix that's not close to any Clifford
        random_matrix = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
        random_matrix = random_matrix / np.linalg.norm(random_matrix)  # Normalize

        with pytest.raises(ValueError, match="Inverse search failed"):
            find_inverse_C1(random_matrix, 'XY')


class TestSequenceOperations:
    """Test suite for sequence operations."""

    def test_append_inverse_empty_sequence(self):
        """Test appending inverse to empty sequence."""
        extended = append_inverse_C1([], clifford_set='XY')
        assert extended == [0]  # Should be identity

    def test_append_inverse_single_element(self):
        """Test appending inverse to single-element sequences."""
        for i in range(min(5, NC1)):  # Test first few gates
            sequence = [i]
            extended = append_inverse_C1(sequence, clifford_set='XY')
            assert len(extended) == 2

            # Verify the sequence becomes identity
            result_matrix = pI
            for gate_index in extended:
                result_matrix = get_c1(gate_index) @ result_matrix

            # Should be close to identity
            if not np.isclose(result_matrix[0, 0], 0):
                result_matrix /= result_matrix[0, 0]
            np.testing.assert_array_almost_equal(result_matrix, pI, decimal=8)

    def test_append_inverse_multiple_elements(self):
        """Test appending inverse to multi-element sequences."""
        test_sequences = [
            [1, 2],  # X, Y
            [0, 1, 2],  # I, X, Y
            [5, 6, 7],  # Some other combinations
        ]

        for sequence in test_sequences:
            extended = append_inverse_C1(sequence, clifford_set='XY')
            assert len(extended) == len(sequence) + 1

            # Verify the extended sequence gives identity
            result_matrix = pI
            for gate_index in extended:
                result_matrix = get_c1(gate_index) @ result_matrix

            # Should be close to identity
            if not np.isclose(result_matrix[0, 0], 0):
                result_matrix /= result_matrix[0, 0]
            np.testing.assert_array_almost_equal(result_matrix, pI, decimal=8)

    def test_append_inverse_test_inverse_parameter(self):
        """Test the test_inverse parameter functionality."""
        sequence = [1, 2]  # X, Y

        # Test with test_inverse=True (should not change functionality)
        extended_true = append_inverse_C1(sequence, clifford_set='XY', test_inverse=True)
        extended_false = append_inverse_C1(sequence, clifford_set='XY', test_inverse=False)

        # Results should be the same
        assert extended_true == extended_false


class TestGateCompositions:
    """Test specific gate compositions and their properties."""

    def test_specific_clifford_compositions(self):
        """Test that specific Clifford gate compositions match expected gates."""
        # Test some known compositions from the _single_qubit_clifford_map

        # C1[7] should be Y @ X
        expected_7 = pY @ pX
        actual_7 = get_c1(7)
        # Normalize phases (Clifford gates can have global phases)
        if not np.isclose(expected_7[0, 0], 0):
            expected_7 /= expected_7[0, 0]
        if not np.isclose(actual_7[0, 0], 0):
            actual_7 /= actual_7[0, 0]
        np.testing.assert_array_almost_equal(actual_7, expected_7, decimal=10)

    def test_all_cliffords_are_distinct(self):
        """Test that all 24 Clifford gates are distinct."""
        gates = [get_c1(i) for i in range(NC1)]

        for i in range(NC1):
            for j in range(i + 1, NC1):
                gate_i = gates[i]
                gate_j = gates[j]

                # Normalize phases
                if not np.isclose(gate_i[0, 0], 0):
                    gate_i_norm = gate_i / gate_i[0, 0]
                else:
                    gate_i_norm = gate_i

                if not np.isclose(gate_j[0, 0], 0):
                    gate_j_norm = gate_j / gate_j[0, 0]
                else:
                    gate_j_norm = gate_j

                # Gates should not be equal (allowing for small numerical errors)
                assert not np.allclose(gate_i_norm, gate_j_norm, atol=1e-10), f"Gates {i} and {j} are too similar"


class TestRandomizedTesting:
    """Test suite with randomized testing for robustness."""

    def test_random_sequence_inverse_property(self):
        """Test inverse property with random sequences."""
        np.random.seed(42)  # For reproducible tests

        for _ in range(20):  # Test 20 random sequences
            seq_length = np.random.randint(1, 10)
            sequence = [np.random.randint(0, NC1) for _ in range(seq_length)]

            extended = append_inverse_C1(sequence, clifford_set='XY')

            # Compose all gates in the extended sequence
            result = pI
            for gate_idx in extended:
                result = get_c1(gate_idx) @ result

            # Should be close to identity (up to phase)
            if not np.isclose(result[0, 0], 0):
                result /= result[0, 0]
            np.testing.assert_array_almost_equal(result, pI, decimal=8)

    def test_repeated_inverse_application(self):
        """Test applying inverse operation repeatedly."""
        np.random.seed(123)

        for _ in range(10):
            # Start with random sequence
            sequence = [np.random.randint(0, NC1) for _ in range(3)]

            # Apply inverse twice
            extended_once = append_inverse_C1(sequence, clifford_set='XY')
            extended_twice = append_inverse_C1(extended_once, clifford_set='XY')

            # The final gate should be identity (index 0)
            assert extended_twice[-1] == 0
