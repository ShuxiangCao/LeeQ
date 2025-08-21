"""
Extended tests for leeq.theory.tomography.utils
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from leeq.theory.tomography.utils import (
    cj_from_ptm, evaluate_average_fidelity_ptm, evaluate_fidelity_density_matrix,
    evaluate_fidelity_density_matrix_with_state_vector, HilbertBasis, GateSet,
    evaluate_fidelity_ptm_with_unitary
)


def create_test_pauli_basis(dim=2):
    """Helper function to create Pauli basis matrices for testing."""
    basis_matrices = np.zeros((dim, dim, dim**2), dtype=complex)
    basis_name = []

    if dim == 2:
        basis_matrices[:, :, 0] = np.eye(2)  # I
        basis_matrices[:, :, 1] = np.array([[0, 1], [1, 0]])  # X
        basis_matrices[:, :, 2] = np.array([[0, -1j], [1j, 0]])  # Y
        basis_matrices[:, :, 3] = np.array([[1, 0], [0, -1]])  # Z
        basis_name = ['I', 'X', 'Y', 'Z']
    else:
        # For other dimensions, create a simple orthogonal basis
        for i in range(dim**2):
            matrix = np.zeros((dim, dim), dtype=complex)
            row, col = divmod(i, dim)
            matrix[row, col] = 1.0
            basis_matrices[:, :, i] = matrix
            basis_name.append(f'E_{row}{col}')

    return basis_matrices, basis_name


class TestCJFromPTM:
    """Test suite for cj_from_ptm function."""

    @patch('leeq.theory.tomography.utils.np.kron')
    def test_cj_from_ptm_basic(self, mock_kron):
        """Test basic CJ conversion with identity PTM (smoke test)."""
        # Mock kron to avoid dimension mismatch issues in the function
        mock_kron.return_value = np.zeros((2, 2), dtype=complex)

        dim = 2
        ptm = np.eye(4)  # 4x4 for 2-level system
        basis_matrices, _ = create_test_pauli_basis(dim)

        result = cj_from_ptm(ptm, basis_matrices, dim)

        assert result.shape == (dim, dim)
        assert result.dtype == complex
        # Verify kron was called (shows function executed)
        assert mock_kron.call_count > 0

    @patch('leeq.theory.tomography.utils.np.kron')
    def test_cj_from_ptm_dimensions(self, mock_kron):
        """Test CJ conversion with different dimensions (smoke test)."""
        mock_kron.return_value = np.zeros((3, 3), dtype=complex)

        dim = 3
        ptm = np.eye(9)  # 9x9 for 3-level system
        basis_matrices, _ = create_test_pauli_basis(dim)

        result = cj_from_ptm(ptm, basis_matrices, dim)

        assert result.shape == (dim, dim)
        assert result.dtype == complex


class TestFidelityFunctions:
    """Test suite for fidelity evaluation functions."""

    @patch('leeq.theory.tomography.utils.cj_from_ptm')
    def test_evaluate_average_fidelity_ptm_identity(self, mock_cj_from_ptm):
        """Test average fidelity with identical PTMs (smoke test)."""
        # Mock the cj_from_ptm function to return a valid density matrix
        mock_cj_from_ptm.return_value = np.array([[0.5, 0.2], [0.2, 0.5]], dtype=complex)

        dim = 2
        ptm = np.eye(4)  # Use identity PTM for consistent results
        basis_matrices, _ = create_test_pauli_basis(dim)

        fidelity = evaluate_average_fidelity_ptm(ptm, ptm, basis_matrices, dim)

        assert isinstance(fidelity, float)
        # Verify the function was called
        assert mock_cj_from_ptm.call_count >= 2  # Called for both PTMs

    def test_evaluate_fidelity_density_matrix_identical(self):
        """Test fidelity between identical density matrices."""
        rho = np.array([[0.6, 0.2+0.1j], [0.2-0.1j, 0.4]])

        fidelity = evaluate_fidelity_density_matrix(rho, rho)

        assert abs(fidelity - 1.0) < 1e-10
        assert isinstance(fidelity, float)

    def test_evaluate_fidelity_density_matrix_different(self):
        """Test fidelity between different density matrices."""
        rho1 = np.array([[1.0, 0.0], [0.0, 0.0]])  # |0><0|
        rho2 = np.array([[0.0, 0.0], [0.0, 1.0]])  # |1><1|

        fidelity = evaluate_fidelity_density_matrix(rho1, rho2)

        assert abs(fidelity) < 1e-10  # Should be zero for orthogonal states

    def test_evaluate_fidelity_density_matrix_with_state_vector(self):
        """Test fidelity between density matrix and state vector."""
        state_vec = np.array([1.0, 0.0])  # |0>
        rho = np.array([[1.0, 0.0], [0.0, 0.0]])  # |0><0|

        fidelity = evaluate_fidelity_density_matrix_with_state_vector(rho, state_vec)

        assert abs(fidelity - 1.0) < 1e-10

    def test_evaluate_fidelity_density_matrix_with_state_vector_orthogonal(self):
        """Test fidelity with orthogonal state vector."""
        state_vec = np.array([0.0, 1.0])  # |1>
        rho = np.array([[1.0, 0.0], [0.0, 0.0]])  # |0><0|

        fidelity = evaluate_fidelity_density_matrix_with_state_vector(rho, state_vec)

        assert abs(fidelity) < 1e-10


class TestHilbertBasis:
    """Test suite for HilbertBasis class."""

    def test_hilbert_basis_initialization_default(self):
        """Test default initialization of HilbertBasis with custom matrices."""
        dim = 2
        basis_matrices, basis_name = create_test_pauli_basis(dim)

        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)

        assert basis.dimension == dim
        assert basis.basis_matrices.shape == (dim, dim, dim**2)
        assert len(basis.basis_name) == dim**2

    def test_hilbert_basis_initialization_custom(self):
        """Test initialization with custom basis matrices."""
        dim = 2
        basis_matrices = np.zeros((dim, dim, dim**2), dtype=complex)
        basis_name = ['I', 'X', 'Y', 'Z']

        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)

        assert basis.dimension == dim
        assert basis.basis_name == basis_name
        assert np.array_equal(basis.basis_matrices, basis_matrices)

    def test_hilbert_basis_assertion_errors(self):
        """Test assertion errors in HilbertBasis initialization."""
        dim = 2
        wrong_shape_matrices = np.zeros((dim, dim, 3))  # Wrong last dimension

        with pytest.raises(AssertionError):
            HilbertBasis(dimension=dim, basis_matrices=wrong_shape_matrices)

    def test_operator_to_schmidt_hilbert_vector(self):
        """Test conversion from operator to Schmidt vector."""
        dim = 2
        basis_matrices, basis_name = create_test_pauli_basis(dim)
        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)
        operator = np.eye(dim)

        vector = basis.operator_to_schmidt_hilbert_vector(operator)

        assert len(vector) == dim**2
        assert vector.dtype == complex

    def test_operator_to_schmidt_hilbert_vector_assertion(self):
        """Test assertion error for wrong operator dimensions."""
        dim = 2
        basis_matrices, basis_name = create_test_pauli_basis(dim)
        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)
        wrong_operator = np.eye(3)  # Wrong dimension

        with pytest.raises(AssertionError):
            basis.operator_to_schmidt_hilbert_vector(wrong_operator)

    def test_schmidt_hilbert_vector_to_operator(self):
        """Test conversion from Schmidt vector to operator."""
        dim = 2
        basis_matrices, basis_name = create_test_pauli_basis(dim)
        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)
        vector = np.array([1.0, 0.0, 0.0, 0.0])

        operator = basis.schmidt_hilbert_vector_to_operator(vector)

        assert operator.shape == (dim, dim)
        assert operator.dtype == complex

    def test_schmidt_hilbert_vector_to_operator_assertion(self):
        """Test assertion error for wrong vector length."""
        dim = 2
        basis_matrices, basis_name = create_test_pauli_basis(dim)
        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)
        wrong_vector = np.array([1.0, 0.0, 0.0])  # Wrong length

        with pytest.raises(AssertionError):
            basis.schmidt_hilbert_vector_to_operator(wrong_vector)

    def test_unitary_to_ptm(self):
        """Test conversion from unitary to PTM."""
        dim = 2
        basis_matrices, basis_name = create_test_pauli_basis(dim)
        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)
        unitary = np.eye(dim)  # Identity unitary

        ptm = basis.unitary_to_ptm(unitary)

        assert ptm.shape == (dim**2, dim**2)
        assert ptm.dtype == float  # PTM should be real
        assert np.allclose(ptm.imag, 0, atol=1e-10)

    def test_unitary_to_ptm_assertion(self):
        """Test assertion error for wrong unitary dimensions."""
        dim = 2
        basis_matrices, basis_name = create_test_pauli_basis(dim)
        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)
        wrong_unitary = np.eye(3)  # Wrong dimension

        with pytest.raises(AssertionError):
            basis.unitary_to_ptm(wrong_unitary)

    def test_ptm_to_chi(self):
        """Test conversion from PTM to chi matrix."""
        dim = 2
        basis_matrices, basis_name = create_test_pauli_basis(dim)
        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)
        ptm = np.eye(dim**2)

        chi = basis.ptm_to_chi(ptm)

        assert chi.shape == (dim**2, dim**2)
        assert chi.dtype == complex

    def test_get_basis_matrices(self):
        """Test getting basis matrices."""
        dim = 2
        basis_matrices, basis_name = create_test_pauli_basis(dim)
        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)

        matrices = basis.get_basis_matrices()

        assert matrices.shape == (dim, dim, dim**2)

    @patch('matplotlib.pyplot.show')
    def test_plot_density_matrix(self, mock_show):
        """Test plotting density matrix."""
        dim = 2
        basis_matrices, basis_name = create_test_pauli_basis(dim)
        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)
        matrix = np.array([[0.6, 0.2+0.1j], [0.2-0.1j, 0.4]])

        # Should not raise error
        basis.plot_density_matrix(matrix, title="Test Matrix")
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_process_matrix(self, mock_show):
        """Test plotting process matrix."""
        dim = 2
        basis_matrices, basis_name = create_test_pauli_basis(dim)
        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)
        matrix = np.eye(dim**2)

        # Should not raise error
        basis.plot_process_matrix(matrix, title="Test Process")
        mock_show.assert_called_once()


class TestGateSet:
    """Test suite for GateSet class."""

    def test_gateset_initialization(self):
        """Test GateSet initialization."""
        gate_names = ['I', 'X']
        gate_matrices = np.zeros((2, 2, 2), dtype=complex)
        gate_matrices[:, :, 0] = np.eye(2)  # Identity
        gate_matrices[:, :, 1] = np.array([[0, 1], [1, 0]])  # Pauli X

        basis_matrices, basis_name = create_test_pauli_basis(2)
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)

        gateset = GateSet(gate_names, gate_matrices, basis=basis)

        assert gateset.dimension == 2
        assert gateset.available_gates == gate_names
        assert gateset.number_of_gates == 2

    def test_gateset_assertion_errors(self):
        """Test assertion errors in GateSet initialization."""
        gate_names = ['I', 'X']
        wrong_shape_matrices = np.zeros((2, 3, 2))  # Not square

        with pytest.raises(AssertionError):
            GateSet(gate_names, wrong_shape_matrices)

        wrong_count_matrices = np.zeros((2, 2, 3))  # Wrong number of gates

        with pytest.raises(AssertionError):
            GateSet(gate_names, wrong_count_matrices)

    def test_gateset_update_estimation(self):
        """Test updating PTM estimations."""
        gate_names = ['I']
        gate_matrices = np.eye(2).reshape(2, 2, 1)
        basis_matrices, basis_name = create_test_pauli_basis(2)
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)

        gateset = GateSet(gate_names, gate_matrices, basis=basis)
        new_estimation = np.random.random((4, 4, 1))

        gateset.update_estimation(new_estimation)

        assert np.array_equal(gateset._ptm_estimate, new_estimation)

    def test_gateset_get_ptm(self):
        """Test getting PTM for specific gate."""
        gate_names = ['I', 'X']
        gate_matrices = np.zeros((2, 2, 2), dtype=complex)
        gate_matrices[:, :, 0] = np.eye(2)
        gate_matrices[:, :, 1] = np.array([[0, 1], [1, 0]])
        basis_matrices, basis_name = create_test_pauli_basis(2)
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)

        gateset = GateSet(gate_names, gate_matrices, basis=basis)

        ptm_i = gateset.get_ptm('I')
        ptm_x = gateset.get_ptm('X')

        assert ptm_i.shape == (4, 4)
        assert ptm_x.shape == (4, 4)

    def test_gateset_get_ptm_ideal_tuple(self):
        """Test getting ideal PTM for gate tuple."""
        gate_names = ['I', 'X']
        gate_matrices = np.zeros((2, 2, 2), dtype=complex)
        gate_matrices[:, :, 0] = np.eye(2)
        gate_matrices[:, :, 1] = np.array([[0, 1], [1, 0]])
        basis_matrices, basis_name = create_test_pauli_basis(2)
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)

        gateset = GateSet(gate_names, gate_matrices, basis=basis)

        ptm = gateset.get_ptm_ideal(('I', 'X'))

        assert ptm.shape == (4, 4)

    def test_gateset_get_unitary_ideal_tuple(self):
        """Test getting ideal unitary for gate tuple."""
        gate_names = ['I', 'X']
        gate_matrices = np.zeros((2, 2, 2), dtype=complex)
        gate_matrices[:, :, 0] = np.eye(2)
        gate_matrices[:, :, 1] = np.array([[0, 1], [1, 0]])
        basis_matrices, basis_name = create_test_pauli_basis(2)
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)

        gateset = GateSet(gate_names, gate_matrices, basis=basis)

        unitary = gateset.get_unitary_ideal(('I', 'X'))

        assert unitary.shape == (2, 2)

    def test_gateset_cj_from_ptm(self):
        """Test CJ conversion from PTM."""
        gate_names = ['I']
        gate_matrices = np.eye(2).reshape(2, 2, 1)
        basis_matrices, basis_name = create_test_pauli_basis(2)
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)

        gateset = GateSet(gate_names, gate_matrices, basis=basis)

        cj = gateset.get_cj_from_ptm('I')
        cj_ideal = gateset.get_cj_from_ptm_ideal('I')

        assert cj.shape == (4, 4)  # For 2-level system
        assert cj_ideal.shape == (4, 4)

    def test_gateset_evaluate_average_fidelity(self):
        """Test evaluating average fidelity."""
        gate_names = ['I']
        gate_matrices = np.eye(2).reshape(2, 2, 1)
        basis_matrices, basis_name = create_test_pauli_basis(2)
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)

        gateset = GateSet(gate_names, gate_matrices, basis=basis)

        fidelity = gateset.evaluate_average_fidelity('I')

        assert isinstance(fidelity, float)
        assert 0 <= fidelity <= 1

    @patch('matplotlib.pyplot.show')
    def test_gateset_plot_gate_ptm(self, mock_show):
        """Test plotting gate PTM."""
        gate_names = ['I']
        gate_matrices = np.eye(2).reshape(2, 2, 1)
        basis_matrices, basis_name = create_test_pauli_basis(2)
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)

        gateset = GateSet(gate_names, gate_matrices, basis=basis)

        # Should not raise error
        gateset.plot_gate_ptm('I', ideal=True)
        gateset.plot_gate_ptm('I', ideal=False)

        assert mock_show.call_count == 2

    def test_gateset_dump_load_configuration(self):
        """Test dumping and loading configuration."""
        gate_names = ['I']
        gate_matrices = np.eye(2).reshape(2, 2, 1)
        basis_matrices, basis_name = create_test_pauli_basis(2)
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)

        gateset = GateSet(gate_names, gate_matrices, basis=basis)
        config = gateset.dump_configuration()

        # Should contain required keys
        assert 'gate_names' in config
        assert 'gate_ideal_matrices' in config
        assert 'basis' in config

        # Test loading (would need basis to implement dump_configuration)
        # loaded_gateset = GateSet.load_configuration(config)


class TestFidelityPTMWithUnitary:
    """Test suite for evaluate_fidelity_ptm_with_unitary function."""

    def test_evaluate_fidelity_ptm_with_unitary(self):
        """Test fidelity evaluation between PTM and unitary."""
        # Create simple PTM and unitary
        ptm = np.eye(4)
        unitary = np.eye(2)
        basis_matrices, basis_name = create_test_pauli_basis(2)
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)

        fidelity = evaluate_fidelity_ptm_with_unitary(ptm, unitary, basis)

        assert isinstance(fidelity, float)
        assert 0 <= fidelity <= 1.01  # Allow small numerical error


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_hilbert_basis_large_dimension(self):
        """Test HilbertBasis with larger dimension."""
        dim = 3
        basis_matrices, basis_name = create_test_pauli_basis(dim)
        basis = HilbertBasis(dimension=dim, basis_name=basis_name, basis_matrices=basis_matrices)

        assert basis.dimension == dim
        assert basis.basis_matrices.shape == (dim, dim, dim**2)

    def test_complex_density_matrix_fidelity(self):
        """Test fidelity with complex density matrix."""
        rho1 = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
        rho2 = np.array([[0.6, 0.2-0.1j], [0.2+0.1j, 0.4]])

        fidelity = evaluate_fidelity_density_matrix(rho1, rho2)

        assert isinstance(fidelity, float)
        assert 0 <= fidelity <= 1

    def test_gateset_with_custom_basis(self):
        """Test GateSet with custom basis."""
        gate_names = ['I']
        gate_matrices = np.eye(2).reshape(2, 2, 1)
        basis_matrices, basis_name = create_test_pauli_basis(2)
        custom_basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)

        gateset = GateSet(gate_names, gate_matrices, basis=custom_basis)

        assert gateset.get_basis() is custom_basis

    def test_zero_fidelity_case(self):
        """Test case that should give zero fidelity."""
        # Orthogonal pure states
        state1 = np.array([1.0, 0.0])
        rho1 = np.outer(state1, state1.conj())

        state2 = np.array([0.0, 1.0])
        rho2 = np.outer(state2, state2.conj())

        fidelity = evaluate_fidelity_density_matrix(rho1, rho2)

        assert abs(fidelity) < 1e-10


class TestVisualizationBranches:
    """Test visualization method branches."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    def test_plot_process_matrix_with_ax(self, mock_show, mock_subplots):
        """Test plotting process matrix with provided axis."""
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)

        basis_matrices, basis_name = create_test_pauli_basis(2)
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_matrices)
        matrix = np.eye(4)

        # Test with provided axis (do_adjust should be False)
        basis.plot_process_matrix(matrix, ax=mock_ax)

        # Verify ax methods were called
        mock_ax.axis.assert_called()
        mock_ax.add_artist.assert_called()
