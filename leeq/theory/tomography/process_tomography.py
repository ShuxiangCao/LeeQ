from typing import List, Any, Tuple
import numpy as np
from .utils import *
from .state_tomography import StandardStateTomography, simulate_ideal_state_tomography_distribution


def simulate_ideal_process_tomography_distribution(unitary: np.ndarray,
                                                   initial_state_density_matrix: np.ndarray,
                                                   preparation_operations_matrices: np.ndarray,
                                                   measurement_operations_matrices: np.ndarray
                                                   ) -> np.ndarray:
    """
    Simulates the ideal process tomography distribution for a given unitary operation,
    initial state density matrix, preparation, and measurement operations.

    Args:
    - unitary (np.ndarray): The unitary matrix representing the quantum process.
    - initial_state_density_matrix (np.ndarray): The density matrix of the initial quantum state.
    - preparation_operations_matrices (np.ndarray): A stack of matrices for state preparation.
    - measurement_operations_matrices (np.ndarray): A stack of matrices for measurement operations.

    Returns:
    - np.ndarray: A 3D array where each "slice" along the third axis corresponds to the probability distribution
                  resulting from a different preparation operation followed by the quantum process and measurements.
    """
    rho0 = initial_state_density_matrix
    probs = []

    for i in range(preparation_operations_matrices.shape[2]):
        prepare_u = preparation_operations_matrices[:, :, i]
        rho = prepare_u @ rho0 @ prepare_u.conj().T
        rho = unitary @ rho @ unitary.conj().T
        probs.append(simulate_ideal_state_tomography_distribution(rho,
                                                                  measurement_operations_matrices=measurement_operations_matrices))

    return np.dstack(probs)


class StandardProcessTomography(StandardStateTomography):
    """
    Implements standard process tomography by extending the capabilities of standard state tomography.
    It integrates preparation and measurement operations specific to process tomography.
    """

    def __init__(self, gate_set: GateSet, measurement_operations, preparation_operations, initial_state_density_matrix,
                 measurement_operators=None):
        """
        Initializes a new instance of the StandardProcessTomography class.

        Args:
        - gate_set (GateSet): The set of quantum gates (unitaries and PTMs).
        - measurement_operations: The set of measurement operations.
        - preparation_operations: The set of preparation operations.
        - initial_state_density_matrix (np.ndarray): The initial state's density matrix.
        - measurement_operators (optional): Specific measurement operators to be used.
        """
        super().__init__(gate_set=gate_set, measurement_operations=measurement_operations,
                         measurement_operators=measurement_operators)
        self._preparation_operations = preparation_operations
        self._initial_state_density_matrix = initial_state_density_matrix

        # Compute the unitary matrices for the preparation operations.
        self.preparation_operations_matrices = np.dstack(
            [self._gate_set.get_unitary_ideal(name) for name in preparation_operations])

        # Compute the Pauli Transfer Matrices (PTMs) for the preparation operations.
        self.preparation_operations_ptms = np.dstack(
            [self._gate_set.get_ptm_ideal(name) for name in preparation_operations])

        # Pre-calculate the prepared density matrices in Hilbert-Schmidt basis for all preparation operations.
        all_prepared_dm = np.einsum('axz,xy,byz->abz',
                                    self.preparation_operations_matrices, self._initial_state_density_matrix,
                                    self.preparation_operations_matrices.conj())

        # all_prepared_dm = np.concatenate([np.eye(all_prepared_dm.shape[0])[:, :, np.newaxis], all_prepared_dm], axis=-1)

        self.preparation_hilbert_schmidt_basis = np.einsum('abz,abw->zw', all_prepared_dm,
                                                           self._basis.get_basis_matrices().conj())

        # Ensure the matrix has full rank.
        assert np.linalg.matrix_rank(self.preparation_hilbert_schmidt_basis) == self.dimension ** 2

    def get_preparation_sequence(self) -> List[Any]:
        """ Returns the sequence of preparation operations. """
        return self._preparation_operations

    def get_preparation_inverse_tensor(self) -> np.ndarray:
        """ Computes the inverse transformation matrix for preparation operations in the Hilbert-Schmidt basis. """
        inv_transformation_matrix = np.linalg.inv(self.preparation_hilbert_schmidt_basis)
        return inv_transformation_matrix

    def linear_inverse_process_tomography(self, observed_distribution: np.ndarray) -> np.ndarray:
        """
        Performs linear inverse process tomography using the observed distribution to estimate the process tensors.

        Args:
        - observed_distribution (np.ndarray): The observed distribution from experiment or simulation.

        Returns:
        - np.ndarray: The estimated process tensor matrices (PTMs).
        """
        inv_measurement_tensor = self.get_linear_inverse_state_tomography_tensor()
        inv_preparation_tensor = self.get_preparation_inverse_tensor()

        pauli_vector_on_preparation = np.einsum("cab,abz->cz", inv_measurement_tensor, observed_distribution)

        ## Add identity element to the Pauli vector
        # pauli_vector_on_preparation = np.concatenate(
        #    [np.array([1] * pauli_vector_on_preparation.shape[0])[:, np.newaxis], pauli_vector_on_preparation], axis=1)

        ptms = np.einsum("zc,xc->xz", pauli_vector_on_preparation, inv_preparation_tensor)

        assert np.sum(np.abs(ptms.imag)) < 1e-5
        return ptms.real

    def simulate_ideal_process_tomography_distribution(self, unitary: np.ndarray) -> np.ndarray:
        """
        Simulates the ideal process tomography distribution using the provided unitary operation.

        Args:
        - unitary (np.ndarray): The unitary operation of the quantum process.

        Returns:
        - np.ndarray: A 3D array of probabilities representing the ideal process tomography distribution.
        """
        return simulate_ideal_process_tomography_distribution(
            unitary=unitary,
            initial_state_density_matrix=self._initial_state_density_matrix,
            preparation_operations_matrices=self.preparation_operations_matrices,
            measurement_operations_matrices=self.measurement_operations_matrices
        )
