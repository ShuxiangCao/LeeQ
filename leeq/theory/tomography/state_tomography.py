import numpy as np
from .utils import *


def simulate_ideal_state_tomography_distribution(rho0: np.ndarray,
                                                 measurement_operations_matrices: np.ndarray) -> np.ndarray:
    """
    Simulate the probability distribution for an ideal state tomography.

    Args:
        rho0 (np.ndarray): The density matrix representing the quantum state.
        measurement_operations_matrices (np.ndarray): The array of unitary matrices used for measurement.

    Returns:
        np.ndarray: An array containing the real parts of the probabilities of the measurement outcomes.
    """
    probabilities = []

    # Compute probabilities for each measurement
    for i in range(measurement_operations_matrices.shape[-1]):
        u = measurement_operations_matrices[:, :, i]
        rho = u @ rho0 @ u.conj().T
        measurement = np.diag(rho)
        probabilities.append(measurement)

    probabilities = np.asarray(probabilities)
    probabilities = probabilities.transpose([1, 0])  # Transpose to align dimensions

    # Check for numerical stability by ensuring the imaginary part is negligible
    assert np.sum(np.abs(probabilities.imag)) < 1e-5

    return probabilities.real


class StandardStateTomography:
    """
    A class to handle standard quantum state tomography using a specific gate set and measurement operations.

    Attributes:
        _gate_set (GateSet): The set of gates (quantum operations).
        _basis (Basis): The basis set used in the gate set.
        dimension (int): The dimension of the Hilbert space.
        measurement_operations (list): A list of names of the measurement operations.
        measurement_operators (np.ndarray): The measurement operators.
        measurement_operations_matrices (np.ndarray): The unitary matrices for measurement operations.
        measurement_operations_ptms (np.ndarray): The Pauli transfer matrices for measurement operations.
        measurement_hilbert_schmidt_basis (np.ndarray): The Hilbert-Schmidt basis for measurements.
    """

    def __init__(self, gate_set: GateSet, measurement_operations: list, measurement_operators: np.ndarray = None):
        self._gate_set = gate_set
        self._basis = gate_set.get_basis()
        self.dimension = gate_set.dimension
        self.measurement_operations = measurement_operations

        # Initialize measurement operators as identity if none provided
        if measurement_operators is None:
            self.measurement_operators = np.zeros([self.dimension, self.dimension, self.dimension])
            for i in range(self.dimension):
                self.measurement_operators[i, i, i] = 1
        else:
            self.measurement_operators = measurement_operators

        # Stack unitary and PTM matrices for each measurement operation
        self.measurement_operations_matrices = np.dstack(
            [self._gate_set.get_unitary_ideal(name) for name in measurement_operations])

        self.measurement_operations_ptms = np.dstack(
            [self._gate_set.get_ptm_ideal(name) for name in measurement_operations])

        # Compute measurement basis in Hilbert-Schmidt space
        all_measurement_basis = np.einsum('xaz,xyv,ybz->abvz',
                                          self.measurement_operations_matrices.conj(), self.measurement_operators,
                                          self.measurement_operations_matrices)  # Index: matrix_1, matrix_2, measured_state, rotation_index

        self.measurement_hilbert_schmidt_basis = np.einsum('abvz,abw->vzw', all_measurement_basis,
                                                           self._basis.get_basis_matrices().conj()) / self.dimension

        # Ensure the measurement basis forms a complete basis
        all_basis = self.measurement_hilbert_schmidt_basis.reshape([-1, self.dimension ** 2])

        rank = np.linalg.matrix_rank(all_basis)
        expected_rank = self.dimension ** 2

        if rank != expected_rank:
            print(f"Measurement basis is not complete. Expected rank: {expected_rank}, got rank: {rank}.")

        assert rank == expected_rank, \
            (f"Measurement basis is not complete. Expected rank: {expected_rank},"
             f" got rank: {rank}.")

    def get_measurement_sequence(self) -> list:
        """Return the sequence of measurement operations."""
        return self.measurement_operations

    def get_linear_inverse_state_tomography_tensor(self) -> np.ndarray:
        """
        Generate the tensor for performing linear inverse state tomography.

        Returns:
            np.ndarray: The transformation tensor for state reconstruction.
        """
        transformation_matrix = self.measurement_hilbert_schmidt_basis.reshape(
            [-1, self.measurement_hilbert_schmidt_basis.shape[-1]])
        inv_transformation_matrix = np.linalg.inv(
            transformation_matrix.T @ transformation_matrix) @ transformation_matrix.T
        return inv_transformation_matrix.reshape(
            [self.measurement_hilbert_schmidt_basis.shape[2],
             self.measurement_hilbert_schmidt_basis.shape[0],
             self.measurement_hilbert_schmidt_basis.shape[1]]
        )

    def linear_inverse_state_tomography(self, observed_distribution: np.ndarray) -> tuple:
        """
        Perform linear inverse state tomography to estimate the state.

        Args:
            observed_distribution (np.ndarray): The observed probability distribution from measurements.

        Returns:
            tuple: A tuple containing the estimated state vector and the corresponding operator.
        """
        inv_transformation_tensor = self.get_linear_inverse_state_tomography_tensor()
        vector = np.einsum("cab,ab->c", inv_transformation_tensor, observed_distribution)
        return vector, self._basis.schmidt_hilbert_vector_to_operator(vector)

    def simulate_ideal_state_tomography_distribution(self, rho0: np.ndarray) -> np.ndarray:
        """
        Wrapper method to simulate the state tomography distribution using the ideal measurement operations matrices.

        Args:
            rho0 (np.ndarray): The density matrix of the quantum state to be simulated.

        Returns:
            np.ndarray: The simulated probability distribution.
        """
        return simulate_ideal_state_tomography_distribution(rho0=rho0,
                                                            measurement_operations_matrices=self.measurement_operations_matrices)
