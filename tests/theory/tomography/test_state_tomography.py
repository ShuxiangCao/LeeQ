import numpy as np
import pytest

from leeq.theory.tomography.state_tomography import (
    StandardStateTomography,
    simulate_ideal_state_tomography_distribution,
)
from leeq.theory.tomography.utils import GateSet, HilbertBasis


def test_simulate_ideal_state_tomography_distribution_rotates_before_measurement():
    rho_zero = np.array([[1, 0], [0, 0]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    bit_flip = np.array([[0, 1], [1, 0]], dtype=complex)
    measurement_operations = np.dstack([identity, bit_flip])

    probabilities = simulate_ideal_state_tomography_distribution(rho_zero, measurement_operations)

    np.testing.assert_allclose(
        probabilities,
        np.array(
            [
                [1, 0],
                [0, 1],
            ]
        ),
    )


def test_simulate_ideal_state_tomography_distribution_rejects_complex_probabilities():
    rho = np.array([[1j, 0], [0, 0]], dtype=complex)
    measurement_operations = np.dstack([np.eye(2, dtype=complex)])

    with pytest.raises(ValueError, match="non-negligible imaginary"):
        simulate_ideal_state_tomography_distribution(rho, measurement_operations)


def test_standard_state_tomography_rejects_incomplete_measurement_basis():
    identity = np.eye(2, dtype=complex)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    basis = HilbertBasis(
        dimension=2,
        basis_name=["I", "X", "Y", "Z"],
        basis_matrices=np.dstack([identity, pauli_x, pauli_y, pauli_z]),
    )
    gate_set = GateSet(
        gate_names=["I"],
        gate_ideal_matrices=np.dstack([identity]),
        basis=basis,
    )

    with pytest.raises(ValueError, match="Measurement basis is not complete"):
        StandardStateTomography(gate_set=gate_set, measurement_operations=["I"])
