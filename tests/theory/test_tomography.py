import pytest
from leeq.theory.tomography.models import *
import scipy


def _check_state_tomography(model, dimension):
    state_tomography = model.construct_state_tomography()

    rho0 = np.random.rand(dimension, dimension) + 1.j * np.random.rand(dimension, dimension)
    rho0 = rho0 + rho0.T.conj()
    rho0 /= np.sum(np.diag(rho0))

    probabilities = state_tomography.simulate_ideal_state_tomography_distribution(rho0)

    pauli_vector, density_matrix = state_tomography.linear_inverse_state_tomography(probabilities)

    assert np.allclose(pauli_vector[0], 1)

    density_matrix[np.abs(density_matrix) < 1e-5] = 0

    if not np.allclose(rho0, density_matrix):
        model.plot_density_matrix(rho0, title='Ground truth', base=2)
        model.plot_density_matrix(density_matrix, title='Tomography result')

    assert np.allclose(rho0, density_matrix)


def _check_process_tomography(model, dimension, gate=None):
    basis = model.get_basis()
    process_tomography = model.construct_process_tomography()
    basis_matrices = basis.get_basis_matrices()

    if gate is None:
        coefficient = np.random.uniform(-1, 1, dimension ** 2)
        coefficient[0] = 0
        t = np.random.uniform(0, 1, 1) * 100
        gate = scipy.linalg.expm(1.j * t * np.einsum("abc,c->ab", basis_matrices, coefficient))

    ptm_truth = basis.unitary_to_ptm(gate)
    basis.ptm_to_chi(ptm_truth)

    probabilities = process_tomography.simulate_ideal_process_tomography_distribution(gate)

    ptm = process_tomography.linear_inverse_process_tomography(probabilities)

    passed = np.allclose(ptm, ptm_truth)

    if not passed:
        from matplotlib import pyplot as plt
        model.plot_process_matrix(ptm_truth, title='PTM ground truth')
        plt.show()
        model.plot_process_matrix(ptm, title='Tomography result')
        plt.show()

    assert passed


class TestSingleQubitModel:

    @pytest.fixture
    def model(self):
        return SingleQubitModel()

    def test_state_tomography(self, model):
        _check_state_tomography(model, 2)

    def test_process_tomography(self, model):
        _check_process_tomography(model, 2)


class TestMultiQubitModel:

    @pytest.mark.parametrize("n_qubit", [1, 2, 3, ])
    def test_state_tomography(self, n_qubit):
        model = MultiQubitModel(number_of_qubit=n_qubit)
        _check_state_tomography(model, 2 ** n_qubit)

    @pytest.mark.parametrize("n_qubit", [1, 2, 3, ])
    def test_process_tomography(self, n_qubit):
        model = MultiQubitModel(number_of_qubit=n_qubit)
        _check_process_tomography(model, 2 ** n_qubit)


class TestMultiQutritModel:

    @pytest.mark.parametrize("n_qubit", [1, 2])
    def test_state_tomography(self, n_qubit):
        model = MultiQutritModel(number_of_qutrit=n_qubit)
        _check_state_tomography(model, 3 ** n_qubit)

    @pytest.mark.parametrize("n_qubit", [1, 2])
    def test_process_tomography(self, n_qubit):
        model = MultiQutritModel(number_of_qutrit=n_qubit)
        _check_process_tomography(model, 3 ** n_qubit)


class TestMultiQuditModel:
    @pytest.mark.parametrize("n_qubit", [1])  # , 2
    def test_state_tomography(self, n_qubit):
        model = MultiQuditModel(number_of_qudits=n_qubit)
        _check_state_tomography(model, 4 ** n_qubit)

    @pytest.mark.parametrize("n_qubit", [1])  # , 2
    def test_process_tomography(self, n_qubit):
        model = MultiQuditModel(number_of_qudits=n_qubit)
        _check_process_tomography(model, 4 ** n_qubit)
