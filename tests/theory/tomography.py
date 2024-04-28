import pytest
from leeq.theory.tomography.models import *
import scipy


class TestSingleQubitModel:

    @pytest.fixture
    def model(self):
        return SingleQubitModel()

    def test_state_tomography(self, model):
        state_tomography = model.construct_state_tomography()

        rho0 = np.random.rand(2, 2) + 1.j * np.random.rand(2, 2)
        rho0 = rho0 + rho0.T.conj()
        rho0 /= np.sum(np.diag(rho0))
        # rho0 = np.zeros([3,3])
        # rho0[0,0] = 1

        probabilities = state_tomography.simulate_ideal_state_tomography_distribution(rho0)

        pauli_vector, density_matrix = state_tomography.linear_inverse_state_tomography(probabilities)

        assert np.allclose(pauli_vector[0], 1)

        density_matrix[np.abs(density_matrix) < 1e-5] = 0

        if not np.allclose(rho0, density_matrix):
            model.plot_density_matrix(rho0, title='Ground truth', base=2)
            model.plot_density_matrix(density_matrix, title='Tomography result')

        assert np.allclose(rho0, density_matrix)
        print('State tomography passed')

    def test_process_tomography(self, model):

        gate = np.asarray([
            [1, 1],
            [1, -1]
        ]) / np.sqrt(2)

        basis = model.get_basis()
        process_tomography = model.construct_process_tomography()
        basis_matrices = basis.get_basis_matrices()

        if gate is None:
            coefficient = np.random.uniform(-1, 1, 4)
            coefficient[0] = 0
            t = np.random.uniform(1, 1, 1) * 100
            gate = scipy.linalg.expm(1.j * t * np.einsum("abc,c->ab", basis_matrices, coefficient))

        ptm_truth = basis.unitary_to_ptm(gate)
        chi_truth = basis.ptm_to_chi(ptm_truth)

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
