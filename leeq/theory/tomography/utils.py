import numpy as np
from typing import Optional, List, Tuple, Union
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
from .gellman import generate_gellmann_basis
import scipy as sp


def cj_from_ptm(ptm: np.ndarray, basis_matrices: np.ndarray, dim: int) -> np.ndarray:
    """
    Converts a Pauli Transfer Matrix (PTM) to a Choi-Jamiolkowski (CJ) representation.

    Args:
        ptm (np.ndarray): The PTM to convert.
        basis_matrices (np.ndarray): The basis matrices of the Hilbert space.
        dim (int): The dimension of the Hilbert space.

    Returns:
        np.ndarray: The CJ representation of the PTM.
    """

    rho = np.zeros([dim, dim], dtype=complex)
    for i in range(ptm.shape[0]):
        for j in range(ptm.shape[1]):
            rho += ptm[i, j] * np.kron(basis_matrices[:, :, j].T, basis_matrices[:, :, i])
    rho /= dim
    return rho


def evaluate_average_fidelity_ptm(ptm_estimated: np.ndarray, ptm_ideal: np.ndarray, basis_matrices: np.ndarray,
                                  dim: int) -> float:
    """
    Evaluates the average fidelity between the estimated and ideal CJ representations of a gate's PTM.

    Args:
        ptm_estimated (np.ndarray): The estimated PTM.
        ptm_ideal (np.ndarray): The ideal PTM.
        basis_matrices (np.ndarray): The basis matrices of the Hilbert space.
        dim (int): The dimension of the Hilbert space.

    Returns:
        float: The average fidelity value, real part only.
    """
    cj_estimated = cj_from_ptm(ptm_estimated, basis_matrices, dim)
    cj_ideal = cj_from_ptm(ptm_ideal, basis_matrices, dim)
    sqrt_cj = sp.linalg.sqrtm(cj_estimated)
    entanglement_fidelity = np.trace(sp.linalg.sqrtm(sqrt_cj @ cj_ideal @ sqrt_cj)) ** 2

    hilbert_space_dimension = ptm_estimated.shape[0] ** (1 / 2)  # One sqrt to move ptm to unitary size

    averaged_fidelity = (hilbert_space_dimension * entanglement_fidelity + 1) / (hilbert_space_dimension + 1)

    return averaged_fidelity.real


def evaluate_fidelity_density_matrix(rho_estimated: np.ndarray, rho_ideal: np.ndarray) -> float:
    """
    Evaluates the fidelity between the estimated and ideal density matrices.

    Args:
        rho_estimated (np.ndarray): The estimated density matrix.
        rho_ideal (np.ndarray): The ideal density matrix.

    Returns:
        float: The fidelity value, real part only.
    """
    sqrt_rho = sp.linalg.sqrtm(rho_estimated)
    fidelity = np.trace(sp.linalg.sqrtm(sqrt_rho @ rho_ideal @ sqrt_rho)) ** 2
    return fidelity.real


def evaluate_fidelity_density_matrix_with_state_vector(rho_estimated: np.ndarray, state_vec_ideal: np.ndarray) -> float:
    """
    Evaluates the fidelity between the estimated density matrix and an ideal state vector.

    Args:
        rho_estimated (np.ndarray): The estimated density matrix.
        state_vec_ideal (np.ndarray): The ideal state vector.

    Returns:
        float: The fidelity value, real part only.
    """
    rho_ideal = np.outer(state_vec_ideal, state_vec_ideal.conj())
    return evaluate_fidelity_density_matrix(rho_estimated, rho_ideal)


class HilbertBasis(object):
    def __init__(self, dimension: int, basis_name: Optional[List[str]] = None,
                 basis_matrices: Optional[np.ndarray] = None):
        """
        Initializes the HilbertBasis with a specified dimension and optionally a basis name and matrices.

        Args:
            dimension (int): The dimension of the Hilbert space.
            basis_name (Optional[List[str]]): Names of the basis matrices.
            basis_matrices (Optional[np.ndarray]): Array of basis matrices with dimensions [dimension, dimension, dimension^2].
        """
        self.dimension = dimension

        if basis_name is None and basis_matrices is None:
            # Generate default basis matrices and names if none are provided
            basis_matrices = generate_gellmann_basis(dimension)
            basis_name = [f"$G_{i}$" for i in range(dimension ** 2)]
        else:
            # Ensure that the provided matrices are valid for the given dimension
            assert basis_matrices.shape[-1] == dimension ** 2
            assert basis_matrices.shape[0] == basis_matrices.shape[1] == dimension

        self.basis_name = basis_name
        self.basis_matrices = basis_matrices

    def operator_to_schmidt_hilbert_vector(self, operator: np.ndarray) -> np.ndarray:
        """
        Converts an operator in Hilbert space to a vector in Schmidt form.

        Args:
            operator (np.ndarray): The operator to convert, with shape [dimension, dimension].

        Returns:
            np.ndarray: The Schmidt vector representation of the operator.
        """
        assert operator.shape[0] == operator.shape[1] == self.dimension
        vector = np.einsum("ab,abw->w", operator, self.basis_matrices.conj())
        return vector

    def schmidt_hilbert_vector_to_operator(self, vector: np.ndarray) -> np.ndarray:
        """
        Converts a vector in Schmidt form back to an operator in Hilbert space.

        Args:
            vector (np.ndarray): The Schmidt vector.

        Returns:
            np.ndarray: The corresponding operator.
        """
        assert len(vector) == self.dimension ** 2
        density_matrix = np.einsum("c,abc->ab", vector, self.basis_matrices) / self.dimension
        return density_matrix

    def unitary_to_ptm(self, unitary: np.ndarray) -> np.ndarray:
        """
        Converts a unitary matrix to a Pauli Transfer Matrix (PTM).

        Args:
            unitary (np.ndarray): The unitary matrix.

        Returns:
            np.ndarray: The corresponding PTM.
        """
        assert unitary.shape[0] == unitary.shape[1] == self.dimension, ("Unitary matrix must be square and equal to "
                                                                        f"the dimension {self.dimension}. Got shape: ",
                                                                        unitary.shape)
        transformed_result = np.einsum("abc,ad,be->dec", self.basis_matrices, unitary.conjugate(), unitary)
        ptm = np.einsum("abc,baf->fc", transformed_result, self.basis_matrices) / self.dimension

        assert np.abs(ptm.imag).max() < 1e-10
        return ptm.real

    def ptm_to_chi(self, ptm: np.ndarray) -> np.ndarray:
        """
        Converts a Pauli Transfer Matrix (PTM) to a process matrix χ.

        Args:
            ptm (np.ndarray): The PTM matrix.

        Returns:
            np.ndarray: The process matrix χ.
        """
        transformed_tensor = np.einsum("ab,ija,klb->ijkl", ptm, self.basis_matrices, self.basis_matrices)
        transformed_tensor = np.einsum("ijkl,lix->xjk", transformed_tensor, self.basis_matrices)
        transformed_tensor = np.einsum("xjk,jky->yx", transformed_tensor, self.basis_matrices)
        return transformed_tensor / self.dimension ** 3

    def get_basis_matrices(self) -> np.ndarray:
        """
        Returns the basis matrices.

        Returns:
            np.ndarray: The basis matrices.
        """
        return self.basis_matrices

    def plot_density_matrix(self, m: np.ndarray, title: Optional[str] = None, base: int = 2):
        """
        Plots a density matrix using color-coded phases and magnitudes.

        Args:
            m (np.ndarray): The density matrix to plot.
            title (Optional[str]): Title of the plot.
            base_value (int): The base of the qudit space. Qubit is 2, qudit is 3, etc.
        """

        from matplotlib import pyplot as plt
        from matplotlib.colors import hsv_to_rgb
        import matplotlib.patches as mpatches

        nx = m.shape[0]
        ny = m.shape[1]
        qubit_number = int(np.round(np.log(nx) / np.log(base)))

        fig, ax = plt.subplots()

        ax.axis("scaled")
        ax.patch.set_facecolor("white")

        mflat = m.flatten()
        ones = np.ones(mflat.shape[0])
        hues = np.arctan2(-mflat.imag, -mflat.real) / (2.0 * np.pi) + 0.5
        colors = hsv_to_rgb(np.dstack((hues, ones, ones)))

        # sizes_lin = (2.0**0.5)*np.abs(mflat)
        sizes_area = (2.0 * np.abs(mflat)) ** 0.5

        sizes = sizes_area

        i = 0
        for ix in range(nx):
            for iy in range(ny):
                ax.add_artist(
                    mpatches.RegularPolygon(
                        xy=(ix, iy),
                        numVertices=4,
                        orientation=np.pi / 4,
                        color=colors[0, i],
                        radius=0.5 * sizes[i],
                        linestyle="None",
                        linewidth=0.0,
                    )
                )
                i += 1

        def int2bin(integer, digits):

            label = np.base_repr(integer, base=base)

            label = "0" * (digits - len(label)) + label

            return label

            if integer >= 0:
                return bin(integer)[2:].zfill(digits)
            else:
                return bin(2 ** digits + integer)[2:]

        labels = [int2bin(x, qubit_number) for x in range(base ** qubit_number)]
        ax.xaxis.set_ticks(np.arange(nx))
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticks(np.arange(ny))
        ax.yaxis.set_ticklabels(labels)
        ax.xaxis.set_tick_params(labelright="on")
        ax.yaxis.set_tick_params(labeltop="on")

        ax.axis([-0.5, -0.5 + nx, -0.5 + nx, -0.5])

        legendhandles = []

        leghues = np.linspace(0, 1, 17)
        ones = np.ones(17)
        legendcolors = hsv_to_rgb(np.dstack((leghues, ones, ones)))
        for i in range(17):
            legendhandles += [
                mpatches.Patch(color=legendcolors[0, i]),
            ]

        # used to  be ax.legend
        fig.subplots_adjust(
            left=0.05, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1, right=0.85
        )

        fig.legend(
            legendhandles,
            [
                "+ 1",
                "",
                "",
                "",
                "+ i",
                "",
                "",
                "",
                "- 1",
                "",
                "",
                "",
                "- i",
                "",
                "",
                "",
                "+1",
            ],
            loc="right",
            bbox_to_anchor=(1.0, 0.5),
        )

        if title is not None:
            ax.set_title(title)

        for iy in range(ny + 1):
            ax.axhline(y=iy - 0.5, ls=":", color="k")
        for ix in range(nx + 1):
            ax.axvline(x=ix - 0.5, ls=":", color="k")

        plt.show()

    def plot_process_matrix(self, m: np.ndarray, ax: Optional[plt.Axes] = None, title: Optional[str] = None,
                            base: int = 2):
        """
        Plots a process matrix using color-coded phases and magnitudes.

        Args:
            m (np.ndarray): The process matrix to plot.
            ax (Optional[plt.Axes]): Axis on which to plot the matrix. If None, a new figure is created.
            title (Optional[str]): Title of the plot.
            base (int): The base of the qudit space. Qubit is 2, qudit is 3, etc.
        """

        from matplotlib import pyplot as plt
        from matplotlib.colors import hsv_to_rgb
        import matplotlib.patches as mpatches

        nx = m.shape[0]
        ny = m.shape[1]
        qubit_number = int(np.round(np.log(nx) / np.log(base ** 2)))
        # qubit_number = int(np.log2(nx) / 2)

        # labels_single = ['I', 'Z', 'Y', 'X']
        # labels_single = self.basis_name

        # Reverse order
        # new_shape = [base ** 2] * qubit_number * 2
        # m = m.reshape(new_shape)
        #
        # transpose_order = [x for x in range(len(new_shape))]
        #
        # input_order = transpose_order[: int(len(new_shape) / 2)][::-1]
        # output_order = transpose_order[int(len(new_shape) / 2):][::-1]
        #
        # m = m.transpose(input_order + output_order)
        #
        # m = m.reshape([nx, ny])

        pauli_labels = self.basis_name

        do_adjust = False

        if ax is None:
            do_adjust = True
            fig, ax = plt.subplots()

        ax.axis("scaled")
        ax.patch.set_facecolor("white")

        mflat = m.flatten()
        ones = np.ones(mflat.shape[0])
        hues = np.arctan2(-mflat.imag, -mflat.real) / (2.0 * np.pi) + 0.5
        colors = hsv_to_rgb(np.dstack((hues, ones, ones)))

        sizes = np.abs(mflat) ** 0.5

        i = 0
        for ix in range(nx):
            for iy in range(ny):
                ax.add_artist(
                    mpatches.RegularPolygon(
                        xy=(ix, iy),
                        numVertices=4,
                        orientation=np.pi / 4,
                        color=colors[0, i],
                        radius=2.0 ** -0.5 * sizes[i],
                        linestyle="None",
                        linewidth=0.0,
                    )
                )  # 0.5*sizes[i]
                i += 1

        ax.xaxis.set_ticks(np.arange(nx))
        ax.xaxis.set_ticklabels(pauli_labels)
        ax.yaxis.set_ticks(np.arange(ny))
        ax.yaxis.set_ticklabels(pauli_labels)
        # ax.xaxis.set_tick_params(labelright='on')
        # ax.yaxis.set_tick_params(labeltop='on')

        ax.axis([-0.5, -0.5 + nx, -0.5 + nx, -0.5])

        legendhandles = []

        leghues = np.linspace(0.0, 1.0, 17)
        ones = np.ones(17)
        legendcolors = hsv_to_rgb(np.dstack((leghues, ones, ones)))
        for i in range(17):
            legendhandles += [
                mpatches.Patch(color=legendcolors[0, i]),
            ]

        for iy in range(ny + 1):
            ax.axhline(y=iy - 0.5, ls=":", color="k")
        for ix in range(nx + 1):
            ax.axvline(x=ix - 0.5, ls=":", color="k")

        if title is not None:
            ax.set_title(title)
        if do_adjust:
            # used to  be ax.legend
            fig.subplots_adjust(
                left=0.05, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1, right=0.85
            )

            fig.legend(
                legendhandles,
                [
                    "+ 1",
                    "",
                    "",
                    "",
                    "+ i",
                    "",
                    "",
                    "",
                    "- 1",
                    "",
                    "",
                    "",
                    "- i",
                    "",
                    "",
                    "",
                    "+1",
                ],
                loc="right",
                bbox_to_anchor=(1.0, 0.5),
            )

    plt.show()


class GateSet:
    def __init__(
            self,
            gate_names: List[str],
            gate_ideal_matrices: np.ndarray,
            basis=None
    ):
        """
        Initializes a set of quantum gates with their names and ideal unitary matrices.

        Args:
            gate_names (List[str]): List of names assigned to the gates.
            gate_ideal_matrices (np.ndarray): A 3D array where each 2D slice represents the ideal
                                              unitary matrix of a gate.
            basis (HilbertBasis, optional): The basis used for gate representation.
                                            Defaults to HilbertBasis with a dimension equal to that of gate matrices.

        Raises:
            AssertionError: If the number of gates does not match the third dimension of gate_ideal_matrices
                            or if the gate matrices are not square.
        """
        # Ensure the gate matrix is square and matches the number of gate names
        assert gate_ideal_matrices.shape[0] == gate_ideal_matrices.shape[
            1], f"Gate matrices must be square, got shape {gate_ideal_matrices.shape}."
        assert len(gate_names) == gate_ideal_matrices.shape[-1], "Number of gate names must match number of gates."

        # Store the dimension from the matrix size
        self.dimension = gate_ideal_matrices.shape[0]

        # If no basis is provided, create a default HilbertBasis with the appropriate dimension
        if basis is None:
            basis = HilbertBasis(dimension=self.dimension)

        self._basis = basis
        self._gate_names = gate_names
        self._name_to_index = {name: i for i, name in enumerate(gate_names)}
        self._gate_ideal_matrices = gate_ideal_matrices

        # Compute the Pauli Transfer Matrices (PTMs) for the ideal matrices
        self._ptms_ideal = np.dstack(
            [self._basis.unitary_to_ptm(self._gate_ideal_matrices[:, :, i])
             for i in range(self._gate_ideal_matrices.shape[-1])]
        )

        # Initialize the PTM estimates to the ideal values
        self._ptm_estimate = self._ptms_ideal.copy()

    def update_estimation(self, estimation: np.ndarray):
        """
        Updates the estimated PTMs for the gates.

        Args:
            estimation (np.ndarray): A 3D array containing the updated PTMs for the gates.
        """
        self._ptm_estimate = estimation

    def get_basis(self):
        """
        Returns the basis used in the gate set.

        Returns:
            HilbertBasis: The basis object used.
        """
        return self._basis

    @property
    def available_gates(self) -> List[str]:
        """
        Returns a list of the available gate names in the gate set.

        Returns:
            List[str]: The names of the gates.
        """
        return self._gate_names

    def get_ptm(self, name: str) -> np.ndarray:
        """
        Retrieves the PTM for a specified gate.

        Args:
            name (str): The name of the gate for which to get the PTM.

        Returns:
            np.ndarray: The PTM of the gate.
        """
        return self._ptm_estimate[:, :, self._name_to_index[name]]

    def get_ptm_ideal(self, name: str) -> np.ndarray:
        """
        Retrieves the ideal PTM for a specified gate.

        Args:
            name (str): The name of the gate for which to retrieve the ideal PTM.

        Returns:
            np.ndarray: The ideal PTM of the gate.
        """
        if isinstance(name, tuple):
            matrices = np.eye(self.dimension ** 2)
            for n in name:
                matrices = self.get_ptm_ideal(n) @ matrices
            return matrices

        return self._ptms_ideal[:, :, self._name_to_index[name]]

    def _cj_from_ptm(self, ptm: np.ndarray) -> np.ndarray:
        """
        Converts a Pauli Transfer Matrix (PTM) to a Choi-Jamiolkowski (CJ) representation.

        Args:
            ptm (np.ndarray): The PTM to convert.

        Returns:
            np.ndarray: The CJ representation of the PTM.
        """
        basis_matrices = self.get_basis().get_basis_matrices()
        dim = self.dimension ** 2

        return cj_from_ptm(ptm, basis_matrices, dim)

    def get_cj_from_ptm(self, name: str) -> np.ndarray:
        """
        Computes the CJ representation of a gate's PTM.

        Args:
            name (str): The name of the gate.

        Returns:
            np.ndarray: The CJ representation of the gate's PTM.
        """
        ptm = self.get_ptm(name)
        return self._cj_from_ptm(ptm)

    def get_cj_from_ptm_ideal(self, name: str) -> np.ndarray:
        """
        Computes the CJ representation of a gate's ideal PTM.

        Args:
            name (str): The name of the gate.

        Returns:
            np.ndarray: The CJ representation of the gate's ideal PTM.
        """
        ptm = self.get_ptm_ideal(name)
        return self._cj_from_ptm(ptm)

    def get_all_ptm_estimate(self) -> np.ndarray:
        """
        Returns all current estimates of PTMs for the gates.

        Returns:
            np.ndarray: A 3D array of all PTM estimates.
        """
        return self._ptm_estimate

    def get_all_ptm_ideal(self) -> np.ndarray:
        """
        Returns all ideal PTMs for the gates.

        Returns:
            np.ndarray: A 3D array of all ideal PTMs.
        """
        return self._ptms_ideal

    def get_all_unitary_ideal(self) -> np.ndarray:
        """
        Returns all ideal unitary matrices for the gates.

        Returns:
            np.ndarray: A 3D array of all ideal unitary matrices.
        """
        return self._gate_ideal_matrices

    def get_unitary_ideal(self, name: Union[str, Tuple[str]]) -> np.ndarray:
        """
        Retrieves the ideal unitary matrix for a specified gate.

        Args:
            name (str): The name of the gate.

        Returns:
            np.ndarray: The ideal unitary matrix of the gate.
        """

        if isinstance(name, tuple):
            matrices = np.eye(self.dimension)
            for n in name:
                matrices = self.get_unitary_ideal(n) @ matrices
            return matrices

        return self._gate_ideal_matrices[:, :, self._name_to_index[name]]

    def plot_gate_ptm(self, name: str, ideal: bool = False, title: str = None):
        """
        Plots the Pauli Transfer Matrix (PTM) of a gate.

        Args:
            name (str): The name of the gate to plot.
            ideal (bool): If True, plots the ideal PTM; otherwise, plots the estimated PTM.
            title (str, optional): The title of the plot. Defaults to a constructed title based on the gate name and matrix type.
        """
        ptm = self.get_ptm_ideal(name) if ideal else self.get_ptm(name)
        if title is None:
            title = f"PTM {'ideal' if ideal else 'estimate'} for {name}"
        self._basis.plot_process_matrix(ptm, title=title)

    @property
    def number_of_gates(self) -> int:
        """
        Returns the number of gates in the gate set.

        Returns:
            int: The number of gates.
        """
        return self._ptms_ideal.shape[-1]

    def evaluate_average_fidelity(self, name: str) -> float:
        ptm_estimated = self.get_ptm(name)
        ptm_ideal = self.get_ptm_ideal(name)
        basis_matrices = self.get_basis().get_basis_matrices()
        dim = self.dimension ** 2
        return evaluate_average_fidelity_ptm(ptm_estimated=ptm_estimated, ptm_ideal=ptm_ideal,
                                             basis_matrices=basis_matrices, dim=dim)

    def dump_configuration(self) -> dict:
        """
        Dumps the configuration of the gate set to a dictionary.

        Returns:
            dict: A dictionary containing the configuration of the gate set.
        """
        return {
            "gate_names": self._gate_names,
            "gate_ideal_matrices": self._gate_ideal_matrices,
            "basis": self._basis.dump_configuration(),
        }

    @staticmethod
    def load_configuration(configuration: dict):
        """
        Loads a gate set configuration from a dictionary.

        Args:
            configuration (dict): A dictionary containing the configuration of the gate set.

        Returns:
            GateSet: An instance of GateSet configured according to the provided dictionary.
        """
        basis_configure = configuration["basis"]
        basis = HilbertBasis(**basis_configure)
        configuration["basis"] = basis
        gateset = GateSet(**configuration)
        return gateset


def evaluate_fidelity_ptm_with_unitary(ptm_estimated: np.ndarray, u_ideal: np.ndarray, basis: HilbertBasis) -> float:
    """
    Evaluates the fidelity between the estimated PTM and an ideal unitary matrix.

    Args:
        ptm_estimated (np.ndarray): The estimated PTM.
        u_ideal (np.ndarray): The ideal unitary matrix.

    Returns:
        float: The fidelity value, real part only.
    """
    ptm_ideal = basis.unitary_to_ptm(u_ideal)
    return evaluate_average_fidelity_ptm(ptm_estimated, ptm_ideal, basis_matrices=basis.get_basis_matrices(),
                                         dim=ptm_estimated.shape[0])
