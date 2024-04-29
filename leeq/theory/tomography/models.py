from scipy.linalg import expm

from .process_tomography import StandardProcessTomography
from .state_tomography import StandardStateTomography
from .utils import *

__ALL__ = ['StandardTomographyModels', 'SingleQubitModel']


def get_unitary_parametrized_by_angle(m: np.ndarray, angle: float) -> np.ndarray:
    """Generate a unitary matrix parameterized by an angle using the given matrix.

    Args:
        m (np.ndarray): The base matrix used for generating the unitary (e.g., Pauli matrices).
        angle (float): The angle to use for parameterization, typically a multiple of pi.

    Returns:
        np.ndarray: The resulting unitary matrix.
    """
    return np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * m


class StandardTomographyModels:
    """
    A class to model standard tomography models using a specified gate set and operation sequences.

    Attributes:
        _gate_set: The set of quantum gates used in the model.
        _basis: The computational basis associated with the gate set.
        _measurement_operations_sequence: A sequence of measurement operations.
        _max_germ_length: Maximum length for germ sequences.
        _germs: A list of germ sequences used in the model.
        _measurement_operators_implementable: Implementable measurement operators (default is the identity).
        _preparation_operations_sequence: A sequence of preparation operations.
        _initial_state: The initial state density matrix (default is the ground state).
    """

    def __init__(self, gate_set,
                 measurement_operations_sequence: list, preparation_operations_sequence: list,
                 measurement_operators_implementable: np.ndarray = None,
                 initial_state: np.ndarray = None) -> None:
        self._gate_set = gate_set
        self._basis = gate_set.get_basis()  # Fetch basis from gate set
        self._measurement_operations_sequence = measurement_operations_sequence

        if measurement_operators_implementable is None:
            # Default to identity operator if none provided
            measurement_operators_implementable = np.zeros(
                [self._basis.dimension, self._basis.dimension, self._basis.dimension])
            for i in range(self._basis.dimension):
                measurement_operators_implementable[i, i, i] = 1

        self._measurement_operators_implementable = measurement_operators_implementable
        self._preparation_operations_sequence = preparation_operations_sequence

        if initial_state is None:
            # Default to ground state if no initial state is provided
            initial_state = np.zeros([self._basis.dimension, self._basis.dimension])
            initial_state[0, 0] = 1

        self._initial_state = initial_state

    def get_basis(self) -> object:
        """Returns the computational basis associated with the gate set."""
        return self._basis

    def get_gate_set(self) -> object:
        """Returns the gate set used in the model."""
        return self._gate_set

    def construct_state_tomography(self) -> 'StandardStateTomography':
        """Constructs a state tomography configuration based on the current model setup."""
        return StandardStateTomography(gate_set=self._gate_set,
                                       measurement_operations=self._measurement_operations_sequence,
                                       measurement_operators=self._measurement_operators_implementable)

    def construct_process_tomography(self) -> 'StandardProcessTomography':
        """Constructs a process tomography configuration based on the current model setup."""
        return StandardProcessTomography(gate_set=self._gate_set,
                                         measurement_operations=self._measurement_operations_sequence,
                                         measurement_operators=self._measurement_operators_implementable,
                                         preparation_operations=self._preparation_operations_sequence,
                                         initial_state_density_matrix=self._initial_state)

    def plot_process_matrix(self, m: np.ndarray, title: str = None, ax=None, base=2):
        """Plots the process matrix with an optional title and on a specific axes."""
        return self._basis.plot_process_matrix(m=m, title=title, ax=ax, base=base)

    def plot_density_matrix(self, m: np.ndarray, title: str = None, base=2):
        """Plots the density matrix with an optional title."""
        return self._basis.plot_density_matrix(m=m, title=title, base=base)

    @property
    def dimension(self) -> int:
        """Returns the dimension of the computational basis."""
        return self.get_basis().dimension


class SingleQubitModel(StandardTomographyModels):
    def __init__(self) -> None:
        """Initialize a model for single-qubit tomography with standard bases and gates.

        This class defines the identity and Pauli matrices as basis operators for single qubit space,
        and constructs gates which are parameterized unitaries using these bases. The model includes
        definitions for measurement and preparation sequences based on these gates.
        """
        # Identity matrix
        I = np.eye(2)

        # Pauli X matrix
        X = np.asarray([
            [0, 1],
            [1, 0]
        ])

        # Pauli Y matrix with complex numbers
        Y = np.asarray([
            [0, -1.j],
            [1.j, 0]
        ])

        # Pauli Z matrix
        Z = np.asarray([
            [1, 0],
            [0, -1]
        ])

        # Stack the basis matrices along the third dimension
        basis_operators = np.dstack([I, X, Y, Z])
        basis_name = ['I', 'X', 'Y', 'Z']

        # Initialize Hilbert basis for the single qubit
        basis = HilbertBasis(dimension=2, basis_name=basis_name, basis_matrices=basis_operators)

        # Define gates with parameterized unitaries (e.g., rotation by pi/2 around X and Y)
        gates = np.dstack([
            np.eye(2),
            get_unitary_parametrized_by_angle(m=X, angle=np.pi / 2),
            get_unitary_parametrized_by_angle(m=Y, angle=np.pi / 2),
            get_unitary_parametrized_by_angle(m=X, angle=np.pi),
        ])

        gate_names = ['I', 'Xp', 'Yp', 'X']

        # Define the gate set using these unitaries and the basis
        gate_set = GateSet(gate_names=gate_names, gate_ideal_matrices=gates, basis=basis)

        # Define standard sequences for measurement and preparation
        measurement_sequence = [('I',), ('Xp',), ('Yp',)]
        preparation_sequence = [('I',), ('Xp',), ('Yp',), ('X',)]

        # Initialize the superclass with these defined components
        super(SingleQubitModel, self).__init__(gate_set=gate_set,
                                               measurement_operations_sequence=measurement_sequence,
                                               preparation_operations_sequence=preparation_sequence)


class MultiQubitModel(StandardTomographyModels):
    def __init__(self, number_of_qubit) -> None:
        """Initialize a model for multi-qubit tomography with standard bases and gates.

        This class defines the identity and Pauli matrices as basis operators for single qubit space,
        and constructs gates which are parameterized unitaries using these bases. The model includes
        definitions for measurement and preparation sequences based on these gates.
        """
        # Identity matrix
        I = np.eye(2)

        # Pauli X matrix
        X = np.asarray([
            [0, 1],
            [1, 0]
        ])

        # Pauli Y matrix with complex numbers
        Y = np.asarray([
            [0, -1.j],
            [1.j, 0]
        ])

        # Pauli Z matrix
        Z = np.asarray([
            [1, 0],
            [0, -1]
        ])

        # Stack the basis matrices along the third dimension
        basis_operators = [I, X, Y, Z]
        basis_name = ['I', 'X', 'Y', 'Z']

        # Generate tensor products of the gates for multi-qubit system
        full_basis_operators = basis_operators
        full_basis_name = basis_name
        for i in range(number_of_qubit - 1):
            kron_basis_operators = []
            kron_basis_name = []
            for name, basis_operator in zip(full_basis_name, full_basis_operators):
                for n, o in zip(basis_name, basis_operators):
                    kron_basis_operators.append(np.kron(basis_operator, o))
                    kron_basis_name.append(name + n)
            full_basis_operators = kron_basis_operators
            full_basis_name = kron_basis_name

        basis_name = full_basis_name
        basis_operators = np.dstack(full_basis_operators)

        # Initialize Hilbert basis for the single qubit
        basis = HilbertBasis(dimension=2 ** number_of_qubit, basis_name=basis_name, basis_matrices=basis_operators)

        # Define gates with parameterized unitaries (e.g., rotation by pi/2 around X and Y)
        gates = [
            np.eye(2),
            get_unitary_parametrized_by_angle(m=X, angle=np.pi / 2),
            get_unitary_parametrized_by_angle(m=Y, angle=np.pi / 2),
            get_unitary_parametrized_by_angle(m=X, angle=np.pi),
        ]

        gate_names = ['I', 'Xp', 'Yp', 'X']

        # Generate tensor products of the gates for multi-qubit system
        full_gates = gates
        full_gates_name = gate_names

        for i in range(number_of_qubit - 1):
            kron_gates = []
            kron_gates_name = []
            for name, gate in zip(full_gates_name, full_gates):
                for n, o in zip(gate_names, gates):
                    kron_gates.append(np.kron(gate, o))
                    kron_gates_name.append(name + ':' + n)
            full_gates = kron_gates
            full_gates_name = kron_gates_name

        gates = np.dstack(full_gates)
        gate_names = full_gates_name

        # Define the gate set using these unitaries and the basis
        gate_set = GateSet(gate_names=gate_names, gate_ideal_matrices=gates, basis=basis)

        # Define standard sequences for measurement and preparation
        measurement_sequence = [('I',), ('Xp',), ('Yp',)]
        preparation_sequence = [('I',), ('Xp',), ('Yp',), ('X',)]

        measurement_sequence_full = measurement_sequence
        preparation_sequence_full = preparation_sequence
        # Find the sequence for multi-qubit system
        for i in range(number_of_qubit - 1):
            measurement_sequence_full = [(x[0] + ':' + y[0],) for x in measurement_sequence_full for y in
                                         measurement_sequence]
            preparation_sequence_full = [(x[0] + ':' + y[0],) for x in preparation_sequence_full for y in
                                         preparation_sequence]

        # Initialize the superclass with these defined components
        super(MultiQubitModel, self).__init__(gate_set=gate_set,
                                              measurement_operations_sequence=measurement_sequence_full,
                                              preparation_operations_sequence=preparation_sequence_full)
