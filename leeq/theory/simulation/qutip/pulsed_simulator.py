import itertools

import numpy as np
import qutip
from qutip import basis, destroy, qeye, tensor, mesolve


class QutipPulsedSimulator(object):
    """
    A simulator for pulsed quantum systems using qutip.
    """

    def __init__(self, level_truncate=4):
        """
        Initialize the QutipPulsedSimulator class.

        Parameters:
            level_truncate (int): The number of self._level_truncate to truncate to.
        """

        self._level_truncate = level_truncate
        self._qubits = {}
        self._H = []
        self._H0 = []
        self._connectivity = []
        self._lindblad_T1s = []
        self._lindblad_T2s = []
        self._pulse_sequence = {}
        self._t_list = []

        self._ops = {
            "a": destroy(self._level_truncate),
            "n": destroy(self._level_truncate).dag() * destroy(self._level_truncate),
            "I": qeye(self._level_truncate),
            "g": basis(self._level_truncate, 0),  # ground state
            "e": basis(self._level_truncate, 1),  # excited state
            "X": basis(self._level_truncate, 0) * basis(self._level_truncate, 1).dag()
            + basis(self._level_truncate, 1) * \
            basis(self._level_truncate, 0).dag(),
            "Y": -1.0j
            * (
                basis(self._level_truncate, 0) * \
                basis(self._level_truncate, 1).dag()
                - basis(self._level_truncate, 1) * \
                basis(self._level_truncate, 0).dag()
            ),
            "Z": -(
                basis(self._level_truncate, 1) * \
                basis(self._level_truncate, 1).dag()
                - basis(self._level_truncate, 0) * \
                basis(self._level_truncate, 0).dag()
            ),
        }
        self._name_to_id = {}

        self._total_time = 0
        self._time_resolution = 0
        self._time_steps = 0

    def add_qubit(
            self,
            name: str,
            frequency: float,
            anharmonicity: float,
            t1: float,
            t2: float):
        """
        Add a qubit to the simulator.

        Parameters:
            name (str): The name of the qubit.
            frequency (float): The frequency of the qubit.
            anharmonicity (float): The anharmonicity of the qubit.
            t1 (float): The T1 time of the qubit.
            t2 (float): The T2 time of the qubit.
        """

        if name in self._qubits:
            raise RuntimeError("Qubit already exists.")

        self._qubits[name] = {
            "f": frequency,
            "alpha": anharmonicity,
            "t1": t1,
            "t2": t2,
        }

    def add_connectivity(
        self, q1_name: str, q2_name: str, coupling: str, strength: float
    ):
        """
        Add a connectivity between two qubits.

        Parameters:
            q1_name (str): The name of the first qubit.
            q2_name (str): The name of the second qubit.
            coupling (str): The type of coupling. Can be 'ZZ', 'XX', 'YY', 'XY', 'XZ', 'YZ'.
            strength (float): The strength of the coupling.
        """

        self._connectivity.append(
            {"q1": q1_name, "q2": q2_name, "coupling": coupling, "J": strength}
        )

    def build_system(self):
        """
        Build the quantum system master equations.
        """
        self._name_to_id = {
            name: i for i, name in enumerate(
                self._qubits.keys())}
        self._build_initial_state()
        self._build_hamiltonian()
        self._build_lindblad_operators()

    def _build_hamiltonian(self):
        """
        Build the Hamiltonian of the system.
        """

        self._H0 = []

        for name, q in self._qubits.items():
            H_1q = [self._ops["I"]] * len(self._qubits)
            if self._level_truncate == 2:
                H_1q[self._name_to_id[name]] = q["f"] * self._ops["n"]
            else:
                H_1q[self._name_to_id[name]] += (
                    q["f"] + q["alpha"] / 2 * (self._ops["n"] - 1)
                ) * self._ops["n"]
            self._H0 += [2 * np.pi * tensor(H_1q)]

        coupling_operators = {
            "Z": self._ops["n"],
            "X": self._ops["a"] + self._ops["a"].dag(),
        }

        for c in self._connectivity:
            H_2q = [self._ops["I"]] * len(self._qubits)

            q_id_1 = self._name_to_id[c["q1"]]
            q_id_2 = self._name_to_id[c["q2"]]

            pauli_1 = c["coupling"][0]
            pauli_2 = c["coupling"][1]

            H_2q[q_id_1] = coupling_operators[pauli_1]
            H_2q[q_id_2] = coupling_operators[pauli_2]

            self._H0 += [2 * np.pi * c["J"] * tensor(H_2q)]

    def _build_lindblad_operators(self):
        """
        Build the Lindblad operators of the system.
        """

        # T1
        self._lindblad_T1s = 0
        for k, q in self._qubits.items():
            t1_terms = [self._ops["I"]] * len(self._qubits)
            t1_terms[self._name_to_id[k]] = self._ops["a"]
            self._lindblad_T1s += np.sqrt(1 / q["t1"]) * tensor(t1_terms)

        # T2
        self._lindblad_T2s = 0
        for k, q in self._qubits.items():
            t2_terms = [self._ops["I"]] * len(self._qubits)
            t2_terms[self._name_to_id[k]] = self._ops["n"]
            t_phi = 1.0 / (1.0 / q["t2"] - 1.0 / (q["t1"] * 2))
            self._lindblad_T2s += np.sqrt(1 / t_phi) * tensor(t2_terms)

    def setup_clock(self, total_time: float, time_resolution: float):
        """
        Setup the clock of the simulator.

        Parameters:
            total_time (float): The total time of the simulation.
            time_resolution (float): The time resolution of the simulation.
        """

        self._total_time = total_time
        self._time_resolution = time_resolution
        self._time_steps = int(total_time / time_resolution + 0.5)

    def reset(self):
        """
        Reset the simulator.
        """

        self._H = self._H0[:]
        self._pulse_sequence = {}
        self._t_list = []
        self._total_time = 0
        self._time_resolution = 0
        self._time_steps = 0

    def _build_initial_state(self):
        """
        Build the initial state of the system.
        """

        initial_state = [self._ops["g"]] * len(self._qubits)
        initial_state = tensor(initial_state)

        self._rho0 = initial_state * initial_state.dag()

    def set_drive_buffer(self, qubit_name: str, pulse: np.ndarray):
        """
        Add a pulse to the simulator.

        Parameters:
            qubit_name (str): The name of the qubit to add the pulse to.
            pulse (np.ndarray): The pulse to add.
        """

        qubit_id = self._name_to_id[qubit_name]

        Hd = [self._ops["I"]] * len(self._qubits)
        Hd[qubit_id] = self._ops["a"] + self._ops["a"].dag()
        self._H += [[tensor(Hd), pulse]]

        self._pulse_sequence[qubit_name] = pulse

    def set_measurement_time(self, measurement_time: list[float]):
        """
        Set the measurement time point.

        Parameters:
            measurement_time (list[float]): The time point to measure.
        """
        # self._t_list = measurement_time
        self._t_list = np.linspace(0, self._total_time, self._time_steps)

    def run(self, no_noise=False):
        """
        Run the simulation.

        Parameters:
            no_noise (bool): Whether to run the simulation without noise.
        """

        # Find all permutations of Zs for the measurement
        Z_ops = [
            tensor(list(p))
            for p in itertools.product(
                [self._ops["I"], self._ops["Z"]], repeat=len(self._qubits)
            )
        ]
        self._t_list = np.linspace(0, self._total_time, self._time_steps)

        result = mesolve(
            H=self._H,
            rho0=self._rho0,
            tlist=self._t_list,
            c_ops=[] if no_noise else [self._lindblad_T1s, self._lindblad_T2s],
            e_ops=Z_ops,
            options=qutip.Options(gui=False),
        )

        return result.expect
