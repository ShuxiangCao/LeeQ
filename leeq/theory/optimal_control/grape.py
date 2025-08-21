from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, basis, destroy, qzero
from qutip.control import grape_unitary_adaptive, plot_grape_control_fields


@dataclass
class HamiltonianParams:
    num_levels_in_transmon = 3
    omega_01 = 5.0 * 2 * np.pi  # Qubit transition frequency (GHz)
    anharmonicity = -0.33 * 2 * np.pi  # Anharmonicity (GHz)


def make_Hamiltonian(params: HamiltonianParams) -> Qobj:
    N = params.num_levels_in_transmon
    ket_2 = basis(N, 2)
    proj_2 = ket_2 * ket_2.dag()
    H0 = params.anharmonicity / (np.pi * 2) * proj_2
    return H0


@dataclass
class GrapeParams:
    T = 30  # Total time for evolution (ns)
    n_ts = 60  # Number of time steps
    num_iterations = 500
    amplitude = 0.015
    eps = 0.05 * np.sqrt(np.pi * amplitude)  # Scaled gradient step size for GRAPE


def run_grape(hamiltonian_params: HamiltonianParams, grape_params: GrapeParams, U_target: Qobj,
              initial_guess: np.ndarray) -> tuple:  # Returns (result, overlap, times)
    H0 = make_Hamiltonian(hamiltonian_params)
    N = hamiltonian_params.num_levels_in_transmon
    times = np.linspace(0, grape_params.T, grape_params.n_ts)  # Time array
    # Control Hamiltonians
    a = destroy(N)
    H_x = a + a.dag()
    H_y = 1j * (a - a.dag())

    # Initial and target unitary operations

    # Initial control fields (random initial guess)
    # Transpose to shape (n_ts, num_controls)
    # GRAPE optimization parameters
    result = grape_unitary_adaptive(
        U=U_target,  # Target unitary
        H0=H0,  # Drift Hamiltonian
        H_ops=[H_x, H_y],  # Control Hamiltonians
        R=grape_params.num_iterations,
        times=times,  # Time grid
        u_start=initial_guess,  # Initial control fields
        interp_kind="linear",  # Linear interpolation of control fields
        eps=grape_params.eps,  # Scaled gradient step size
        use_interp=False,  # No interpolation (use provided times directly)
        # alpha=1e-3,  # No custom cost function parameter
        # beta=1e-3,  # No custom cost function parameter
        phase_sensitive=False,  # Not phase-sensitive
    )
    return result, np.abs(overlap(U_target, result.U_f)), times


def overlap(A, B):
    return (A.dag() * B).tr() / A.shape[0]


def get_single_qubit_pulse_grape(qubit_frequency,
                                 anharmonicity,
                                 width,
                                 sampling_rate,
                                 initial_guess=None,
                                 max_amplitude=0.015,
                                 num_levels_in_transmon=3,
                                 num_iterations=100):

    params = HamiltonianParams()
    params.num_levels_in_transmon = num_levels_in_transmon
    params.omega_01 = qubit_frequency / 1e3 * 2 * np.pi,  # Qubit transition frequency (GHz)
    params.anharmonicity = anharmonicity / 1e3 * 2 * np.pi  # Anharmonicity (GHz)

    ham_params = params
    amplitude = max_amplitude / 2
    grape_params = GrapeParams()
    grape_params.T = width * 1e3  # Total time for evolution (ns)
    grape_params.n_ts = int(width * sampling_rate)  # Number of time steps
    grape_params.num_iterations = num_iterations
    grape_params.amplitude = max_amplitude / 2
    grape_params.eps = 0.05 * np.sqrt(np.pi * amplitude)  # Scaled gradient step size for GRAPE

    N = ham_params.num_levels_in_transmon

    id = qzero(N)
    for i in range(2, N):
        id = id + basis(N, i) * basis(N, i).dag()

    U_target = (basis(N, 0) * basis(N, 1).dag() + basis(N, 1) * basis(N, 0).dag()) + id

    if initial_guess is None:
        control_fields_x = np.ones(grape_params.n_ts) * grape_params.amplitude
        control_fields_y = np.zeros(grape_params.n_ts)

        initial_guess = np.array([control_fields_x, control_fields_y])

    result, fidelity, times = run_grape(hamiltonian_params=ham_params, grape_params=grape_params,
                                        U_target=U_target, initial_guess=initial_guess)

    assert fidelity > 0.999, f'fidelity {fidelity} is too low'

    return result


def example():
    result = get_single_qubit_pulse_grape(
        qubit_frequency=5000,
        anharmonicity=-200,
        width=0.05,
        sampling_rate=2e3,
        initial_guess=None,
        max_amplitude=0.015,
    )

    times = np.arange(len(result.u))

    plot_grape_control_fields(times, result.u, ['x', 'y'])
    # Print results
    plt.grid()
    plt.show()


def example_bak():
    ham_params = HamiltonianParams()
    grape_params = GrapeParams()

    N = ham_params.num_levels_in_transmon

    id = qzero(N)
    for i in range(2, N):
        id = id + basis(N, i) * basis(N, i).dag()
    # higher_levels.[:2,:2] = qeye(N-2)

    U_target = (basis(N, 0) * basis(N, 1).dag() + basis(N, 1) * basis(N, 0).dag()) + id

    control_fields_x = np.ones(grape_params.n_ts) * grape_params.amplitude
    control_fields_y = np.zeros(grape_params.n_ts)
    initial_guess = np.array([control_fields_x, control_fields_y])

    result, fidelity, times = run_grape(HamiltonianParams(), GrapeParams(), U_target, initial_guess)

    # Plot optimized control fields
    plot_grape_control_fields(times, result.u, ['x', 'y'])
    # Print results
    plt.grid()
    plt.show()


if __name__ == '__main__':
    example()
