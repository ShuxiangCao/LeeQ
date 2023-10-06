from typing import List, Dict, Tuple, Union

import numpy as np
import scipy


class VirtualTransmon(object):
    """
    A virtual transmon device, recording the parameters and implement simulations.
    """

    def __init__(self, name: str, qubit_frequency: float, anharmonicity: float, t1: float, t2: float,
                 readout_frequency: float, readout_linewith: float = 1, readout_dipsersive_shift: float = 1,
                 truncate_level=4, quiescent_state_distribution: Union[List[float], None] = None,
                 frequency_selectivity_window: float = 50):
        """
        Initialize the VirtualTransmon class.

        Parameters:
            name (str): The name of the transmon.
            qubit_frequency (float): The frequency of the qubit.
            anharmonicity (float): The anharmonicity of the qubit.
            t1 (float): The T1 time of the qubit. The higher levels are assumed to have the T1 divided by the level.
            t2 (float): The T2 time of the qubit, the higher levels are assumed to have the T2 divided by the level.
            readout_frequency (float): The frequency of the readout.
            readout_linewith (float): The linewidth of the readout.
            readout_dipsersive_shift (float): The dispersive shift of the readout. The higher levels are assumed to
                have the same dispersive shift.
            truncate_level (int): The number of levels of the quantum system to truncate to.
            quiescent_state_distribution (Union[List[float], None]): The distribution of the quiescent state. If None,
                the ground state perfectly prepared.
            frequency_selectivity_window (float): The frequency selectivity window of the transmon. If the the drive
                frequency is outside of the window (centered at the transition frequency), we simplify the simulation
                by assuming the drive does not affect the system. Otherwise an X/Y + Z rotation is applied.
        """

        self.name = name
        self.qubit_frequency = qubit_frequency
        self.anharmonicity = anharmonicity
        self.t1 = t1
        self.t2 = t2
        self.readout_frequency = readout_frequency
        self.readout_linewidth = readout_linewith
        self.readout_dipsersive_shift = readout_dipsersive_shift
        self.truncate_level = truncate_level
        self.quiescent_state_distribution = quiescent_state_distribution
        self.frequency_selectivity_window = frequency_selectivity_window
        self._density_matrix = None

        self._transition_ops = {}
        self._build_transition_operators()

        self._resonator_frequencies = None
        self._build_resonator_response()

        self.reset_state()

    def _build_resonator_response(self):
        """
        Find the resonator frequency when the transmon is at different state. Here we use the simplest approximation.
        Which consider the dispersive shift are the same.
        """
        self._resonator_frequencies = np.asarray(
            [self.readout_frequency - i * self.readout_dipsersive_shift for i in range(self.truncate_level)]
        )

    def get_resonator_response(self, f: float):
        """
        Get the resonator response at a given frequency.

        Parameters:
            f (float): The frequency to be evaluated.

        Returns:
            np.ndarray: The resonator response.
        """
        Q = self.readout_frequency / self.readout_linewidth
        s_11 = 1 - 2j * Q / (1 + 1.j * Q * (f - self._resonator_frequencies) / self._resonator_frequencies)

        return s_11

    def _build_transition_operators(self):
        """
        Build the transition Hamiltonian operators.
        The dictionary has the following format:
        {
            frequency: (number of photons involved in this transition, I drive (X) Hamiltonian, Q drive (Y)
                Hamiltonian, Z Hamiltonian)
        }
        """

        single_photon_transition_frequencies = [self.qubit_frequency + i * self.anharmonicity for i in
                                                range(self.truncate_level-1)
                                                ]

        level_energies = [0] + list(np.cumsum(single_photon_transition_frequencies))

        for i in range(self.truncate_level - 1):
            for j in range(i + 1, self.truncate_level):
                photon_number = j - i
                frequency = (level_energies[j] - level_energies[i]) / photon_number

                X_term = np.zeros((self.truncate_level, self.truncate_level), dtype=np.complex128)
                X_term[i, j] = 1
                X_term[j, i] = 1

                Y_term = np.zeros((self.truncate_level, self.truncate_level), dtype=np.complex128)
                Y_term[i, j] = -1.0j
                Y_term[j, i] = 1.0j

                X_term /= photon_number * np.sqrt(i + 1)
                Y_term /= photon_number * np.sqrt(i + 1)

                Z_term = np.zeros((self.truncate_level, self.truncate_level), dtype=np.complex128)
                Z_term[i, i] = 1

                self._transition_ops[frequency / photon_number] = (j - i, X_term, Y_term, Z_term)

    def apply_drive(self, frequency: float, pulse_shape: np.ndarray, sampling_rate: int):
        """
        Apply a drive to the transmon.

        Parameters:
            frequency (float): The frequency of the drive, in MHz unit.
            pulse_shape (np.ndarray): The pulse shape of the drive.
            sampling_rate (int): The sampling rate of the drive, in Msps unit.
        """

        transition_drives = [(transition_freq, transition_operators)
                             for transition_freq, transition_operators in self._transition_ops.items() if
                             np.abs(transition_freq - frequency) < self.frequency_selectivity_window / 2
                             ]

        if len(transition_drives) == 0:
            return

        if len(transition_drives) > 1:
            raise ValueError(
                f'Transitions {[x[0] for x in transition_drives]} are too close, lower the frequency'
                f' selectivity window, or the simulator does not suit your purpose.')

        transition_frequency, (photon_number, operator_x, operator_y, operator_z) = transition_drives[0]
        drive_frequency_difference = frequency - transition_frequency

        # Here for speed and simplicity we assume the Hamiltonian is almost
        # steady, therefore we simply take the average.
        drive_aggregate = np.average(pulse_shape)
        strength_x = np.real(drive_aggregate)
        strength_y = np.imag(drive_aggregate)

        hamiltonian = np.pi * 2 * (
                strength_x * operator_x + strength_y * operator_y +
                drive_frequency_difference * photon_number * operator_z)

        unitary = scipy.linalg.expm(-1.0j * hamiltonian * len(pulse_shape) / sampling_rate)

        self._density_matrix = unitary @ self._density_matrix @ unitary.conj().T

        # Apply T1 and T2 noise
        # TODO: apply the noise

    def apply_readout(self, return_type: str, sampling_number: int = 0, sampling_rate: int = None,
                      iq_noise_std: float = 0, trace_noise_std: float = 0, readout_frequency: float = None,
                      readout_width: float = None,
                      readout_shape: np.ndarray = None):

        """
        Apply the readout to the transmon.

        Parameters:
            return_type (str): The type of the return value. Can be 'population_distribution', 'IQ', 'traces',
            sampling_rate (int): The sampling rate of the readout, in Msps unit.
            sampling_number (int): The number of samples to be returned. Only works for 'IQ' and 'traces' return.
            iq_noise_std (float): The standard deviation of the IQ noise, only used for 'IQ'.
            trace_noise_std (float): The standard deviation of the trace increment noise, only used for 'traces'.
            readout_frequency (float): The frequency of the readout, in MHz unit. Only used for 'IQ' and 'traces'.
            readout_width (float): The width of the readout, in ns unit. Only used for 'IQ'.
            readout_shape (np.ndarray): The shape of the readout, in ns unit. Only used for 'traces'.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]: The
                return value.
        """

        assert return_type in ['population_distribution', 'IQ', 'IQ_average', 'traces'], \
            (f'Invalid return type "{return_type}",'
             f' acceptable types are "population_distribution", "IQ_average", "traces".')

        if return_type == 'population_distribution':
            population_distribution = np.diag(self._density_matrix).astype(np.float64)
            return population_distribution

        if return_type == 'IQ':
            return self._apply_readout_iq(readout_frequency=readout_frequency,
                                          sampling_number=sampling_number,
                                          iq_noise_std=iq_noise_std,
                                          readout_width=readout_width)

        if return_type == 'IQ_average':
            return self._apply_averaged_iq(readout_frequency=readout_frequency,
                                           iq_noise_std=iq_noise_std,
                                           readout_width=readout_width)

        if return_type == 'traces':
            # TODO: implement the trace
            raise NotImplementedError()

    def _apply_averaged_iq(self, readout_frequency: float, iq_noise_std: float,
                           readout_width=None):
        """
        Apply the readout to the transmon and return the averaged IQ data.

        Parameters:
            readout_frequency (float): The frequency of the readout.
            iq_noise_std (float): The standard deviation of the IQ noise.
            readout_width (float): The width of the readout.

        Returns:
            np.ndarray: The averaged IQ data in complex value.
        """

        population_distribution = np.diag(self._density_matrix).astype(np.float64)
        readout_response = self.get_resonator_response(readout_frequency)
        return np.mean(population_distribution * readout_response) * readout_width

    def _apply_readout_iq(self, readout_frequency: float, sampling_number: int, iq_noise_std: float,
                          readout_width=None):
        """
        Apply the readout to the transmon and return the IQ data.

        Parameters:
            sampling_number (int): The number of samples to be returned.
            iq_noise_std (float): The standard deviation of the IQ noise.

        Returns:
            np.ndarray: The IQ data in complex value.
        """
        population_distribution = np.diag(self._density_matrix).astype(np.float64)

        # Set the mean and standard deviation for each axis
        mu, sigma = 0, iq_noise_std  # mean and standard deviation

        # Generate independent Gaussian noise for each axis
        noise_x = np.random.normal(mu, sigma, sampling_number)
        noise_y = np.random.normal(mu, sigma, sampling_number)

        population_distribution = np.diag(self._density_matrix).astype(np.float64)
        readout_response = self.get_resonator_response(readout_frequency)

        sampled_state = np.random.choice(np.arange(self.truncate_level), size=sampling_number,
                                         p=population_distribution)
        single_shot_response = readout_response[sampled_state]

        noise = noise_x + 1j * noise_y

        single_shot_response += noise * np.std(readout_response) * 3

        return single_shot_response

    def reset_state(self):
        """
        Reset the state of the transmon to the ground state.

        Returns:
        """
        if self.quiescent_state_distribution is not None:
            self._density_matrix = np.diag(self.quiescent_state_distribution).astype(np.complex128)
        else:
            self._density_matrix = np.zeros((self.truncate_level, self.truncate_level), dtype=np.complex128)
            self._density_matrix[0, 0] = 1


class VirtualTransmonDevice(object):
    """
    A class that represents a device, recording the parameters.
    """

    def __init__(self, name: str, transmons: List[VirtualTransmon]):
        """
        Initialize the Device class.

        Parameters:
            name (str): The name of the device.
            transmons (List[VirtualTransmon]): The transmons in the device.
        """
        self._name = name
        self.transmons = transmons

    @property
    def name(self):
        """
        The name of the device.

        Returns:
            str: The name of the device.
        """
        return self._name

    def __repr__(self):
        return f'Device(name={self._name})'

    def __str__(self):
        return self.__repr__()


class TransmonRotatedFrameSimulator(object):
    """
    A simulator that simplify the dynamics and simulate behaviour in the rotated frame, as a fast simulator mainly for
    experiment code testing purpose. The simulator does not simulate multiple qubit interactions, however it does
    simulate the spectroscopy experiment and multilevel behaviour of a transmon, therefore pretty useful for testing
    the automated tune up code.

    Note that if you send two signal to a same transmon at the same time, the simulator will not simulate the correct
    behaviour, instead it treat that the pulse is applied one after another.

    The simulator simulates the behaviour of the quantum system and the measurement separately.

    For simulating the behaviour of the quantum system, we assume the following:
    1. We assume all the pulses can be converted into a single interaction terms, and therefore generate one
         unitary term.
    2. We appy the unitary to the state obtain the state.

    For simulating the measurement, we assume the following:
    1. We assume the measurement is a projective measurement.
    2. We return a point on the IQ plane, with noise, based on the state of the system.
    """

    def __init__(self, device: VirtualTransmonDevice):
        """
        Initialize the TransmonRotatedFrameSimulator class.

        Parameters:
            device (VirtualTransmonDevice): The device to be simulated.
        """
        self._device = device
