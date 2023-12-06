import copy
from typing import List, Union

import numpy as np
from matplotlib import pyplot as plt

from labchronicle import log_and_record, register_browser_function
from leeq import Experiment, Sweeper, basic_run
from leeq.theory.cliffords import get_clifford_from_id
from leeq.utils.compatibility import prims
import scipy.optimize as so


class RandomizedBenchmarkingTwoLevelSubspaceMultilevelSystem(Experiment):
    """
    Class for running a randomized benchmarking experiment on a multilevel system.
    """

    @log_and_record
    def run(self,
            dut_list: List,
            collection_name: str = 'f01',
            seq_length: Union[int, np.ndarray] = 256,
            kinds: int = 20,
            cliff_set: str = 'XY',
            pi_half_only: bool = False,
            mprim_index: int = 0,
            seed: int = 42) -> None:
        """
        Executes a run with a sequence of operations on a list of devices under test (DUTs).

        Args:
            dut_list: A list of device under test (DUT) instances.
            collection_name: Name of the collection to use, with default 'f01'.
            seq_length: Either an integer specifying the max length of the sequence, or an array of sequence lengths.
            kinds: The number of kinds/types of sequences to generate.
            cliff_set: The set of Clifford gates to use, either 'XY' or 'VZX'.
            pi_half_only: If True, only pi/2 rotations are used in gate generation.
            mprim_index: Index to select the measurement primitive.
            seed: Seed for random number generation to ensure reproducibility.

        Returns:
            None: This method modifies the instance's state but does not return any value.
        """
        # Validate input arguments
        if not isinstance(seq_length, (int, np.ndarray)):
            raise TypeError("seq_length must be an integer or a numpy array")
        if not isinstance(collection_name, str):
            raise TypeError("collection_name must be a string")
        if not cliff_set in {'XY', 'VZX'}:
            raise ValueError("cliff_set must be either 'XY' or 'VZX'")
        if not collection_name in {'f01', 'f12', 'f23', 'f02', 'f13'}:
            raise ValueError("collection_name must be one of the specified collections")

        if cliff_set == 'VZX':
            raise NotImplementedError("Clifford set 'VZX' not yet implemented.")

        # Convert seq_length to a numpy array if it's an integer
        if isinstance(seq_length, int):
            seq_length = np.arange(0, seq_length + 1, 16)

        self.seq_length = seq_length

        # Seed the random number generator
        np.random.seed(seed)

        # Determine flip-up and flip-down local Pauli blocks (LPB) based on collection_name
        flip_up_lpb, flip_down_lpb = None, None
        if collection_name[1] == '0':
            flip_lpb = prims.ParallelLPB([dut.get_c1('f01')['I'] for dut in dut_list])
            flip_up_lpb = flip_down_lpb = flip_lpb
        elif collection_name[1] == '1':
            flip_lpb = prims.ParallelLPB([dut.get_c1('f01')['X'] for dut in dut_list])
            flip_up_lpb = flip_down_lpb = flip_lpb
        elif collection_name[1] == '2':
            flip_up_lpb = prims.ParallelLPB([dut.get_c1('f01')['X'] + dut.get_c1('f12')['X'] for dut in dut_list])
            flip_down_lpb = prims.ParallelLPB([dut.get_c1('f12')['X'] + dut.get_c1('f01')['X'] for dut in dut_list])

        self.cliff_set = cliff_set

        # Retrieve the C1 gates for each DUT
        c1s = [dut.get_c1(collection_name) for dut in dut_list]

        # Define a function to get a Clifford gate based on its ID
        def get_clifford(c1, clifford_id):
            return prims.SerialLPB([c1[x] for x in get_clifford_from_id(clifford_id)])

        # Generate the gate sequences
        self.gates = [
            [(get_clifford(c1s[i], j) if cliff_set == 'XY' else c1s[i].get_VZ_clifford(j, pi_half_only=pi_half_only))
             for j in range(24)]
            for i in range(len(dut_list))
        ]

        # Create LPBs for each sequence
        lpbs = []
        for seq in seq_length:
            lpb_list_same_length = []
            for k in range(kinds):
                lpb_for_different_duts_list = []
                for j in range(len(dut_list)):
                    _, lpb = self.generate_sequence(length=seq, dut_id=j)
                    lpb_for_different_duts_list.append(lpb)
                lpb_list_same_length.append(prims.ParallelLPB(lpb_for_different_duts_list))
            lpbs += lpb_list_same_length

        # Create a sweep LPB
        sweep_lpb = prims.SweepLPB(lpbs)
        swp = Sweeper.from_sweep_lpb(sweep_lpb)

        # Get measurement primitives
        m_prims = [dut.get_measurement_prim_intlist(mprim_index) for dut in dut_list]
        final_lpb = flip_up_lpb + sweep_lpb + flip_down_lpb + prims.ParallelLPB(m_prims)

        # Execute the basic run
        basic_run(final_lpb, swp, '<zs>')

        # Store the results
        self.results = np.asarray([np.squeeze(x.result()).transpose() for x in m_prims])

    def generate_sequence(self, length: int, dut_id: int):
        """
        Generate a random Clifford sequence of the specified length.

        Args:
            length: The length of the sequence.
            dut_id: The index of the DUT to use.

        Returns:
            A tuple containing the sequence and the corresponding LPB.
        """
        clifford_indices = np.random.randint(24, size=length)

        from leeq.theory.cliffords import append_inverse_C1

        clifford_indices_identity = append_inverse_C1(clifford_indices, self.cliff_set)

        gate_list = self.gates[dut_id]

        lpb = prims.SeriesLPB([gate_list[i] for i in clifford_indices_identity])

        return clifford_indices_identity, lpb

    def to_dense_probabilities(self, data_measured):  # [qubit index,sample index,anything else]
        """
        Convert the data measured to dense probabilities.

        Parameters:
            data_measured: Data retrieved from FPGA. The shape is [qubit index,sample index,anything else...].
        Returns:
            A numpy array of dense probabilities.
        """

        base = np.max(data_measured) + 1
        self.base = base

        data_measured = copy.copy(data_measured)

        qubit_count = data_measured.shape[0]
        aggregated_data = data_measured[0]

        for i in range(1, qubit_count):
            aggregated_data += base ** i * data_measured[i]

        bins = np.asarray([np.sum((aggregated_data == i).astype(int), axis=0) for i in range(base ** qubit_count)])

        return bins / data_measured.shape[1]

    def _analyze_decay_single_qubit(self, i):
        """
        Analyze the decay of the randomized benchmarking experiment.

        This method fits the data to an exponential decay function and saves
        the decay rate and the covariance matrix to instance attributes.

        Parameters:
            i (int): The index of the qubit to analyze.

        Returns:
            rb_mean (np.ndarray): List of mean values of the probabilities for each sequence.
            rb_std (np.ndarray): List of standard deviation of the probabilities for each sequence.
            rb_popt (np.ndarray): List of optimal values for the exponential decay parameters.
            rb_pcov (np.ndarray): List of covariance matrices for the parameter estimates.
            rb_probs (np.ndarray): List of the converted probabilities distribution of each qubit.

        """

        assert i < self.results.shape[0], f"Unexpected qubit index {i}, maximum index {self.results.shape[0] - 1}"

        # Retrieving arguments and initializing variables
        args = self.retrieve_args(self.run)
        seq_length = self.seq_length
        kinds = args['kinds']

        # Initialize lists to store results
        rb_mean: List[float] = []
        rb_std: List[float] = []
        rb_popt: List[np.ndarray] = []
        rb_pcov: List[np.ndarray] = []

        # Define the exponential decay function
        decay_function = lambda x, a, p0, p1: a * np.exp(p0 * x) + p1
        bounds = [(-1, -1, -1), (1, 1, 1)]

        results_single_qubit = self.results[i, :, :].reshape([1] + list(self.results.shape)[1:])

        # Reshape the results and convert to dense probabilities
        reshaped_results = results_single_qubit.reshape(
            [results_single_qubit.shape[0], results_single_qubit.shape[1], len(seq_length), kinds]
        )
        probs = self.to_dense_probabilities(reshaped_results)

        # Process the probabilities and perform curve fitting
        for i in range(probs.shape[0]):
            # Calculate mean and standard deviation across the sequences
            rb_result = probs[i, :, :]
            rb_mean.append(rb_result.mean(axis=1))
            rb_std.append(rb_result.std(axis=1))

        for i, mean in enumerate(rb_mean):
            # Initial guess and data preparation
            if i == 0:
                initial_guess = [1, 0, 1]
                data = mean
            else:
                initial_guess = [1, 0, 0]
                data = 1 - mean

            # Curve fitting while ensuring std is not zero by adding a small value
            popt, pcov = so.curve_fit(
                decay_function,
                seq_length,
                data,
                bounds=bounds,
                sigma=rb_std[i] + 1e-10,
                p0=initial_guess
            )

            # Append the fitting results to the instance attributes
            rb_popt.append(popt)
            rb_pcov.append(pcov)

        return rb_mean, rb_std, rb_popt, rb_pcov, probs

    def analyze_decay(self) -> None:
        """
        Analyze the decay of the randomized benchmarking experiment.

        This method fits the data to an exponential decay function and saves
        the decay rate and the covariance matrix to instance attributes.

        Attributes updated:
            self.mean (List[float]): List of mean values of the probabilities for each sequence.
            self.std (List[float]): List of standard deviation of the probabilities for each sequence.
            self.popt (List[np.ndarray]): List of optimal values for the exponential decay parameters.
            self.pcov (List[np.ndarray]): List of covariance matrices for the parameter estimates.
            self.probs (List[np.ndarray]): List of the converted probabilities distribution of each qubit.
        """
        # Retrieving arguments and initializing variables
        args = self.retrieve_args(self.run)
        seq_length = self.seq_length
        kinds = args['kinds']

        # Initialize lists to store results
        self.mean: List[float] = []
        self.std: List[float] = []
        self.popt: List[np.ndarray] = []
        self.pcov: List[np.ndarray] = []
        self.probs: List[np.ndarray] = []

        for i in range(self.results.shape[0]):
            rb_mean, rb_std, rb_popt, rb_pcov, probs = self._analyze_decay_single_qubit(i)
            self.mean.append(rb_mean)
            self.std.append(rb_std)
            self.popt.append(rb_popt)
            self.pcov.append(rb_pcov)
            self.probs.append(probs)

    @register_browser_function(available_after=(run,))
    def plot(self):
        """
        Plot the results of the randomized benchmarking experiment.
        """
        self.analyze_decay()
        for i in range(self.results.shape[0]):
            self.plot_single_qubit_result(i)

    def plot_single_qubit_result(self, qubit_id, save_path=None):
        """
        Plot the results of the randomized benchmarking experiment.

        Args:
            qubit_id: the id of the qubit to plot.
            save_path: Path to save the plot to.
        """
        try:
            # self.analyze_decay()
            analyze_success = True
        except Exception as e:
            print(repr(e))
            analyze_success = False

        args = self.retrieve_args(self.run)
        x = self.seq_length
        kinds = args['kinds']
        xs = np.linspace(x[0], x[-1], 100)

        levels = len(self.popt[qubit_id])

        colors = ['#E31B23', '#005CAB', '#FFC325', '#56AB4B']

        fig = plt.figure()
        for i in range(levels):
            mean = self.mean[qubit_id]
            popt = self.popt[qubit_id]
            std = self.std[qubit_id]

            plt.errorbar(x, mean[i], std[i], fmt='o', markersize=4, capsize=4, color=colors[i],
                         label=rf'$| {i} \rangle$')
            for j in range(self.probs[qubit_id].shape[2]):
                plt.scatter(x, self.probs[qubit_id][i, :, j], marker='.', color='k', alpha=0.2)

            if analyze_success:

                data = popt[i][0] * np.exp(popt[i][1] * xs) + popt[i][2]

                if i > 0:
                    data = 1 - data

                plt.plot(xs, data, color=colors[i])

        d = 2  # for two level system

        plt.axhline(y=1 / d, color='k', linestyle='--')

        perr = np.sqrt(np.diag(self.pcov[qubit_id][0]))
        infidelity = (d - 1) / d * (1 - np.exp(self.popt[qubit_id][0][1]))
        error_bar = (d - 1) / d * (perr[1] / np.exp(self.popt[qubit_id][0][1]))

        plt.title(
            f'Qubit {qubit_id} {args["cliff_set"]} {args["collection_name"]} randomized benchmarking \n Fidelity:{1 - infidelity} $\pm$ {error_bar}')
        plt.xlabel('Sequence length')
        plt.ylabel(r'Population')
        plt.legend()

        perr = perr[1]  # np.abs(perr[1] * p)

        r = infidelity

        infidelity = r  # (d - 1) / d * (1 - p)
        error_bar = (d - 1) / d * perr

        N_g = 1.825
        infidelity_per_gate = 1 - (1 - infidelity) ** (1 / 1.825)
        infidelity_per_gate_std = (1 - infidelity_per_gate) * (1 / N_g) * error_bar / (1 - r)

        print(f"Qubit {qubit_id}: Infidelity per clifford:", infidelity)
        print(f"Qubit {qubit_id}: Error bar per clifford:", error_bar)
        print(f"Qubit {qubit_id}: Infidelity per physical gate", infidelity_per_gate)
        print(f"Qubit {qubit_id}: Error bar per physical gate", infidelity_per_gate_std)

        if save_path is not None:
            plt.savefig(save_path)

        return fig

#    def analyze_decay(self) -> None:
#        """
#        Analyze the decay of the randomized benchmarking experiment.
#
#        This method fits the data to an exponential decay function and saves
#        the decay rate and the covariance matrix to instance attributes.
#
#        Attributes updated:
#            self.mean (List[float]): List of mean values of the probabilities for each sequence.
#            self.std (List[float]): List of standard deviation of the probabilities for each sequence.
#            self.popt (List[np.ndarray]): List of optimal values for the exponential decay parameters.
#            self.pcov (List[np.ndarray]): List of covariance matrices for the parameter estimates.
#        """
#        # Retrieving arguments and initializing variables
#        args = self.retrieve_args(self.run)
#        seq_length = self.seq_length
#        kinds = args['kinds']
#
#        # Initialize lists to store results
#        self.mean: List[float] = []
#        self.std: List[float] = []
#        self.popt: List[np.ndarray] = []
#        self.pcov: List[np.ndarray] = []
#
#        # Define the exponential decay function
#        decay_function = lambda x, a, p0, p1: a * np.exp(p0 * x) + p1
#        bounds = [(-1, -1, -1), (1, 1, 1)]
#
#        # Reshape the results and convert to dense probabilities
#        reshaped_results = self.results.reshape(
#            [self.results.shape[0], self.results.shape[1], len(seq_length), kinds]
#        )
#        probs = self.to_dense_probabilities(reshaped_results)
#        self.probs = probs
#
#        # Process the probabilities and perform curve fitting
#        for i in range(probs.shape[0]):
#            # Calculate mean and standard deviation across the sequences
#            rb_result = probs[i, :, :]
#            self.mean.append(rb_result.mean(axis=1))
#            self.std.append(rb_result.std(axis=1))
#
#        for i, mean in enumerate(self.mean):
#            # Initial guess and data preparation
#            if i == 0:
#                initial_guess = [1, 0, 1]
#                data = mean
#            else:
#                initial_guess = [1, 0, 0]
#                data = 1 - mean
#
#            # Curve fitting while ensuring std is not zero by adding a small value
#            popt, pcov = so.curve_fit(
#                decay_function,
#                seq_length,
#                data,
#                bounds=bounds,
#                sigma=self.std[i] + 1e-10,
#                p0=initial_guess
#            )
#
#            # Append the fitting results to the instance attributes
#            self.popt.append(popt)
#            self.pcov.append(pcov)
#
#    @register_browser_function(available_after=(run,))
#    def plot(self, save_path=None):
#        """
#        Plot the results of the randomized benchmarking experiment.
#
#        Args:
#            save_path: Path to save the plot to.
#        """
#
#        try:
#            self.analyze_decay()
#            analyze_success = True
#        except Exception as e:
#            print(repr(e))
#            analyze_success = False
#
#        args = self.retrieve_args(self.run)
#        x = self.seq_length
#        kinds = args['kinds']
#        xs = np.linspace(x[0], x[-1], 100)
#
#        levels = len(self.popt)
#
#        colors = ['#E31B23', '#005CAB', '#FFC325', '#56AB4B']
#
#        fig = plt.figure()
#        for i in range(levels):
#
#            plt.errorbar(x, self.mean[i], self.std[i], fmt='o', markersize=4, capsize=4, color=colors[i],
#                         label=rf'$| {i} \rangle$')
#            for j in range(self.probs.shape[2]):
#                plt.scatter(x, self.probs[i, :, j], marker='.', color='k', alpha=0.2)
#
#            if analyze_success:
#
#                data = self.popt[i][0] * np.exp(self.popt[i][1] * xs) + self.popt[i][2]
#
#                if i > 0:
#                    data = 1 - data
#
#                plt.plot(xs, data, color=colors[i])
#
#        d = 2  # for two level system
#
#        plt.axhline(y=1 / d, color='k', linestyle='--')
#
#        perr = np.sqrt(np.diag(self.pcov[0]))
#        infidelity = (d - 1) / d * (1 - np.exp(self.popt[0][1]))
#        error_bar = (d - 1) / d * (perr[1] / np.exp(self.popt[0][1]))
#
#        plt.title(
#            f'{args["cliff_set"]} {args["collection_name"]} randomized benchmarking \n Fidelity:{1 - infidelity} $\pm$ {error_bar}')
#        plt.xlabel('Sequence length')
#        plt.ylabel(r'Population')
#        plt.legend()
#
#        perr = perr[1]  # np.abs(perr[1] * p)
#
#        r = infidelity
#
#        infidelity = r  # (d - 1) / d * (1 - p)
#        error_bar = (d - 1) / d * perr
#
#        N_g = 1.825
#        infidelity_per_gate = 1 - (1 - infidelity) ** (1 / 1.825)
#        infidelity_per_gate_std = (1 - infidelity_per_gate) * (1 / N_g) * error_bar / (1 - r)
#
#        print("Infidelity per clifford:", infidelity)
#        print("Error bar per clifford:", error_bar)
#        print('Infidelity per physical gate', infidelity_per_gate)
#        print('Error bar per physical gate', infidelity_per_gate_std)
#
#        if save_path is not None:
#            plt.savefig(save_path)
#
#        plt.show()
#
#        return fig
#
#    @register_browser_function(available_after=(run,))
#    def plot_leakage(self, save_path=None):
#        """
#        Plot the results of the randomized benchmarking experiment.
#
#        Args:
#            save_path: Path to save the plot to.
#        """
#        try:
#            self.analyze_decay()
#            analyze_success = True
#        except Exception as e:
#            print(repr(e))
#            analyze_success = False
#
#        args = self.retrieve_args(self.run)
#        x = self.seq_length
#        kinds = args['kinds']
#        xs = np.linspace(x[0], x[-1], 100)
#
#        levels = len(self.popt)
#
#        colors = ['#E31B23', '#005CAB', '#FFC325', '#56AB4B']
#
#        fig = plt.figure()
#
#        collection_name = args["collection_name"]
#        used_sublevel = [0, int(collection_name[2])]
#
#        d = 2  # for two level system
#
#        for i in range(levels):
#
#            if i in used_sublevel:
#                continue
#
#            alpha = self.popt[i][1]
#            B = 1 - self.popt[i][2]
#            lpg = (1 - np.exp(alpha)) * B
#            print(rf'Leakage per gate level {i} :{lpg}')
#
#            plt.errorbar(x, self.mean[i], self.std[i], fmt='o', markersize=4, capsize=4, color=colors[i],
#                         label=rf'$| {i} \rangle$')
#            for j in range(self.probs.shape[2]):
#                plt.scatter(x, self.probs[i, :, j], marker='.', color='k', alpha=0.2)
#
#            if analyze_success:
#
#                data = self.popt[i][0] * np.exp(self.popt[i][1] * xs) + self.popt[i][2]
#
#                if i > 0:
#                    data = 1 - data
#
#                plt.plot(xs, data, color=colors[i])
#
#            perr = np.sqrt(np.diag(self.pcov[0]))
#            infidelity = (d - 1) / d * (1 - np.exp(self.popt[0][1]))
#            error_bar = (d - 1) / d * (perr[1] / np.exp(self.popt[0][1]))
#
#        perr = np.sqrt(np.diag(self.pcov[0]))
#        infidelity = (d - 1) / d * (1 - np.exp(self.popt[0][1]))
#        error_bar = (d - 1) / d * (perr[1] / np.exp(self.popt[0][1]))
#
#        plt.title(
#            f'{args["cliff_set"]} {args["collection_name"]} leakage randomized benchmarking')
#        plt.xlabel('Sequence length')
#        plt.ylabel(r'Population')
#        plt.legend()
#
#        print("p:", self.popt[0][1])
#
#        if save_path is not None:
#            plt.savefig(save_path)
#
#        plt.show()
#
#        return fig
#
