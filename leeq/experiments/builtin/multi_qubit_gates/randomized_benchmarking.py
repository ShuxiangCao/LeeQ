import multiprocessing
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties as unc
from joblib import Parallel, delayed
from k_agents.inspection.decorator import text_inspection, visual_inspection
from scipy import optimize as so
from tqdm.notebook import tqdm
from uncertainties.umath import exp as uexp

from leeq import Experiment, Sweeper, basic_run
from leeq.chronicle import log_and_record, register_browser_function
from leeq.core.primitives.logical_primitives import (
    LogicalPrimitiveBlock,
    LogicalPrimitiveBlockParallel,
    LogicalPrimitiveBlockSerial,
    LogicalPrimitiveBlockSweep
)
from leeq.theory.cliffords.two_qubit_cliffords import NC2, append_inverse_C2
from leeq.theory.utils import to_dense_probabilities


def _rb2q_find_sequence(seq_length, gateset, interleaved, cliff_set='XY'):
    interleaved_clifford_indexes = None
    sequences = []
    for j in seq_length:
        clifford_indexes = np.random.randint(gateset, size=j)
        if interleaved is not None:
            interleaved_clifford_indexes = np.zeros(dtype=clifford_indexes.dtype, shape=(2 * j,))
            interleaved_clifford_indexes[::2] = clifford_indexes
            interleaved_clifford_indexes[1::2] = interleaved
            clifford_indexes = interleaved_clifford_indexes

        clifford_indexes = append_inverse_C2(clifford_indexes, cliff_set=cliff_set)

        sequences.append(clifford_indexes)

    return sequences


class RandomizedBenchmarking2Qubits(Experiment):

    def build_sequence(self, indexes: List[int]) -> LogicalPrimitiveBlock:
        """Build a sequence from given indexes, considering whether to ignore identity."""
        return LogicalPrimitiveBlockSerial([self.c2.get_clifford(k, ignore_identity=False) for k in indexes])

    def find_a_sequences(self, seq):
        """Build sequences from clifford indexes, including interleaved sequences if specified."""
        return [self.build_sequence(clifford_indexes) for clifford_indexes in seq]

    @log_and_record
    def run(self, duts, c2, seq_length: int, kinds: int, interleaved: Optional[bool] = None):
        """Run the randomized benchmarking with parallel processing."""

        self.control_qubit = duts[0]
        self.target_qubit = duts[1]
        self.c2 = c2

        num_cores = multiprocessing.cpu_count()
        with Parallel(n_jobs=num_cores) as parallel:
            results = parallel(delayed(_rb2q_find_sequence)(seq_length, NC2, interleaved) for _ in
                               tqdm(range(kinds), desc="Generating Sequences"))
            sequences = parallel(
                delayed(self.find_a_sequences)(result) for result in tqdm(results, desc="Processing Sequences"))

        lpbs = [sequence for sublist in sequences for sequence in sublist]

        mprims = [dut.get_measurement_prim_intlist(0) for dut in duts]

        lpb = LogicalPrimitiveBlockSweep(lpbs)
        swp = Sweeper.from_sweep_lpb(lpb)

        basic_run(lpb + LogicalPrimitiveBlockParallel(mprims), swp, '<zs>')

        self.results = [mprim.result() for mprim in mprims]

    def analyze_result(self):
        kinds = self._get_run_args_dict()['kinds']
        seq_length = self._get_run_args_dict()['seq_length']
        duts = self._get_run_args_dict()['duts']

        data = np.squeeze(np.asarray(self.results)).transpose([0, 2, 1])
        data = to_dense_probabilities(data)
        data = data.reshape([2 ** len(duts), kinds, len(seq_length)])
        success_probability = data.mean(axis=1)[0, :]
        self.success_probability = success_probability

        # Define the exponential decay function
        def decay_function(x, a, p0, p1): return a * np.exp(p0 * x) + p1

        bounds = [(-1, -1, -1), (1, 1, 1)]
        initial_guess = [1, 0, 1]

        # Curve fitting while ensuring std is not zero by adding a small
        # value
        popt, pcov = so.curve_fit(
            decay_function,
            seq_length,
            success_probability,
            bounds=bounds,
            p0=initial_guess
        )
        self.popt = popt
        self.pcov = pcov

        u_params = unc.correlated_values(popt, pcov)
        from uncertainties.umath import exp as uexp

        d = 4  # for two level system

        self.perr = np.sqrt(np.diag(pcov))
        self.infidelity = (d - 1) / d * (1 - uexp(u_params[1]))
        self.error_bar = self.infidelity.s

    @register_browser_function()
    def plot1(self):
        dark_navy = '#000080'
        dark_purple = '#800080'

        self.analyze_result()

        args = self._get_run_args_dict()
        seq_length = args['seq_length']

        lseq = np.linspace(0, np.amax(seq_length) + 1, 1001)

        fig = plt.figure()

        plt.scatter(seq_length, self.success_probability, marker='o', label='RB', color=dark_navy)

        # fit_curve = self.popt[0] * np.exp(self.popt[1] ** lseq) + self.popt[2]
        fit_curve = self.popt[0] * np.exp(self.popt[1] * lseq) + self.popt[2]

        print(self.popt)
        print(lseq)
        print(fit_curve)
        plt.plot(lseq, fit_curve, color=dark_navy)

        plt.title(f'Randomized benchmarking 2Q \n F={1 - self.infidelity}')
        plt.xlabel(u"Number of of 2Q Cliffords")
        plt.ylabel(u"P(00)")
        plt.legend()

        return fig


class RandomizedBenchmarking2QubitsInterleavedComparison(Experiment):
    """
    A class to compare standard and interleaved randomized benchmarking for two qubits.

    Attributes:
        seq_length_std (List[int]): Sequence lengths for standard RB.
        seq_length_interleaved (List[int]): Adjusted sequence lengths for interleaved RB.
        fit_params (Dict[str, Any]): Fitting parameters and statistics for both RB methods.
        success_probability (Dict[str, List[float]]): Success probabilities for standard and interleaved RB.
        infidelity (float): Calculated relative infidelity between standard and interleaved RB.
    """

    _experiment_result_analysis_instructions = """
    This is the analysis of the randomized benchmarking experiment. The experiment is considered successful if no
    parameter needs to be updated based on the visual plot inspection, and the infidelity values are physical. Otherwise
    the experiment is failed.
    """

    @log_and_record
    def run(self, duts: List[Any], c2: Any, seq_length: List[int], kinds: int = 10, interleaved: int = 10299) -> None:
        """
        Executes the experiment comparing standard and interleaved randomized benchmarking for two qubit gates.

        Args:
            duts (List[Any]): Devices under test.
            c2 (Any): Control parameter for the qubits.
            seq_length (List[int]): List of sequence lengths for the RB.
            kinds (int): Parameter indicating different kinds of sequences.
            interleaved (int): Specifies the index of the gate interleaving. By default, it is set to 10299 for CZ gate.

        Example:
            >>> cz_index = 10299
            >>> c2 = prims.build_CZ_stark_from_parameters(control_q=dut1, target_q=dut2, trunc=1.0,**calibrated_params)
            >>> rb_interleaved = RandomizedBenchmarking2QubitsInterleavedComparison(duts=[dut1,dut2], kinds=10,
            >>>     seq_length=[0,2,4,6,8,10,12,16,20,24], c2=c2,interleaved=cz_index)
        """
        self.seq_length_std = seq_length
        standard_rb = RandomizedBenchmarking2Qubits(
            duts=duts,
            c2=c2,
            seq_length=seq_length,
            kinds=kinds,
            interleaved=None
        )

        # Compute interleaved sequence lengths
        self.seq_length_interleaved = [i // 2 + 1 if i % 2 else i // 2 for i in seq_length]

        interleaved_rb = RandomizedBenchmarking2Qubits(
            duts=duts,
            c2=c2,
            seq_length=self.seq_length_interleaved,
            kinds=kinds,
            interleaved=interleaved
        )

        # Analyze results from both standard and interleaved RB
        standard_rb.analyze_result()
        interleaved_rb.analyze_result()

        # Store fit parameters and related statistics
        self.fit_params = {
            'standard': {
                'popt': standard_rb.popt,
                'pcov': standard_rb.pcov,
                'perr': standard_rb.perr,
                'infidelity': standard_rb.infidelity,
                'error_bar': standard_rb.error_bar
            },
            'interleaved': {
                'popt': interleaved_rb.popt,
                'pcov': interleaved_rb.pcov,
                'perr': interleaved_rb.perr,
                'infidelity': interleaved_rb.infidelity,
                'error_bar': interleaved_rb.error_bar
            }
        }

        # Calculate success probabilities for both types
        self.success_probability = {
            'standard': standard_rb.success_probability,
            'interleaved': interleaved_rb.success_probability
        }

        # Compute the ratio of probabilities to find relative infidelity
        p_s = uexp(unc.ufloat(standard_rb.popt[1], standard_rb.perr[1]))
        p_i = uexp(unc.ufloat(interleaved_rb.popt[1], interleaved_rb.perr[1]))
        self.infidelity = (4 - 1) / 4 * (1 - (p_i / p_s))

    @text_inspection
    def fitting(self) -> Union[str, None]:
        return f"""
        The standard randomized benchmarking reports infidelity pgc to be {self.fit_params['standard']['infidelity']}.
        The interleaved randomized benchmarking reports infidelity pgc to be {self.fit_params['interleaved']['infidelity']}.
        The infidelity of the interleaved gate is {self.infidelity}.
        """

    @register_browser_function()
    @visual_inspection("""
    This is the analysis of the randomized benchmarking experiment. The experiment is considered successful two clear 
    exponential decays are observed. if the decay is too fast, the experiment is failed reduce the sequence length.
    If the decay is too slow, the experiment is failed and increase the sequence length. If the decay rate is proper,
    the experiment is successful.  
    """)
    def plot(self) -> None:
        """
        Generates a plot comparing the fitting curves and success probabilities for
        standard and interleaved randomized benchmarking.
        """

        fidelity = 1 - self.infidelity
        data = {
            'Infidelity': self.infidelity,
            'Fidelity': fidelity
        }
        df = pd.DataFrame(list(data.items()), columns=['Parameter', 'Value'])

        # Formatting the float values to three decimal places
        df['Value'] = df['Value'].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
        print(df)

        args = self._get_run_args_dict()
        seq_length = args['seq_length']

        lseq = np.linspace(0, np.amax(seq_length) + 1, 1001)

        fig = plt.figure()
        # Define the colors
        dark_navy = '#000080'
        dark_purple = '#800080'

        # Plot scatter data with the specified colors
        plt.scatter(self.seq_length_std, self.success_probability['standard'], marker='o', color=dark_navy,
                    label='Standard')
        plt.scatter(self.seq_length_interleaved, self.success_probability['interleaved'], marker='o', color=dark_purple,
                    label='Interleaved')

        fit_curve_standard = self.fit_params['standard']['popt'][0] * np.exp(
            self.fit_params['standard']['popt'][1]) ** lseq + self.fit_params['standard']['popt'][2]

        fit_curve_interleaved = self.fit_params['interleaved']['popt'][0] * np.exp(
            self.fit_params['interleaved']['popt'][1]) ** lseq + self.fit_params['interleaved']['popt'][2]

        # Plot the fit curves with the same colors as the scatter data
        plt.plot(lseq, fit_curve_standard, label='', color=dark_navy)
        plt.plot(lseq, fit_curve_interleaved, label='', color=dark_purple)

        # Add title and labels
        plt.title(f'Randomized benchmarking 2Q \n F={1 - self.infidelity}')
        plt.xlabel(u"Number of of 2Q Cliffords")
        plt.ylabel(u"P(00)")
        plt.legend()
        return fig
