import multiprocessing

import numpy as np
from scipy import optimize as so
from labchronicle import register_browser_function, log_and_record
from matplotlib import pyplot as plt
from typing import List, Optional
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import multiprocessing
import uncertainties as unc
from uncertainties.umath import exp as uexp

from leeq import Sweeper, basic_run, Experiment
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep, LogicalPrimitiveBlockParallel, \
    LogicalPrimitiveBlockSerial, LogicalPrimitiveBlock
from leeq.theory.cliffords.two_qubit_cliffords import append_inverse_C2, NC2, NC2_lim
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
        kinds = self.retrieve_args(self.run)['kinds']
        seq_length = self.retrieve_args(self.run)['seq_length']
        duts = self.retrieve_args(self.run)['duts']

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

        d = 4  # for two level system

        self.perr = np.sqrt(np.diag(pcov))
        self.infidelity = (d - 1) / d * (1 - np.exp(popt[1]))
        self.error_bar = (d - 1) / d * (self.perr[1] / np.exp(popt[1]))

    @register_browser_function()
    def plot(self):
        self.analyze_result()

        args = self.retrieve_args(self.run)
        seq_length = args['seq_length']

        lseq = np.linspace(0, np.amax(seq_length) + 1, 1001)
        colors = ['k', 'r', 'g', 'b', 'c', 'm', 'o', 'y']
        plt.scatter(seq_length, self.success_probability, marker='o', label='Experiment')
        fit_curve = self.popt[0] * np.exp(self.popt[1]) ** lseq + self.popt[2]
        plt.plot(lseq, fit_curve, label='Fitting', color='k')

        plt.title(f'Randomized benchmarking 2Q \n F={1 - self.infidelity}+-{self.error_bar}')
        plt.xlabel(u"Number of of 2Q Cliffords")
        plt.ylabel(u"P(00)")
        plt.legend()
        plt.show()


class RandomizedBenchmarking2QubitsInterleavedComparison(Experiment):

    @log_and_record
    def run(self, duts, c2, seq_length: int, kinds: int, interleaved: int):
        self.seq_length_std = seq_length
        standard_rb = RandomizedBenchmarking2Qubits(
            duts=duts,
            c2=c2,
            seq_length=seq_length,
            kinds=kinds,
            interleaved=None
        )

        self.seq_length_interleaved = [i // 2 + 1 if i % 2 else i // 2 for i in seq_length]

        interleaved_rb = RandomizedBenchmarking2Qubits(
            duts=duts,
            c2=c2,
            seq_length=[i // 2 + 1 if i % 2 else i // 2 for i in seq_length],
            kinds=kinds,
            interleaved=interleaved
        )

        standard_rb.analyze_result()
        interleaved_rb.analyze_result()

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

        self.success_probability = {
            'standard': standard_rb.success_probability,
            'interleaved': interleaved_rb.success_probability
        }

        p_s = uexp(unc.ufloat(standard_rb.popt[1], standard_rb.perr[1]))
        p_i = uexp(unc.ufloat(interleaved_rb.popt[1], interleaved_rb.perr[1]))

        self.infidelity = (4 - 1) / 4 * (1 - (p_i / p_s))

    @register_browser_function()
    def plot(self):
        args = self.retrieve_args(self.run)
        seq_length = args['seq_length']

        lseq = np.linspace(0, np.amax(seq_length) + 1, 1001)
        colors = ['k', 'r', 'g', 'b', 'c', 'm', 'o', 'y']

        plt.scatter(self.seq_length_std, self.success_probability['standard'], marker='o', label='Standard')
        plt.scatter(self.seq_length_interleaved, self.success_probability['interleaved'], marker='o',
                    label='Interleaved')

        fit_curve_standard = self.fit_params['standard']['popt'][0] * np.exp(
            self.fit_params['standard']['popt'][1]) ** lseq + self.fit_params['standard']['popt'][2]
        fit_curve_interleaved = self.fit_params['interleaved']['popt'][0] * np.exp(
            self.fit_params['interleaved']['popt'][1]) ** lseq + self.fit_params['interleaved']['popt'][2]

        plt.plot(lseq, fit_curve_standard, label='Fitting standard RB', color='k')
        plt.plot(lseq, fit_curve_interleaved, label='Fitting interleaved RB', color='r')

        plt.title(f'Randomized benchmarking 2Q \n F={1 - self.infidelity}')
        plt.xlabel(u"Number of of 2Q Cliffords")
        plt.ylabel(u"P(00)")
        plt.legend()
        plt.show()
