import numpy as np
from labchronicle import register_browser_function, log_and_record
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockParallel
from .base import PyGSTiRBExperiment

from typing import Optional, List


class PyGSTiSingleQubitRB(PyGSTiRBExperiment):
    """
    A class that configures and runs single qubit randomized benchmarking experiments using pyGSTi.
    """

    @log_and_record
    def run(self, dut, depths: Optional[List[int]] = None, k: int = 10, randomizeout: bool = True,
            citerations: int = 20, mprim_indexes=0):
        """
        Executes the randomized benchmarking experiment.

        Parameters:
        - dut: The device under test.
        - depths (Optional[List[int]]): List of depths at which to perform the benchmarking.
                                        If None, a default set of depths is used.
        - k (int): Number of random Clifford sequences to generate per depth.
        - randomizeout (bool): Whether to randomize output errors in the experiment.
        - citerations (int): Number of compiler iterations for the experiment setup.
        - mprim_indexes (int): The index of the measurement primitive.

        Imports pyGSTi related modules and sets up the processor and experiment design.
        """
        import pygsti
        from pygsti.processors import QubitProcessorSpec as QPS
        from pygsti.processors import CliffordCompilationRules as CCR

        if depths is None:
            # depths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664,
            #           1792]
            depths = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        qubit_labels = ['Q0']
        gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2']
        pspec = QPS(num_qubits=1, gate_names=gate_names, availability={}, qubit_labels=qubit_labels)

        compilations = {
            'absolute': CCR.create_standard(pspec, 'absolute', ('paulis', '1Qcliffords'), verbosity=0),
            'paulieq': CCR.create_standard(pspec, 'paulieq', ('1Qcliffords', 'allcnots'), verbosity=0)
        }

        design = pygsti.protocols.CliffordRBDesign(pspec, compilations, depths, k, qubit_labels=qubit_labels,
                                                   randomizeout=randomizeout, citerations=citerations)

        super().run(duts=[dut], design=design, mprim_indexes=mprim_indexes)

    def _construct_lpbs(self):
        """
        Constructs the local Pauli blocks (LPBs) for the gates from device under test (DUT) specifications.
        """
        self._lpbs = []
        for dut in self.duts:
            c1 = dut.get_c1('f01')
            self._lpbs.append({
                'Gxpi2': c1['Xp'],
                'Gxmpi2': c1['Xm'],
                'Gypi2': c1['Yp'],
                'Gympi2': c1['Ym']
            })

    def get_gate_lpb(self, gate_name: str, duts, qubit_index: int):
        """
        Retrieves the local Pauli block for a specified gate.

        Parameters:
        - gate_name (str): The name of the gate.
        - duts: The devices under test.
        - qubit_index (int): Index of the qubit for which to get the LPB.

        Returns:
        - The local Pauli block corresponding to the gate and qubit.
        """
        gate_lpbs = [self._lpbs[0][gate_name]]
        if len(gate_lpbs) == 1:
            return gate_lpbs[0]
        return LogicalPrimitiveBlockParallel(gate_lpbs)

    @register_browser_function()
    def process_data_and_visualize(self):
        """
        Processes the collected randomized benchmarking data and visualizes the results.
        """
        self.process_rb_data()
        self.visualize_rb_results()
