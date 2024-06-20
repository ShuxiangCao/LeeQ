from typing import List

import numpy as np
from labchronicle import register_browser_function

import leeq.theory.utils
from leeq import Experiment, Sweeper, basic_run
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial, LogicalPrimitiveBlockParallel, \
    LogicalPrimitiveBlockSweep
from leeq.utils import setup_logging
from leeq.utils.compatibility import *

__all__ = ['PyGSTiExperiment', 'PyGSTiRBExperiment']

logger = setup_logging(__name__)

from typing import List, Any, Tuple, Callable


class PyGSTiExperiment(Experiment):
    """
    A class for handling pyGSTi-based quantum experiments, focusing on experiment setup,
    data collection, and processing. Inherits from a generic Experiment class.
    """

    def _pygsti_design_to_lpbs(self, exp_design: "pygsti design",
                               duts: List['TransmonElements'],
                               mprim_indexes: List[int],
                               gate_name_to_lpb_func: Callable) -> Tuple[List[Any], List[Any]]:
        """
        Converts a pyGSTi design to a list of logical primitive blocks (LPBs).

        Args:
        exp_design: The experiment design from pyGSTi.
        duts: Devices under test, typically quantum elements like transmons.
        mprim_indexes: Indices of measurement primitives in the devices.
        gate_name_to_lpb_func: A function that maps gate names to LPBs.

        Returns:
        A tuple containing the list of LPB sequences and a list of measurement primitives.
        """
        list_of_experiments = exp_design.all_circuits_needing_data
        self.list_of_experiments = list_of_experiments

        mprims = [duts[i].get_measurement_prim_intlist(mprim_indexes[i]) for i in range(len(duts))]
        lpb_sequences = []

        for circuit in list_of_experiments:
            depth = circuit.num_layers
            lpb = []

            layer = circuit.to_label()
            layer_lpb = []
            for gate in layer.components:
                gate_name = gate.name
                if gate_name == 'COMPOUND':
                    continue

                qubit_index = gate.qubits
                if qubit_index is None:
                    qubit_index = 0
                layer_lpb.append(gate_name_to_lpb_func(gate_name, duts, qubit_index))

            if len(layer_lpb) == 0:
                lpb.append(duts[0].get_default_c1()['I'])
            elif len(layer_lpb) == 1:
                lpb.append(layer_lpb[0])
            else:
                lpb.append(LogicalPrimitiveBlockSerial(layer_lpb))

            if len(lpb) == 0:
                lpb = [duts[0].get_default_c1()['I']]

            lpb = LogicalPrimitiveBlockSerial(lpb) if len(lpb) > 1 else lpb[0]
            lpb = lpb + LogicalPrimitiveBlockParallel(mprims)
            lpb_sequences.append(lpb)

        return lpb_sequences, mprims

    def _construct_lpbs(self):
        """
        Placeholder for LPB construction logic.
        """
        pass

    def _process_collected_data(self):
        """
        Processes collected data using pyGSTi and updates the internal state.
        """
        import pygsti

        self.ds = pygsti.data.DataSet(outcome_labels=self.outcome_labels)

        for i, circuit in enumerate(self.list_of_experiments):
            self.ds.add_count_dict(circuit,
                                   {label: self.sample_counts[j, i] for j, label in enumerate(self.outcome_labels)})

        self.protocol_data = pygsti.protocols.ProtocolData(edesign=self.exp_design, dataset=self.ds)

    def _build_outcome_labels(self, duts_count: int):
        """
        Builds a list of outcome labels based on the number of devices.

        Args:
        duts_count: The number of devices under test.
        """
        self.outcome_labels = ['0', '1']

        for i in range(duts_count - 1):
            self.outcome_labels = [x + '0' for x in self.outcome_labels] + [x + '1' for x in self.outcome_labels]

    def run(self, design, duts: List['TransmonElements'], mprim_indexes: List[int]):
        """
        Main method to run the experiment with the given design and devices.

        Args:
        design: The experimental design to use.
        duts: List of devices under test.
        mprim_indexes: List of measurement primitive indexes.
        """
        from pygsti.protocols import CircuitListsDesign

        if not isinstance(design, CircuitListsDesign):
            raise ValueError("Design must be a pyGSTi Design object")

        if mprim_indexes is None:
            mprim_indexes = 0

        if isinstance(mprim_indexes, int):
            mprim_indexes = [mprim_indexes] * len(duts)

        self.duts = duts
        self.exp_design = design
        self._construct_lpbs()
        self._build_outcome_labels(len(duts))
        lpbs, mprims = self._pygsti_design_to_lpbs(exp_design=design,
                                                   duts=duts,
                                                   mprim_indexes=mprim_indexes,
                                                   gate_name_to_lpb_func=self.get_gate_lpb)

        sweep_lpb = LogicalPrimitiveBlockSweep(lpbs)
        swp = Sweeper.from_sweep_lpb(sweep_lpb)

        final_lpb = sweep_lpb

        basic_run(final_lpb, swp, "<zs>")

        self.result = np.dstack([mprim.result() for mprim in mprims]).transpose([2, 0, 1])
        self.sample_counts = leeq.theory.utils.to_dense_sample_counts(self.result)
        self._process_collected_data()

    @register_browser_function()
    def process_data_and_visualize(self):
        """
        Placeholder for data processing and visualization logic.
        """
        pass


class PyGSTiRBExperiment(PyGSTiExperiment):
    """
    A subclass to handle Randomized Benchmarking (RB) experiments using pyGSTi.
    """

    def process_rb_data(self):
        """
        Processes RB data and returns the results.

        Returns:
        RB results object.
        """
        import pygsti
        protocol = pygsti.protocols.RB()
        self.rb_results = protocol.run(self.protocol_data)
        return self.rb_results

    def visualize_rb_results(self):
        """
        Visualizes the RB results using pyGSTi's built-in visualization tools.
        """
        import pygsti
        ws = pygsti.report.Workspace()
        plot = ws.RandomizedBenchmarkingPlot(self.rb_results)

        for fig in plot.figs:
            fig.plotlyfig.show()
