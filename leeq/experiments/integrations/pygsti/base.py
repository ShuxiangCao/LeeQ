from leeq import Experiment
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial, LogicalPrimitiveBlockParallel
from leeq.utils import setup_logging

logger = setup_logging(__name__)


def gate_name_to_lpb_single_qubit(label, dut):
    gate_lpbs = []

    if isinstance(label, str):
        c1 = dut.get_default_c1()
        gate_at_ith_qubit = label[0].upper()
        if gate_at_ith_qubit == 'I':
            return c1['I']
        else:
            return c1[gate_at_ith_qubit + 'p']
    else:
        for i in range(len(label)):
            c1 = duts[i].get_default_c1()
            gate_at_ith_qubit = label[i].upper()
            if gate_at_ith_qubit == 'I':
                continue
            gate_lpbs.append(c1[gate_at_ith_qubit + 'p'])

    if len(gate_lpbs) == 1:
        return gate_lpbs[0]

    return prims.ParallelLPB(gate_lpbs)


def pygsti_design_to_lpbs(exp_design, duts, mprim_indexes, gate_name_to_lpb_func):
    list_of_experiments = exp_design.all_circuits_needing_data

    lpb_sequences = []

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
            lpb.append(LogicalPrimitiveBlockParallel(layer_lpb))

        if len(lpb) == 0:
            lpb = [duts[0].get_default_c1()['I']]

        lpb = LogicalPrimitiveBlockSerial(lpb) if len(lpb) > 1 else lpb[0]
        lpb = lpb + LogicalPrimitiveBlockParallel(mprims)
        lpb_sequences.append(lpb)

    return lpb_sequences, mprims


class PyGSTiExperiment(Experiment):

    def _construct_lpbs(self, ):
        pass

    def run(self, design, duts, mprim_indexes):
        from pygsti.protocols import CircuitListsDesign

        if not isinstance(design, CircuitListsDesign):
            msg = "design must be a pyGSTi Design object"
            raise ValueError(msg)

        self.duts = duts
        self._construct_lpbs()
        lpbs, mprims = pygsti_design_to_lpbs(exp_design=design,
                                             duts=duts,
                                             mprim_indexes=mprim_indexes,
                                             gate_name_to_lpb_func=self.get_gate_lpb
                                             )


class PyGSTiExperimentQubitGates(PyGSTiExperiment):

    def _construct_lpbs(self):
        self._lpbs = []
        duts = self.duts
        for dut in duts:
            c1 = dut.get_c1('f01')
            self._lpbs.append({
                'Gxpi2': c1['Xp'],
                'Gxmpi2': c1['Xm'],
                'Gypi2': c1['Yp'],
                'Gympi2': c1['Ym']
            })

    def get_gate_lpb(self, gate_name, duts, qubit_index):

        label = gate_name.__str__()

        gate_lpbs = []

        if isinstance(label, str):
            return self._lpbs[0][label]
        else:
            for i in range(len(label)):
                lpb_map = self._lpbs[i]
                gate_lpbs.append(lpb_map[label])

        if len(gate_lpbs) == 1:
            return gate_lpbs[0]

        return LogicalPrimitiveBlockParallel(gate_lpbs)
