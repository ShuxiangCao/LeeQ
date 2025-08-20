from leeq.chronicle import log_and_record

from .base import GeneralisedTomographyBase, GeneralisedSingleDutStateTomography, GeneralisedStateTomography, \
    GeneralisedProcessTomography

from leeq.theory.tomography import MultiQuditModel
from leeq.utils.compatibility import prims


class QuditTomographyBase(GeneralisedTomographyBase):

    def initialize_gate_lpbs(self, duts):
        _gate_lpbs_func = {}

        _gate_lpbs_func['I'] = lambda dut: dut.get_gate('I')
        _gate_lpbs_func['Xp_01'] = lambda dut: dut.get_gate('Xp', transition_name='f01')
        _gate_lpbs_func['Yp_01'] = lambda dut: dut.get_gate('Yp', transition_name='f01')
        _gate_lpbs_func['Xp_12'] = lambda dut: dut.get_gate('Xp', transition_name='f12')
        _gate_lpbs_func['Yp_12'] = lambda dut: dut.get_gate('Yp', transition_name='f12')
        _gate_lpbs_func['Xp_23'] = lambda dut: dut.get_gate('Xp', transition_name='f23')
        _gate_lpbs_func['Yp_23'] = lambda dut: dut.get_gate('Yp', transition_name='f23')

        gate_names = list(_gate_lpbs_func.keys())
        full_gate_names = gate_names

        number_of_qubits = len(duts)
        for i in range(number_of_qubits - 1):
            new_gate_names = []
            for gate_name in gate_names:
                for full_gate_name in full_gate_names:
                    new_gate_names.append(full_gate_name + ':' + gate_name)
            full_gate_names = new_gate_names

        def _get_multi_qubit_lpb_from_name(name):
            gate_names = name.split(':')
            lpb = prims.ParallelLPB(
                [_gate_lpbs_func[gate_name](duts[i]) for i, gate_name in enumerate(gate_names)])
            return lpb

        self._gate_lpbs = dict(zip(full_gate_names, [_get_multi_qubit_lpb_from_name(name) for name in full_gate_names]))


class MultiQuditsStateTomography(GeneralisedStateTomography, QuditTomographyBase):
    @log_and_record
    def run(self, duts, mprim_index=2, initial_lpb=None, extra_measurement_duts=None):
        model = MultiQuditModel(len(duts))
        super().run(duts, model, mprim_index, initial_lpb, extra_measurement_duts, base=4)


class MultiQuditsProcessTomography(GeneralisedProcessTomography, QuditTomographyBase):
    @log_and_record
    def run(self, duts, lpb, mprim_index=2, initial_lpb=None, extra_measurement_duts=None):
        model = MultiQuditModel(len(duts))
        super().run(duts=duts, model=model, lpb=lpb, mprim_index=mprim_index,
                    extra_measurement_duts=extra_measurement_duts, base=4)
