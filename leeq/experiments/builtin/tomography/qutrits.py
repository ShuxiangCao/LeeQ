from labchronicle import log_and_record

from leeq.theory.tomography import MultiQutritModel
from leeq.utils.compatibility import prims
from .base import GeneralisedTomographyBase, GeneralisedSingleDutStateTomography, GeneralisedStateTomography, \
    GeneralisedProcessTomography


class QutritTomographyBase(GeneralisedTomographyBase):

    def initialize_gate_lpbs(self, duts):
        _gate_lpbs_func = {}

        _gate_lpbs_func['I'] = lambda dut: dut.get_gate('I')
        _gate_lpbs_func['Xp_01'] = lambda dut: dut.get_gate('Xp', transition_name='f01')
        _gate_lpbs_func['Yp_01'] = lambda dut: dut.get_gate('Yp', transition_name='f01')
        _gate_lpbs_func['Xp_12'] = lambda dut: dut.get_gate('Xp', transition_name='f12')
        _gate_lpbs_func['Yp_12'] = lambda dut: dut.get_gate('Yp', transition_name='f12')
        _gate_lpbs_func['H'] = lambda dut: dut.get_gate('qutrit_hadamard')
        _gate_lpbs_func['Z1'] = lambda dut: dut.get_gate('qutrit_clifford_green', angle=(1, 0))
        _gate_lpbs_func['Z2'] = lambda dut: dut.get_gate('qutrit_clifford_green', angle=(0, 1))

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


class MultiQubitsStateTomography(GeneralisedStateTomography, QutritTomographyBase):
    @log_and_record
    def run(self, duts, mprim_index=1, initial_lpb=None, extra_measurement_duts=None):
        model = MultiQutritModel(len(duts))
        super().run(duts, model, mprim_index, initial_lpb, extra_measurement_duts)


class MultiQubitsProcessTomography(GeneralisedProcessTomography, QutritTomographyBase):
    @log_and_record
    def run(self, duts, lpb, mprim_index=1, initial_lpb=None, extra_measurement_duts=None):
        model = MultiQutritModel(len(duts))
        super().run(duts=duts, model=model, lpb=lpb, mprim_index=mprim_index,
                    extra_measurement_duts=extra_measurement_duts)
