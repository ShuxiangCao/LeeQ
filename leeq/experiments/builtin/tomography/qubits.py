from labchronicle import log_and_record
from leeq.utils.compatibility import prims
from .base import GeneralisedTomographyBase, GeneralisedSingleDutStateTomography, GeneralisedStateTomography, \
    GeneralisedProcessTomography


class QubitTomographyBase(GeneralisedTomographyBase):
    def initialize_gate_lpbs(self, duts):

        gate_names = ['I', 'Xp', 'Yp', 'X']
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
                [duts[i].get_gate(gate_name, transition_name='f01') for i, gate_name in enumerate(gate_names)])
            return lpb

        self._gate_lpbs = dict(zip(full_gate_names, [_get_multi_qubit_lpb_from_name(name) for name in full_gate_names]))


class SingleQubitStateTomography(GeneralisedSingleDutStateTomography):

    def initialize_gate_lpbs(self, dut):
        self._gate_lpbs = {}

        self._gate_lpbs['I'] = dut.get_gate('I')
        self._gate_lpbs['Xp'] = dut.get_gate('Xp', transition_name='f01')
        self._gate_lpbs['Yp'] = dut.get_gate('Yp', transition_name='f01')
        self._gate_lpbs['X'] = dut.get_gate('X', transition_name='f01')

    @log_and_record
    def run(self, dut, mprim_index=0, initial_lpb=None, extra_measurement_duts=None):
        from leeq.theory.tomography import SingleQubitModel
        model = SingleQubitModel()
        super().run(dut, model, mprim_index, initial_lpb, extra_measurement_duts)


class MultiQubitsStateTomography(GeneralisedStateTomography, QubitTomographyBase):
    @log_and_record
    def run(self, duts, mprim_index=0, initial_lpb=None, extra_measurement_duts=None):
        from leeq.theory.tomography import MultiQubitModel
        model = MultiQubitModel(len(duts))
        super().run(duts, model, mprim_index, initial_lpb, extra_measurement_duts)


class MultiQubitsProcessTomography(GeneralisedProcessTomography, QubitTomographyBase):
    @log_and_record
    def run(self, duts, lpb, mprim_index=0, initial_lpb=None, extra_measurement_duts=None):
        from leeq.theory.tomography import MultiQubitModel
        model = MultiQubitModel(len(duts))
        super().run(duts=duts, model=model, lpb=lpb, mprim_index=mprim_index,
                    extra_measurement_duts=extra_measurement_duts)
