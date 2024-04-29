import copy

import numpy as np
from labchronicle import log_and_record, register_browser_function
from matplotlib import pyplot as plt

from leeq import Experiment, basic_run, Sweeper
from leeq.theory.utils import to_dense_probabilities
from leeq.utils.compatibility import prims
from .base import GeneralisedTomographyBase
from ... import sweeper


class GeneralisedSingleDutStateTomography(Experiment, GeneralisedTomographyBase):
    def run(self, dut, model, mprim_index=1, initial_lpb=None, extra_measurement_duts=None):
        self.model = model
        self.initialize_gate_lpbs(dut=dut)

        self.state_tomography_model = self.model.construct_state_tomography()

        measurement_sequences = self.state_tomography_model.get_measurement_sequence()

        measurement_lpb = prims.SweepLPB([self.get_lpb(x) for x in measurement_sequences])
        swp_measurement = Sweeper.from_sweep_lpb(measurement_lpb)

        mprim = dut.get_measurement_prim_intlist(mprim_index)

        lpb = measurement_lpb + mprim

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        basic_run(lpb, swp_measurement, 'prob')
        self.result = np.squeeze(mprim.result())

        self.vec, self.dm = self.state_tomography_model.linear_inverse_state_tomography(self.result.T)

    @register_browser_function(available_after=(run,))
    def plot(self):
        self.model.plot_density_matrix(self.dm)
        plt.show()


class GeneralisedSingleDutProcessTomography(Experiment, GeneralisedTomographyBase):

    @log_and_record
    def run(self, dut, model, lpb=None, mprim_index=1, extra_measurement_duts=None):
        self.model = model
        self.initialize_gate_lpbs(dut=dut)

        self.process_tomography_model = self.model.construct_process_tomography()

        measurement_sequences = self.process_tomography_model.get_measurement_sequence()
        preparation_sequences = self.process_tomography_model.get_preparation_sequence()

        measurement_lpb = prims.SweepLPB([self.get_lpb(x) for x in measurement_sequences])
        swp_measurement = Sweeper.from_sweep_lpb(measurement_lpb)

        preparation_lpb = prims.SweepLPB([self.get_lpb(x) for x in preparation_sequences])
        swp_preparation = Sweeper.from_sweep_lpb(preparation_lpb)

        mprim = dut.get_measurement_prim_intlist(mprim_index)

        if lpb is not None:
            lpb = preparation_lpb + lpb + measurement_lpb + mprim
        else:
            lpb = preparation_lpb + measurement_lpb + mprim

        basic_run(lpb, swp_measurement + swp_preparation, 'prob')
        self.result = mprim.result()

        self.ptm = self.process_tomography_model.linear_inverse_process_tomography(self.result.transpose([0, 2, 1]))

    @register_browser_function(available_after=(run,))
    def plot(self):
        self.model.plot_process_matrix(self.ptm)
        plt.show()


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


class GeneralisedStateTomography(Experiment, GeneralisedTomographyBase):
    def run(self, duts, model, mprim_index=1, initial_lpb=None, extra_measurement_duts=None):
        self.model = model
        self.initialize_gate_lpbs(duts=duts)

        self.state_tomography_model = self.model.construct_state_tomography()

        measurement_sequences = self.state_tomography_model.get_measurement_sequence()

        measurement_lpb = prims.SweepLPB([self.get_lpb(x) for x in measurement_sequences])
        swp_measurement = Sweeper.from_sweep_lpb(measurement_lpb)

        mprims = [dut.get_measurement_prim_intlist(mprim_index) for dut in duts]

        lpb = measurement_lpb + prims.ParallelLPB(mprims)

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        basic_run(lpb, swp_measurement, '<zs>')
        self.result = np.squeeze([mprim.result() for mprim in mprims])

        self.prob = to_dense_probabilities(self.result.transpose([0, 2, 1]))

        self.vec, self.dm = self.state_tomography_model.linear_inverse_state_tomography(self.prob)

    @register_browser_function(available_after=(run,))
    def plot(self):
        self.model.plot_density_matrix(self.dm)
        plt.show()


class GeneralisedProcessTomography(Experiment, GeneralisedTomographyBase):

    @log_and_record
    def run(self, duts, model, lpb=None, mprim_index=1, extra_measurement_duts=None):
        self.model = model
        self.initialize_gate_lpbs(duts=duts)

        self.process_tomography_model = self.model.construct_process_tomography()

        measurement_sequences = self.process_tomography_model.get_measurement_sequence()
        preparation_sequences = self.process_tomography_model.get_preparation_sequence()

        measurement_lpb = prims.SweepLPB([self.get_lpb(x) for x in measurement_sequences])
        swp_measurement = Sweeper.from_sweep_lpb(measurement_lpb)

        preparation_lpb = prims.SweepLPB([self.get_lpb(x) for x in preparation_sequences])
        swp_preparation = Sweeper.from_sweep_lpb(preparation_lpb)

        mprims = [dut.get_measurement_prim_intlist(mprim_index) for dut in duts]

        if lpb is not None:
            lpb = preparation_lpb + lpb + measurement_lpb + prims.ParallelLPB(mprims)
        else:
            lpb = preparation_lpb + measurement_lpb + prims.ParallelLPB(mprims)

        basic_run(lpb, swp_measurement + swp_preparation, '<zs>')
        self.result = np.squeeze([mprim.result() for mprim in mprims])

    def analyze_data(self):

        self.prob = to_dense_probabilities(self.result.transpose([0, 3, 1, 2]))
        self.ptm = self.process_tomography_model.linear_inverse_process_tomography(self.prob)

    @register_browser_function(available_after=(run,))
    def plot(self):
        self.analyze_data()
        self.model.plot_process_matrix(self.ptm, base=2)
        plt.show()


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
