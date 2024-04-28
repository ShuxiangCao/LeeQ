import numpy as np
from labchronicle import log_and_record, register_browser_function
from matplotlib import pyplot as plt

from leeq import Experiment, basic_run, Sweeper
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
