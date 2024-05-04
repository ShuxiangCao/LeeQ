import numpy as np
from labchronicle import register_browser_function, log_and_record
from matplotlib import pyplot as plt

from leeq import Experiment, Sweeper, basic_run
from leeq.theory.utils import to_dense_probabilities
from leeq.utils.compatibility import prims


class GeneralisedTomographyBase(object):

    def initialize_gate_lpbs(self, dut):
        pass

    def get_lpb(self, name):
        if isinstance(name, tuple) or isinstance(name, list):
            lpbs = [self.get_lpb(x) for x in name]
            lpb = prims.SeriesLPB(lpbs)
            return lpb

        return self._gate_lpbs[name]


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
