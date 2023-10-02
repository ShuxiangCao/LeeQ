import numpy as np
from labchronicle import log_and_record, register_browser_function

from leeq.experiments.experiments import Experiment


class DummyObject(object):
    def __init__(self):
        self.result = 0
        self.result_raw = 0


dummy_obj = DummyObject()


class SimpleSampleExperiment(Experiment):
    # Define the quantum element

    # Generate circuit
    @log_and_record
    def run(self, dut, lpb, mprim_index=0):
        mprim = dut.get_measurement_prim_intlist(mprim_index)

        basic(lpb + mprim, None, '<zs>')

        self.results = mprim.result()
        self.raw_values = mprim.result_raw()

    @register_browser_function
    def plot(self):
        dummy_obj.result = self.results
        dummy_obj.result_raw = self.raw_values


def test_pulse_on_qutip_pip():
    pass

# Prepare experiment setup for qutip pip

# Configure the global setup

# Run the experiment

# Check the result
