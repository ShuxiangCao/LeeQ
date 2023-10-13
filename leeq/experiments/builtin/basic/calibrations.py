import numpy as np
from scipy import optimize as so
import plotly.graph_objects as go
from labchronicle import log_and_record, register_browser_function
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.core.primitives.built_in.common import *
from leeq.experiments.sweeper import SweepParametersSideEffectFactory
from leeq import Experiment, Sweeper, ExperimentManager


class ResonatorSweepTransmissionWithExtraInitialLPB(Experiment):
    """
    Class representing a resonator sweep transmission experiment with extra initial LPB.
    Inherits from a generic "experiment" class.
    """

    @log_and_record
    def run(self,
            dut_qubit: TransmonElement,
            start: float = 8000,
            stop: float = 9000,
            step: float = 5.0,
            num_avs: int = 1000,
            rep_rate: float = 10.0,
            mp_width: float = None,
            initial_lpb=None,
            update: bool = True,
            amp: float = 1.0) -> None:
        """
        Run the resonator sweep transmission experiment.

        The initial lpb is for exciting the qubit to a different state.

        Parameters:
            dut_qubit: The device under test (DUT) qubit.
            start (float): Start frequency for the sweep. Default is 8000.
            stop (float): Stop frequency for the sweep. Default is 9000.
            step (float): Frequency step for the sweep. Default is 5.0.
            num_avs (int): Number of averages. Default is 1000.
            rep_rate (float): Repetition rate. Default is 10.0.
            mp_width (float): Measurement pulse width. If None, uses rep_rate. Default is None.
            initial_lpb: Initial linear phase behavior (LPB). Default is None.
            update (bool): Whether to update. Default is True.
            amp (float): Amplitude. Default is 1.0.
        """
        # Sweep the frequency
        mp = dut_qubit.get_default_measurement_prim_intlist().clone()

        # Update pulse width
        mp.update_pulse_args(width=rep_rate) if mp_width is None else mp.update_pulse_args(width=mp_width)
        if amp is not None:
            mp.update_pulse_args(amp=amp)

        mp.set_transform_function(None)

        lpb = initial_lpb + mp if initial_lpb is not None else mp

        # Define sweeper
        swp = Sweeper(
            np.arange,
            n_kwargs={"start": start, "stop": stop, "step": step},
            params=[SweepParametersSideEffectFactory.func(mp.update_freq, {}, "freq", name='frequency')],
        )

        # Perform the experiment
        with ExperimentManager().status().with_parameters(
                shot_number=num_avs,
                shot_period=rep_rate,
                acquisition_type='IQ_average'
        ):
            ExperimentManager().run(lpb, swp)

        result = np.squeeze(mp.result())

        # Save results
        self.result = {
            "Magnitude": np.absolute(result),
            "Phase": np.angle(result),
            "Real": np.real(result),
            "Imaginary": np.imag(result),
        }
