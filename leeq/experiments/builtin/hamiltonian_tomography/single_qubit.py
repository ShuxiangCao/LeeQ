from typing import List, Union, Optional

import numpy as np
from labchronicle import log_and_record
from leeq import Experiment, Sweeper, basic_run, SweepParametersSideEffectFactory
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock, LogicalPrimitive, \
    LogicalPrimitiveBlockParallel, LogicalPrimitiveBlockSweep
from leeq.utils import setup_logging

logger = setup_logging(__name__)


class HamiltonianTomographySingleQubitBase(Experiment):
    """
    Base class for implementing Hamiltonian tomography.

    """

    def _run(self,
             duts: List[TransmonElement],
             lpb: LogicalPrimitiveBlock,
             swp: Sweeper,
             tomography_axis: Union[str, List[str]],
             initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]] = None,
             collection_name='f01',
             mprim_index='0'
             ):
        """
        This method runs the tomography experiment.

        Prameters:
        ----------
        duts: List[TransmonElement]
            List of transmon elements to be characterized.
        lpb: LogicalPrimitiveBlock
            Logical primitive block to be used for the experiment, describe the pulse that generates the
             Hamiltonian to be characterized.
        swp: Sweeper
            Sweeper object to be used for the experiment.
        tomography_axis: Union[str, List[str]]
            Axis along which the Hamiltonian tomography is to be performed. Must be one of 'X', 'Y', 'Z'.
        initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]]
            Initial logical primitive block to be used for the experiment.
        collection_name: str
            Name of the collection (transition frequency) to be used for the experiment.
        """

        if isinstance(tomography_axis, str):
            tomography_axis = [tomography_axis]

        if not isinstance(tomography_axis, list):
            msg = 'tomography_axis must be a list of strings or a string'
            logger.error(msg)
            raise ValueError(msg)

        def get_tomography_lpb(dut, axis):
            """Get the LPB that rotates the qubit measurement axis to Z axis."""
            assert axis in ['X', 'Y', 'Z']
            c1 = dut.get_c1(collection_name)
            if axis == 'X':
                return c1['Yp']
            elif axis == 'Y':
                return c1['Xm']
            elif axis == 'Z':
                return c1['I']

        tomography_lpb = []

        for axis in tomography_axis:
            if axis not in ['X', 'Y', 'Z']:
                msg = 'Axis must be one of X, Y, Z. Got: {}'.format(axis)
                logger.error(msg)
                raise ValueError(msg)
            tomography_lpb.append(LogicalPrimitiveBlockParallel([get_tomography_lpb(dut, axis) for dut in duts]))

        mprims = [dut.get_measurement_prim_intlist(mprim_index) for dut in duts]

        tomography_lpb = LogicalPrimitiveBlockSweep(tomography_lpb)
        swp_tomo = Sweeper.from_sweep_lpb(tomography_lpb)

        self._tomography_axis = tomography_axis
        self._duts = duts

        lpb_run = lpb + tomography_lpb + LogicalPrimitiveBlockParallel(mprims)

        if initial_lpb is not None:
            lpb_run = initial_lpb + lpb_run

        basic_run(lpb_run, swp + swp_tomo, "<zs>")

        self.results = [mprim.results() for mprim in mprims]


class HamiltonianTomographySingleQubitStarkShift(HamiltonianTomographySingleQubitBase):
    """
    Hamiltonian tomography reveals the Z term under the stark shift drive.
    """

    @log_and_record
    def run(self,
            dut: TransmonElement,
            stark_amp: float,
            stark_freq: float,
            start_time: float,
            stop_time: float,
            step_time: float,
            initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]] = None,
            collection_name: str = 'f01',
            mprim_index: str = '0'
            ):
        """
        This method runs the stark shift tomography experiment.

        Parameters:
        -----------
        dut: TransmonElement
            Transmon element to be characterized.
        start_time: float
            Start time of the stark shift drive.
        stop_time: float
            Stop time of the stark shift drive.
        step_time: float
            Step time of the stark shift drive.
        stark_amp: float
            Amplitude of the stark shift drive.
        stark_freq: float
            Frequency of the stark shift drive.
        initial_lpb: Optional[Union[LogicalPrimitiveBlock, LogicalPrimitive]]
            Initial logical primitive block to be used for the experiment.
        collection_name: str
            Name of the collection (transition frequency) to be used for the experiment.
        mprim_index: str
            Index of the measurement primitive to be used for the experiment.
        """

        # Get c1 from the DUT qubit
        c1 = dut.get_c1(collection_name)
        stark_shift_drive = c1['X'].clone()

        parameters = {
            'freq': stark_freq,
            'shape': 'square',
            'phase': 0.,
            'amp': stark_amp
        }

        stark_shift_drive.update_pulse_args(**parameters)

        # Set up sweep parameters
        swpparams = [SweepParametersSideEffectFactory.func(
            stark_shift_drive.update_pulse_args, {}, 'width'
        )]

        swp = Sweeper(
            np.arange,
            n_kwargs={'start': start_time, 'stop': stop_time, 'step': step_time},
            params=swpparams
        )

        initial_preparation_lpb = dut.get_c1(collection_name)['Ym']  # Start by moving the bloch vector to X axis

        if initial_lpb is not None:
            initial_preparation_lpb = initial_lpb + initial_preparation_lpb

        self._run(duts=[dut],
                  lpb=stark_shift_drive,
                  swp=swp,
                  tomography_axis=['X', 'Y'],
                  initial_lpb=initial_preparation_lpb,
                  collection_name=collection_name,
                  mprim_index=mprim_index
                  )
