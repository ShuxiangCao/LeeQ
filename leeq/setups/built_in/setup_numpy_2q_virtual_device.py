import functools
from typing import List, Union, Dict
from uuid import UUID

import numpy as np

from leeq.compiler.individual_lpb_compiler import IndividualLPBCompiler
from leeq.core.context import ExperimentContext
from leeq.core.engine.measurement_result import MeasurementResult
from leeq.core.primitives.built_in.common import PhaseShift, DelayPrimitive
from leeq.core.primitives.logical_primitives import (
    LogicalPrimitiveBlock,
    LogicalPrimitiveBlockSerial,
    LogicalPrimitiveBlockSweep,
    LogicalPrimitiveBlockParallel,
    MeasurementPrimitive,
)
from leeq.experiments.sweeper import Sweeper
from leeq.setups.setup_base import ExperimentalSetup
from leeq.theory.simulation.numpy.rotated_frame_simulator import VirtualTransmon


class Numpy2QVirtualDeviceSetup(ExperimentalSetup):
    """
    The QuTipLocalSetup class defines a local setup for for using the qutip package to simulate the experiment
    at pulse level, at the local machine.
    """

    def __init__(self, sampling_rate=1e6):
        """
        Initialize the QuTipQIPLocalSetup class.

        Parameters:
            sampling_rate (float): The sampling rate of the experiment. In Msps unit.
        """
        name = "numpy_2q_fast_virtual_transmons"
        from leeq.core.engine.grid_sweep_engine import GridSerialSweepEngine

        self._compiler = IndividualLPBCompiler(
            sampling_rate={
                0: sampling_rate,
                1: sampling_rate,
                2: sampling_rate,
                3: sampling_rate,
            }
        )
        self._engine = GridSerialSweepEngine(
            compiler=self._compiler, setup=self, name=name + ".engine"
        )

        self._current_context = None
        self._sampling_rate = sampling_rate
        self._measurement_results: Dict[UUID, MeasurementResult] = {}
        self._uuid_to_qubit_shape = None
        self._simulators = {
            "q0": VirtualTransmon(
                name="q0",
                qubit_frequency=4144,
                anharmonicity=-198,
                t1=102,
                t2=120,
                readout_frequency=8818.23,
                readout_linewith=1,
                readout_dipsersive_shift=0.26,
                truncate_level=4,
                quiescent_state_distribution=[0.9, 0.07, 0.03, 0],
            ),
            "q1": VirtualTransmon(
                name="q1",
                qubit_frequency=4022,
                anharmonicity=-195,
                t1=88,
                t2=99,
                readout_frequency=9518.23,
                readout_linewith=1,
                readout_dipsersive_shift=0.5,
                truncate_level=4,
                quiescent_state_distribution=[0.85, 0.11, 0.04, 0],
            ),
        }

        super().__init__(name)

        self._status.add_channel(channel=0, name="q0_drive")
        self._status.add_channel(channel=1, name="q0_read")
        self._status.add_channel(channel=2, name="q1_drive")
        self._status.add_channel(channel=3, name="q1_read")

        self._channel_to_qubit = {
            0: "q0",
            1: "q0",
            2: "q1",
            3: "q1",
        }

    def run(self, lpb: LogicalPrimitiveBlock, sweep: Sweeper):
        """
        Run the experiment.

        The experiment run iterates all the parameters described by the sweeper. Each iteration can be break into
         four steps:

        1. Compile the measurement lpb to instructions that going to be passed to the compiler.
        2. Upload the instruction to the compiler, including changing frequencies of the generators etc. Get everything
            ready for the experiment.
        3. Fire the experiment and wait for it to finish.
        4. Collect the data from the compiler and commit it to the measurement primitives.

        So the setup should implement the following methods:
        2. `update_setup_parameters`: Update the setup parameters of the compiler. Usually this function calculates
            the frequencies of the generators etc and pass it to specific experiment setup class to further upload.
        3. `fire_experiment`: Fire the experiment and wait for it to finish.
        4. `collect_data`: Collect the data from the compiler and commit it to the measurement primitives.

        Note that the collected data will be committed to the measurement primitives by the engine, so the setup
            should not commit the data to the measurement primitives.

        The compiler should mainly implement the compile_lpb, like compiling the lpb into a format acceptable by the
        particular setup. This abstraction is to allow the compiler to be used in different setups. For example, the
        compiler can be used in a local setup, or a remote setup.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.
            sweep (Sweeper): The sweeper to use.
        """
        return self._engine.run(lpb, sweep)

    def _validate_lpb(self, lpb):
        """
        Validate the logical primitive block. This function is used to validate the logical primitive block before
        applying it to the simulator.
        """

        assert lpb.children is None, (
                "The children of the logical primitive block should be None. Got class "
                + str(lpb.__class__)
        )

        # Found the pulse shape etc

        if lpb.uuid not in self._uuid_to_qubit_shape:
            msg = f"Cannot find the pulse shape for uuid {lpb.uuid}"
            self.logger.error(msg)
            raise ValueError(msg)

        qubit, shape = self._uuid_to_qubit_shape[lpb.uuid]

        if lpb.channel not in self._channel_to_qubit:
            msg = f"Channel {lpb.channel} is not mapped to any qubit."
            self.logger.error(msg)
            raise ValueError(msg)

        qubit = self._channel_to_qubit[lpb.channel]

        if qubit not in self._simulators:
            msg = f"Qubit {qubit} is not found in the simulator."
            self.logger.error(msg)
            raise ValueError(msg)

    @functools.singledispatchmethod
    def _apply_lpb(self, lpb):
        """
        Apply the logical primitive block to the simulator, recursively.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to apply.
        """
        # These are gate primitives
        self._validate_lpb(lpb)

        qubit: VirtualTransmon = self._simulators[self._channel_to_qubit[lpb.channel]]

        qubit.apply_drive(
            frequency=lpb.freq,
            pulse_shape=self._uuid_to_qubit_shape[lpb.uuid][1],
            sampling_rate=int(self._sampling_rate),
        )

    @_apply_lpb.register
    def _(self, lpb: MeasurementPrimitive):
        """
        Apply the measurement primitive to the simulator.
        """
        self._validate_lpb(lpb)

        qubit: VirtualTransmon = self._simulators[
            self._channel_to_qubit[lpb.channel].split("_")[0]
        ]

        result = qubit.apply_readout(
            return_type=lpb.tags.get("return_value_type", "IQ_average"),
            sampling_number=self._status.get_parameters(key="Shot_Number"),
            iq_noise_std=1,
            trace_noise_std=1,
            readout_frequency=lpb.freq,
            readout_width=lpb.width,
            readout_shape=self._uuid_to_qubit_shape[lpb.uuid][1],
        )

        if isinstance(result, complex):
            result = np.asarray([result])

        if lpb.uuid not in self._measurement_results:
            measurement_result = MeasurementResult(
                step_no=self._current_context.step_no, data=result, mprim_uuid=lpb.uuid)
            self._measurement_results[lpb.uuid] = measurement_result
        else:
            measurement_result = self._measurement_results[lpb.uuid]
            measurement_result.append_data(result)

    @_apply_lpb.register
    def _(self, lpb: DelayPrimitive):
        pass

    @_apply_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockParallel):
        for i in range(len(lpb.children)):
            self._apply_lpb(lpb.children[i])

    @_apply_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockSerial):
        for i in range(len(lpb.children)):
            self._apply_lpb(lpb.children[i])

    @_apply_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockSweep):
        return self._apply_lpb(lpb.current_lpb)

    @_apply_lpb.register
    def _(self, lpb: PhaseShift):
        """
        Apply the phase shift to the simulator. Actually does nothing since phase has been applied in the compiler.
        """
        pass

    def update_setup_parameters(self, context: ExperimentContext):
        """
        Update the setup parameters of the compiler. It accepts the compiled instructions from the compiler, and update
        the local cache first. then use push_instrument_settings to push the settings to the instruments.

        Parameters:
            context (Any): The context between setup and compiler. Generated by the compiler.
        """
        self._current_context = context

        for simulator in self._simulators.values():
            simulator.reset_state()

        self._uuid_to_qubit_shape = {}

        for (channel, freq), pos, shape_buffer, lp_uuid in context.instructions[
            "pulse_sequence"
        ]:
            if channel not in self._channel_to_qubit:
                msg = f"Channel {channel} is not mapped to any qubit."
                self.logger.error(msg)
                raise ValueError(msg)

            self._uuid_to_qubit_shape[lp_uuid] = (
                self._channel_to_qubit[channel],
                shape_buffer.shape,
            )

    def fire_experiment(self, context=None):
        """
        Fire the experiment and wait for it to finish.
        """

        if context is not None:
            self._current_context = context

        self._apply_lpb(context.lpb)

    def collect_data(self, context: ExperimentContext):
        """
        Collect the data from the compiler and commit it to the measurement primitives.
        """

        if context is not None:
            self._current_context = context

        self._current_context.results = [
            x for x in self._measurement_results.values()]

        return context
