import functools

import numpy as np
from functools import singledispatchmethod

from leeq.compiler.compiler_base import LPBCompiler, MeasurementSequence
from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory
from leeq.core.context import ExperimentContext
from leeq.core.primitives.built_in.common import PhaseShift
from leeq.core.primitives.logical_primitives import (LogicalPrimitiveCombinable,
                                                     LogicalPrimitiveBlockParallel,
                                                     LogicalPrimitiveBlockSerial,
                                                     LogicalPrimitiveBlockSweep,
                                                     MeasurementPrimitive, LogicalPrimitiveBlock
                                                     )

import numpy as np
from typing import Dict
from leeq.utils import setup_logging

logger = setup_logging(__name__)


class IndividualLPBCompiler(LPBCompiler):

    def __init__(self, sampling_rate: dict[float]):
        """
        Initialize the IndividualLPBCompiler class.

        Parameters:
            sampling_rate (Dict[float]): The sampling rate of the compiler, indexed by the channel number.
                In Mega samples per second.
        """
        super().__init__("IndividualLPBCompiler: Msps" + str(sampling_rate))
        self._measurement_sequence = None
        self._phase_shift = None
        self._sampling_rate = sampling_rate
        self._pulse_fragments = []
        self._lengths = {}

    def compile_lpb(self, context: ExperimentContext, lpb: LogicalPrimitiveCombinable):
        """
        Compile the logical primitive block to instructions that going to be passed to the compiler.
        The compiled instructions are a buffer of pulse shape.

        The way this compiler works is that it first compiles the logical primitive block into a list of pulse fragments,
        each of which is a tuple of (channel, frequency, position, pulse_shape). After that, the full size of the pulse
        sequence is calculated, and a large buffer is created to host the full pulse, and the pulse fragments are
        assembled into a full pulse sequence.
        """
        self._measurement_sequence = MeasurementSequence()
        self._phase_shift = {}

        self._compile_lpb(lpb, 0)
        context.instructions = {
            "measurement_sequence": self._measurement_sequence.get_measurements(),
            'pulse_sequence': self._pulse_fragments,
            'lengths': self._lengths,
        }

        self._measurement_sequence = None
        self._phase_shift = None
        return context

    def _update_lengths(self, channel, freq, length):
        """
        Update the lengths of a particular channel.

        Parameters:
            channel (int): The channel number.
            freq (float): The frequency of the pulse.
            length (int): The length of the pulse.
        """
        if (channel, freq) in self._lengths and self._lengths[(channel, freq)] > length:
            msg = "The pulse length is too short for channel " + str(channel) + " and frequency " + str(freq)
            msg += ". The length is " + str(length) + " while the previous length is " + str(
                self._lengths[(channel, freq)])
            logger.error(msg)
            raise ValueError(msg)

        self._lengths[(channel, freq)] = length

    @functools.singledispatchmethod
    def _compile_lpb(self, lpb: LogicalPrimitiveCombinable, current_position: int):
        """
        Compile the logical primitive block to instructions that going to be passed to the compiler.
        Recursively implement the compiling.

        Parameters:
            lpb (LogicalPrimitiveCombinable): The logical primitive block to compile.
            current_position (int): The current position (sample count) of the pulse sequence.

        Returns:
            PulseSequence: The compiled pulse sequence.
        """

        assert lpb.children is None, "The children of the logical primitive block should be None. Got class " + \
                                     str(lpb.__class__)

        # Get the parameters
        pulse_shape_name = lpb.shape
        pulse_channel = lpb.channel
        pulse_shape_parameters = lpb.get_parameters()

        if 'phase' in pulse_shape_parameters and pulse_channel in self._phase_shift and \
                pulse_shape_parameters['transition_name'] in self._phase_shift[pulse_channel]:
            pulse_shape_parameters['phase'] += self._phase_shift[pulse_channel][lpb.transition_name]

        # Compile the pulse shape
        factory = PulseShapeFactory()
        pulse_shape = factory.compile_pulse_shape(pulse_shape_name, sampling_rate=self._sampling_rate[lpb.channel],
                                                  **pulse_shape_parameters)

        if isinstance(lpb, MeasurementPrimitive):
            tags = lpb.tags
            tags.update(pulse_shape_parameters)
            tags.update({
                'uuid': lpb.uuid,
            })

            # If the logical primitive is a measurement primitive, then return the measurement sequence
            self._measurement_sequence.add_measurement(current_position, (pulse_channel, lpb.freq), tags)

        self._pulse_fragments.append(((pulse_channel, lpb.freq), current_position, pulse_shape, lpb.uuid))

        length = len(pulse_shape)  # Always 1D

        self._update_lengths(pulse_channel, lpb.freq, length + current_position)

        return length

    @_compile_lpb.register
    def _(self, lpb: PhaseShift, current_position: int):
        parameters = lpb.get_parameters()
        if lpb.channel not in self._phase_shift:
            self._phase_shift[lpb.channel] = {}

        for k, m in parameters['transition_multiplier'].items():
            if k not in self._phase_shift[lpb.channel]:
                self._phase_shift[lpb.channel][k] = m * parameters['phase_shift']
            else:
                self._phase_shift[lpb.channel][k] += m * parameters['phase_shift']

        return 0

    @_compile_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockParallel, current_position: int):
        lengths_list = [
            self._compile_lpb(child, current_position) for child in lpb.children
        ]

        length = max(lengths_list)

        return length

    @_compile_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockSerial, current_position: int):

        size = 0

        for i in range(len(lpb.children)):
            size += self._compile_lpb(lpb.children[i], current_position + size)

        return size

    @_compile_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockSweep, current_position: int):
        return self._compile_lpb(lpb.current_lpb, current_position)
