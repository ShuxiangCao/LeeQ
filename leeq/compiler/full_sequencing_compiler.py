import functools

import numpy as np
from functools import singledispatchmethod

from leeq.compiler.compiler_base import LPBCompiler, MeasurementSequence
from leeq.compiler.individual_lpb_compiler import IndividualLPBCompiler
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


class FullSequencingCompiler(LPBCompiler):

    def __init__(self, sampling_rate: dict[float]):
        """
        Initialize the FullSequencingCompiler class.

        Parameters:
            sampling_rate (Dict[float]): The sampling rate of the compiler, indexed by the channel number.
                In Mega samples per second.
        """
        super().__init__("FullSequencingCompiler: Msps" + str(sampling_rate))
        self._individual_lpb_compiler = IndividualLPBCompiler(sampling_rate)
        self._sampling_rate = sampling_rate

    def compile_lpb(self, context: ExperimentContext, lpb: LogicalPrimitiveCombinable):
        """
        Compile the logical primitive block to instructions that going to be passed to the compiler.
        The compiled instructions are a buffer of pulse shape.

        The way this compiler works is that it first compiles the logical primitive block into a list of pulse fragments,
        each of which is a tuple of (channel, frequency, position, pulse_shape). After that, the full size of the pulse
        sequence is calculated, and a large buffer is created to host the full pulse, and the pulse fragments are
        assembled into a full pulse sequence.
        """
        self._individual_lpb_compiler.compile_lpb(context, lpb)
        pulse_fragments = context.instructions['pulse_sequence']
        context.instructions['pulse_sequence'] = self._assemble_pulse_fragments(pulse_fragments,
                                                                                context.instructions['lengths'])

        return context

    def _assemble_pulse_fragments(self, pulse_fragments, lengths):
        """
        Assemble the pulse fragments into a full pulse sequence.
        """

        sequences = {}

        max_time_span = max([v / self._sampling_rate[channel] for (channel, freq), v in lengths.items()])

        for (channel, freq), v in lengths.items():
            sequences[(channel, freq)] = np.zeros(int(max_time_span * self._sampling_rate[channel] + 0.5),
                                                  dtype=np.complex64)

        for (channel, freq), position, pulse_shape_data, lp_uuid in pulse_fragments:
            sequences[(channel, freq)][position:position + len(pulse_shape_data)] += pulse_shape_data

        return sequences
