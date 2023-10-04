import functools

import numpy as np
from functools import singledispatchmethod

from leeq.compiler.compiler_base import CompilerBase
from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory
from leeq.core.context import ExperimentContext
from leeq.core.primitives.built_in.common import PhaseShift
from leeq.core.primitives.logical_primitives import (LogicalPrimitiveCombinable,
                                                     LogicalPrimitiveBlockParallel,
                                                     LogicalPrimitiveBlockSerial,
                                                     LogicalPrimitiveBlockSweep,
                                                     MeasurementPrimitive
                                                     )

import numpy as np
from typing import Dict
from leeq.utils import setup_logging

logger = setup_logging(__name__)


class PulseSequence:
    """
    The PulseSequence class is used to store compiled pulse sequence shapes, allowing
    concatenation, addition, and slicing of the sequence in multiple channels.
    """

    def __init__(self, sequences: Dict[str, np.ndarray] = None):
        """
        Initialize the PulseSequence class.
        """
        self._sequences = sequences if sequences is not None else {}

    def _pad_and_combine(self, seq1, seq2, operation='add'):
        max_length = max(len(seq1), len(seq2))
        padded_seq1 = np.pad(seq1, (0, max_length - len(seq1)), mode='constant')
        padded_seq2 = np.pad(seq2, (0, max_length - len(seq2)), mode='constant')

        if operation == 'add':
            return padded_seq1 + padded_seq2
        elif operation == 'concat':
            return np.concatenate((padded_seq1, padded_seq2))
        else:
            raise ValueError("Unsupported operation")

    def _combine_sequences(self, other, operation):
        new_sequences = {}
        all_keys = set(self._sequences.keys()).union(other._sequences.keys())

        for key in all_keys:
            seq1 = self._sequences.get(key, np.array([], dtype=np.complex64))
            seq2 = other._sequences.get(key, np.array([], dtype=np.complex64))

            if operation == 'add':
                new_sequences[key] = self._pad_and_combine(seq1, seq2, 'concat')
            elif operation == 'mul':
                new_sequences[key] = self._pad_and_combine(seq1, seq2, 'add')
            else:
                raise ValueError("Unsupported operation")

        # Make sure all the sequences have the same length
        max_length = 0
        for key in new_sequences:
            max_length = max(max_length, len(new_sequences[key]))

        for key in new_sequences:
            new_sequences[key] = np.pad(new_sequences[key], (0, max_length - len(new_sequences[key])), mode='constant')

        return PulseSequence(new_sequences)

    def get_sequences(self):
        """
        Get the sequences.
        """
        return self._sequences

    def __add__(self, other):
        return self._combine_sequences(other, 'add')

    def __mul__(self, other):
        return self._combine_sequences(other, 'mul')

    def __len__(self):
        if len(self._sequences) == 0:
            return 0
        else:
            return len(next(iter(self._sequences.values())))


class MeasurementSequence:
    """
    The MeasurementSequence class is used to annotate when to start the aquisition of the signal.
    """

    def __init__(self):
        """
        Initialize the MeasurementSequence class.
        """
        self._measurements = []

    def add_measurement(self, position, channel, tags):
        """
        Add a measurement to the measurement sequence.
        """
        self._measurements.append((position, channel, tags))

    def get_measurements(self):
        """
        Get the measurements.
        """
        return self._measurements


class FullSequencingCompilerBak(CompilerBase):

    def __init__(self, sampling_rate: dict[float]):
        """
        Initialize the FullSequencingCompiler class.

        Parameters:
            sampling_rate (Dict[float]): The sampling rate of the compiler, indexed by the channel number.
                In Mega samples per second.
        """
        super().__init__("FullSequencingCompiler: Msps" + str(sampling_rate))
        self._measurement_sequence = None
        self._phase_shift = None
        self._sampling_rate = sampling_rate

    def compile_lpb(self, context: ExperimentContext, lpb: LogicalPrimitiveCombinable):
        """
        Compile the logical primitive block to instructions that going to be passed to the compiler.
        The compiled instructions are a buffer of pulse shape.
        """
        self._measurement_sequence = MeasurementSequence()
        self._phase_shift = {}

        compiled_sequence = self._compile_lpb(lpb, 0)

        context.instructions = {
            "pulse_sequence": compiled_sequence.get_sequences(),
            "measurement_sequence": self._measurement_sequence.get_measurements()
        }

        self._measurement_sequence = None
        self._phase_shift = None
        return context

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
            # If the logical primitive is a measurement primitive, then return the measurement sequence
            self._measurement_sequence.add_measurement(current_position, (pulse_channel, lpb.freq), lpb.tags)

        return PulseSequence(sequences={(pulse_channel, lpb.freq): pulse_shape})

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

        return PulseSequence()

    @_compile_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockParallel, current_position: int):
        sequences_list = [
            self._compile_lpb(child, current_position) for child in lpb.children
        ]

        return functools.reduce(lambda x, y: x * y, sequences_list)

    @_compile_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockSerial, current_position: int):
        sequence = PulseSequence()
        for i in range(len(lpb.children)):
            next_sequence = self._compile_lpb(lpb.children[i], current_position)
            sequence = sequence + next_sequence
            current_position = len(sequence)

        return sequence

    @_compile_lpb.register
    def _(self, lpb: LogicalPrimitiveBlockSweep, current_position: int):
        return self._compile_lpb(lpb.children[0], current_position)

    def commit_measurement(self, context: ExperimentContext):
        """
        Commit the measurement result to the measurement primitives.
        """
        pass


class FullSequencingCompiler(CompilerBase):

    def __init__(self, sampling_rate: dict[float]):
        """
        Initialize the FullSequencingCompiler class.

        Parameters:
            sampling_rate (Dict[float]): The sampling rate of the compiler, indexed by the channel number.
                In Mega samples per second.
        """
        super().__init__("FullSequencingCompiler: Msps" + str(sampling_rate))
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
            'pulse_sequence': self._assemble_pulse_fragments()
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
            # If the logical primitive is a measurement primitive, then return the measurement sequence
            self._measurement_sequence.add_measurement(current_position, (pulse_channel, lpb.freq), lpb.tags)

        self._pulse_fragments.append(((pulse_channel, lpb.freq), current_position, pulse_shape))

        length = len(pulse_shape)  # Always 1D

        self._update_lengths(pulse_channel, lpb.freq, length + current_position)

        return length

    def _assemble_pulse_fragments(self):
        """
        Assemble the pulse fragments into a full pulse sequence.
        """

        sequences = {}

        max_time_span = max([v / self._sampling_rate[channel] for (channel, freq), v in self._lengths.items()])

        for (channel, freq), v in self._lengths.items():
            sequences[(channel, freq)] = np.zeros(int(max_time_span * self._sampling_rate[channel]), dtype=np.complex64)

        for (channel, freq), position, pulse_shape_data in self._pulse_fragments:
            sequences[(channel, freq)][position:position + len(pulse_shape_data)] += pulse_shape_data

        return sequences

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
        return self._compile_lpb(lpb.children[0], current_position)

    def commit_measurement(self, context: ExperimentContext):
        """
        Commit the measurement result to the measurement primitives.
        """
        pass
