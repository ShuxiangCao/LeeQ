import pytest
import numpy as np

from leeq.compiler.full_sequecing.full_sequencing import PulseSequence, MeasurementSequence


# Test cases for PulseSequence
def test_pulse_sequence_initialization():
    pulse_seq = PulseSequence()
    assert pulse_seq.get_sequences() == {}


def test_pulse_sequence_concatenate():
    seq1 = PulseSequence({"ch1": np.array([1, 2, 3], dtype=np.complex64)})
    seq2 = PulseSequence({"ch1": np.array([4, 5, 6], dtype=np.complex64)})
    added_seq = seq1 + seq2
    np.testing.assert_array_equal(added_seq.get_sequences()["ch1"], np.array([1, 2, 3, 4, 5, 6], dtype=np.complex64))


def test_pulse_sequence_parallel():
    seq1 = PulseSequence({"ch1": np.array([1, 2, 3], dtype=np.complex64)})
    seq2 = PulseSequence({"ch1": np.array([4, 5, 6], dtype=np.complex64)})
    multiplied_seq = seq1 * seq2
    np.testing.assert_array_equal(multiplied_seq.get_sequences()["ch1"], np.array([5, 7, 9], dtype=np.complex64))


def test_pulse_sequence_parallel():
    seq1 = PulseSequence({"ch1": np.array([1, 2, 3], dtype=np.complex64)})
    seq2 = PulseSequence({"ch1": np.array([4, 5, 6], dtype=np.complex64)})
    multiplied_seq = seq1 * seq2
    np.testing.assert_array_equal(multiplied_seq.get_sequences()["ch1"], np.array([5, 7, 9], dtype=np.complex64))


def test_pulse_sequence_parallel_different_length():
    seq1 = PulseSequence({"ch1": np.array([1, 2, 3, 2, 0], dtype=np.complex64)})
    seq2 = PulseSequence({"ch1": np.array([4, 5, 6], dtype=np.complex64)})
    multiplied_seq = seq1 * seq2
    np.testing.assert_array_equal(multiplied_seq.get_sequences()["ch1"], np.array([5, 7, 9, 2, 0], dtype=np.complex64))


def test_pulse_sequence_length():
    seq = PulseSequence({"ch1": np.array([1, 2, 3], dtype=np.complex64)})
    assert len(seq) == 3


def test_pulse_sequence_no_channel_overlap_addition():
    seq1 = PulseSequence({"ch1": np.array([1, 2, 3], dtype=np.complex64)})
    seq2 = PulseSequence({"ch2": np.array([4, 5, 6], dtype=np.complex64)})
    combined_seq = seq1 + seq2
    assert set(combined_seq.get_sequences().keys()) == {"ch1", "ch2"}


def test_invalid_operation():
    seq1 = PulseSequence({"ch1": np.array([1, 2, 3], dtype=np.complex64)})
    with pytest.raises(ValueError):
        seq1._pad_and_combine(seq1.get_sequences()["ch1"], seq1.get_sequences()["ch1"], operation='invalid_op')


# Test cases for MeasurementSequence
def test_measurement_sequence_initialization():
    ms = MeasurementSequence()
    assert ms.get_measurements() == []


def test_add_measurement():
    ms = MeasurementSequence()
    ms.add_measurement(0, 'ch1', ['tag1', 'tag2'])
    assert ms.get_measurements() == [(0, 'ch1', ['tag1', 'tag2'])]
