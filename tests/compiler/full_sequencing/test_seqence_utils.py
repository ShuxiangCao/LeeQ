from leeq.compiler.full_sequencing_compiler import MeasurementSequence


# Test cases for MeasurementSequence
def test_measurement_sequence_initialization():
    ms = MeasurementSequence()
    assert ms.get_measurements() == []


def test_add_measurement():
    ms = MeasurementSequence()
    ms.add_measurement(0, 'ch1', ['tag1', 'tag2'] )
    assert ms.get_measurements() == [(0, 'ch1', ['tag1', 'tag2'])]
