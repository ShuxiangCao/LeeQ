import numpy as np
import pytest

from leeq.compiler.lbnl_qubic.circuit_list_compiler import compare_dicts, segment_array


def test_compare_dicts_accepts_nested_numeric_values_with_tolerance():
    left = {"pulse": {"amp": 0.1 + 1e-8, "phase": 0.25}, "shape": "square"}
    right = {"pulse": {"amp": 0.1, "phase": 0.25}, "shape": "square"}

    assert compare_dicts(left, right)


def test_compare_dicts_rejects_key_and_value_mismatches():
    assert not compare_dicts({"amp": 0.1}, {"amp": 0.2})
    assert not compare_dicts({"amp": 0.1}, {"phase": 0.1})


def test_compare_dicts_requires_dictionary_inputs():
    with pytest.raises(ValueError, match="Both inputs should be dictionaries"):
        compare_dicts({"amp": 0.1}, [("amp", 0.1)])


def test_segment_array_splits_flat_and_changing_regions():
    data = np.array([0, 0, 0, 1, 2, 3, 3, 3, 3], dtype=float)

    flat_regions, changing_regions = segment_array(data, threshold=0.01, min_flat_length=2)

    assert flat_regions == [(0, 3), (5, 9)]
    assert changing_regions == [(3, 5)]


def test_segment_array_merges_short_flat_regions_into_changes():
    data = np.array([0, 1, 1, 2, 3, 3, 3], dtype=float)

    flat_regions, changing_regions = segment_array(data, threshold=0.01, min_flat_length=3)

    assert flat_regions == [(4, 7)]
    assert changing_regions == [(0, 3), (3, 4)]
