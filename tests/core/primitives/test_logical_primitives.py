import pytest

from leeq.core.primitives.logical_primitives import (
    LogicalPrimitive,
    LogicalPrimitiveBlockParallel,
    LogicalPrimitiveBlockSerial,
    LogicalPrimitiveBlockSweep,
)


class MockPrimitive(LogicalPrimitive):
    @staticmethod
    def _validate_parameters(parameters: dict):
        if "amp" not in parameters:
            raise ValueError("amp is required")


def test_clone_overrides_parameters_without_mutating_original():
    primitive = MockPrimitive("drive", {"amp": 0.5, "phase": 0.0})

    clone = primitive.clone_with_parameters({"amp": 0.25}, name_postfix="_half")
    clone.update_parameters(phase=0.5)

    assert clone._name == "drive_half"
    assert clone.amp == 0.25
    assert clone.phase == 0.5
    assert primitive.get_parameters() == {"amp": 0.5, "phase": 0.0}


def test_clone_rejects_unknown_parameter_keys():
    primitive = MockPrimitive("drive", {"amp": 0.5})

    with pytest.raises(ValueError, match="not a subset"):
        primitive.clone_with_parameters({"width": 0.02})


def test_shallow_copy_shares_parameters_but_not_tags():
    primitive = MockPrimitive("drive", {"amp": 0.5})
    primitive.tag(role="original")

    shallow = primitive.shallow_copy()
    shallow.update_parameters(amp=0.75)
    shallow.tag(role="copy")

    assert primitive.amp == 0.75
    assert primitive.tags == {"role": "original"}
    assert shallow.tags == {"role": "copy"}


def test_block_composition_flattens_matching_block_types():
    first = MockPrimitive("first", {"amp": 0.1})
    second = MockPrimitive("second", {"amp": 0.2})
    third = MockPrimitive("third", {"amp": 0.3})

    serial = (first + second) + (third + first.clone())
    parallel = (first * second) * (third * first.clone())

    assert isinstance(serial, LogicalPrimitiveBlockSerial)
    assert len(serial.children) == 4
    assert isinstance(parallel, LogicalPrimitiveBlockParallel)
    assert len(parallel.children) == 4


def test_sweep_block_exposes_only_selected_child_nodes():
    first = MockPrimitive("first", {"amp": 0.1})
    second = MockPrimitive("second", {"amp": 0.2})
    sweep = LogicalPrimitiveBlockSweep(children=[first, second])

    assert sweep.current_lpb is first
    assert sweep.nodes == first.nodes

    sweep.set_selected(1)

    assert sweep.current_lpb is second
    assert sweep.nodes == second.nodes
