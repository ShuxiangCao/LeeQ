import pytest
import uuid
from leeq.core.primitives.logical_primitives import (LogicalPrimitive, LogicalPrimitiveBlockParallel,
                                                     LogicalPrimitiveBlockSerial, LogicalPrimitiveFactory)


class MockLogicalPrimitive(LogicalPrimitive):

    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)

    @staticmethod
    def _validate_parameters(parameters: dict):
        pass


def test_logical_primitive_creation():
    lp = MockLogicalPrimitive(name='primitive', parameters={'param1': 1, 'param2': 2})
    assert lp._name == 'primitive'
    assert lp._parameters == {'param1': 1, 'param2': 2}


def test_logical_primitive_clone():
    lp = MockLogicalPrimitive(name='primitive', parameters={'param1': 1, 'param2': 2})
    cloned_lp = lp.clone()
    assert cloned_lp._name != lp._name
    assert cloned_lp._parameters == lp._parameters


def test_logical_primitive_block_serial_addition():
    lp1 = MockLogicalPrimitive(name='primitive1', parameters={})
    lp2 = MockLogicalPrimitive(name='primitive2', parameters={})
    block_serial = lp1 + lp2
    assert isinstance(block_serial, LogicalPrimitiveBlockSerial)
    assert block_serial._children == [lp1, lp2]


def test_logical_primitive_block_parallel_multiplication():
    lp1 = MockLogicalPrimitive(name='primitive1', parameters={})
    lp2 = MockLogicalPrimitive(name='primitive2', parameters={})
    block_parallel = lp1 * lp2
    assert isinstance(block_parallel, LogicalPrimitiveBlockParallel)
    assert block_parallel._children == [lp1, lp2]


def test_logical_primitive_block_parallel_clone():
    lp1 = MockLogicalPrimitive(name='primitive1', parameters={})
    lp2 = MockLogicalPrimitive(name='primitive2', parameters={})
    block_parallel = lp1 * lp2
    cloned_block = block_parallel.clone()
    assert cloned_block._name != block_parallel._name
    assert cloned_block._children != block_parallel._children
    assert all(cloned_child._name != child._name for cloned_child, child in
               zip(cloned_block._children, block_parallel._children))


def test_logical_primitive_block_serial_clone():
    lp1 = MockLogicalPrimitive(name='primitive1', parameters={})
    lp2 = MockLogicalPrimitive(name='primitive2', parameters={})
    block_serial = lp1 + lp2
    cloned_block = block_serial.clone()
    assert cloned_block._name != block_serial._name
    assert cloned_block._children != block_serial._children
    assert all(cloned_child._name != child._name for cloned_child, child in
               zip(cloned_block._children, block_serial._children))


def test_logical_primitive_add_assertion():
    lp1 = MockLogicalPrimitive(name='primitive1', parameters={})
    with pytest.raises(AssertionError):
        lp1 + 5  # Non LogicalPrimitiveCombinable object


def test_logical_primitive_mul_assertion():
    lp1 = MockLogicalPrimitive(name='primitive1', parameters={})
    with pytest.raises(AssertionError):
        lp1 * 5  # Non LogicalPrimitiveCombinable object
