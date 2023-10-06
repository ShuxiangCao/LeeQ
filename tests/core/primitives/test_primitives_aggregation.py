import uuid
import pytest
from pytest import fixture

from leeq.core import LogicalPrimitive
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock, LogicalPrimitiveBlockParallel, \
    LogicalPrimitiveBlockSerial


class MockLogicalPrimitive(LogicalPrimitive):

    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)

    @staticmethod
    def _validate_parameters(parameters: dict):
        pass


@pytest.fixture
def sample_logical_primitive():
    return MockLogicalPrimitive(name="SamplePrimitive", parameters={"param1": 1, "param2": 2})


@pytest.fixture
def sample_logical_primitive_block(sample_logical_primitive):
    return LogicalPrimitiveBlock(name="SampleBlock", children=[sample_logical_primitive])


@pytest.fixture
def sample_logical_primitive_block_parallel(sample_logical_primitive):
    return LogicalPrimitiveBlockParallel(children=[sample_logical_primitive])


def test_node_access_via_uuid(sample_logical_primitive_block):
    # Extract a UUID from the block's children
    some_uuid = list(sample_logical_primitive_block.nodes.keys())[0]

    # Verify that we can use it to access the corresponding node
    assert sample_logical_primitive_block.nodes[some_uuid] == sample_logical_primitive_block.children[0]


def test_nodes_aggregation_in_block(sample_logical_primitive):
    # Create another primitive and add it to the block
    another_primitive = MockLogicalPrimitive(name="AnotherPrimitive", parameters={"param3": 3, "param4": 4})

    sample_logical_primitive_block_parallel = sample_logical_primitive * another_primitive

    # Make sure the new primitive's UUID and node are present in the block's nodes dict
    assert another_primitive.uuid in sample_logical_primitive_block_parallel.nodes
    assert sample_logical_primitive_block_parallel.nodes[another_primitive.uuid] == another_primitive

    # Also check that the existing children are still accessible
    assert sample_logical_primitive.uuid in sample_logical_primitive_block_parallel.nodes
    assert sample_logical_primitive_block_parallel.nodes[sample_logical_primitive.uuid] == sample_logical_primitive


configuration = {
    'lpb_collections': {
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 4144.417053428905,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 0.025,
            'alpha': 425.1365229849309,
            'trunc': 1.2
        },
        'f12': {
            'type': 'SimpleDriveCollection',
            'freq': 4144.417053428905,
            'channel': 2,
            'shape': 'blackman_drag',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 0.025,
            'alpha': 425.1365229849309,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9144.41,
            'channel': 1,
            'shape': 'square',
            'amp': 0.21323904814245054 / 5 * 4,
            'phase': 0.,
            'width': 1,
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        }
    }
}


@fixture
def qubit():
    dut = TransmonElement(
        name='test_qubit',
        parameters=configuration
    )

    return dut


def test_add_primitive_to_serial_block(sample_logical_primitive):
    serial_block = sample_logical_primitive + sample_logical_primitive
    another_primitive = MockLogicalPrimitive(name="AnotherPrimitive", parameters={"param3": 3, "param4": 4})

    combined_block = serial_block + another_primitive

    # Ensure we can access the added primitive via UUID
    assert another_primitive.uuid in combined_block.nodes
    assert combined_block.nodes[another_primitive.uuid] == another_primitive


def test_multiply_primitive_to_parallel_block(sample_logical_primitive):
    parallel_block = sample_logical_primitive * sample_logical_primitive
    another_primitive = MockLogicalPrimitive(name="AnotherPrimitive", parameters={"param3": 3, "param4": 4})

    combined_block = parallel_block * another_primitive

    # Ensure we can access the multiplied primitive via UUID
    assert another_primitive.uuid in combined_block.nodes
    assert combined_block.nodes[another_primitive.uuid] == another_primitive


def test_add_serial_block_to_primitive(sample_logical_primitive, sample_logical_primitive_block):
    combined_block = sample_logical_primitive + sample_logical_primitive_block

    # Ensure we can access all nodes from the resulting block via UUID
    assert sample_logical_primitive.uuid in combined_block.nodes
    assert combined_block.nodes[sample_logical_primitive.uuid] == sample_logical_primitive

    child_uuid = list(sample_logical_primitive_block.nodes.keys())[0]
    assert child_uuid in combined_block.nodes
    assert combined_block.nodes[child_uuid] == sample_logical_primitive_block.children[0]


def test_add_serial_block_to_serial_block(sample_logical_primitive_block):
    another_serial_block = LogicalPrimitiveBlockSerial(
        children=[MockLogicalPrimitive(name="AnotherPrimitive", parameters={"param3": 3, "param4": 4})])
    combined_block = sample_logical_primitive_block + another_serial_block

    # Ensure all nodes from both blocks are accessible via UUID in the resulting block
    for uuid, node in {**sample_logical_primitive_block.nodes, **another_serial_block.nodes}.items():
        assert uuid in combined_block.nodes
        assert combined_block.nodes[uuid] == node


def test_multiply_parallel_block_to_parallel_block(sample_logical_primitive_block_parallel):
    another_parallel_block = LogicalPrimitiveBlockParallel(
        [MockLogicalPrimitive(name="AnotherPrimitive", parameters={"param3": 3, "param4": 4})])
    combined_block = sample_logical_primitive_block_parallel * another_parallel_block

    # Ensure all nodes from both blocks are accessible via UUID in the resulting block
    for uuid, node in {**sample_logical_primitive_block_parallel.nodes, **another_parallel_block.nodes}.items():
        assert uuid in combined_block.nodes
        assert combined_block.nodes[uuid] == node


def test_integration(qubit):
    lpb_1 = qubit.get_c1('f01')['X']
    lpb_2 = qubit.get_lpb_collection('f12')['Y']
    mprim_1 = qubit.get_measurement_prim_trace('0')
    mprim_2 = qubit.get_measurement_primitive('0')

    block = lpb_1 + lpb_2 * (mprim_1 * mprim_2)

    assert lpb_1.uuid in block.nodes
    assert lpb_2.uuid in block.nodes
    assert mprim_1.uuid in block.nodes
    assert mprim_2.uuid in block.nodes
    assert len(block.nodes) == 4
