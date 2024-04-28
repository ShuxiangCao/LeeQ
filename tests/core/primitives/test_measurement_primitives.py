import pytest
from leeq.core.primitives.logical_primitives import MeasurementPrimitive


class MockMeasurementPrimitive(MeasurementPrimitive):

    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)

    @staticmethod
    def _validate_parameters(parameters: dict):
        pass


@pytest.fixture
def primitive():
    return MockMeasurementPrimitive(name="test_primitive", parameters={})


def test_initialization(primitive):
    assert isinstance(primitive, MeasurementPrimitive)
    assert primitive.get_default_result_id() == 0
    assert primitive.get_result_id_offset() == 0


def test_transform_function_setting_and_getting(primitive):
    def transform_func(data, factor):
        return data * factor

    primitive.set_transform_function(transform_func, factor=2)
    func, kwargs = primitive.get_transform_function()
    assert callable(func)
    assert kwargs == {'factor': 2}


def test_default_result_id_setting_and_getting(primitive):
    primitive.set_default_result_id(1)
    assert primitive.get_default_result_id() == 1


def test_result_id_offset_setting_and_getting(primitive):
    primitive.set_result_id_offset(1)
    assert primitive.get_result_id_offset() == 1


def test_commit_and_result_methods(primitive):
    import numpy as np
    data_1 = np.array([1, 2, 3])
    data_2 = np.array([4, 5, 6])

    with pytest.raises(RuntimeError):
        primitive.commit_measurement(data=data_1.reshape([1, 3]), indices=(0,))

    # The shape should be [sweep_shape, result_ids ,data_shape]
    # For this example, it should be [2, 1, 3]

    primitive.allocate_measurement_buffer(
        sweep_shape=[2, ], number_of_measurements=1, data_shape=[3, 1]
    )
    primitive.commit_measurement(data=data_1.reshape([3, 1]), indices=(0,))
    primitive.commit_measurement(data=data_2.reshape([3, 1]), indices=(1,))

    stacked_data = np.array([data_1, data_2]).reshape([2, 3, 1])
    result = primitive.result(
        result_id=0,
        raw_data=True)

    assert np.array_equal(
        result,
        stacked_data)


def test_commit_and_result_methods_multiple_measurements(primitive):
    import numpy as np
    data_step_1 = np.array([
        [1, 2, 3],
        [4, 5, 6]]
    )
    data_step_2 = np.array(
        [[1, 2, 3],
         [4, 5, 6]]
    ) + 10

    # The shape should be [sweep_shape, result_ids ,data_shape]
    # For this example, it should be [2, 2, 3]

    primitive.allocate_measurement_buffer(
        sweep_shape=[2, ], number_of_measurements=2, data_shape=[3, 1]
    )
    primitive.commit_measurement(data=data_step_1.reshape([2, 3, 1]), indices=(0,))
    primitive.commit_measurement(data=data_step_2.reshape([2, 3, 1]), indices=(1,))

    stacked_data_result_1 = np.array([data_step_1[0, :], data_step_2[0, :]]).astype(np.complex128).reshape([2, 3, 1])
    stacked_data_result_2 = np.array([data_step_1[1, :], data_step_2[1, :]]).astype(np.complex128).reshape([2, 3, 1])

    result_1 = primitive.result(
        result_id=0,
        raw_data=True)
    result_2 = primitive.result(
        result_id=1,
        raw_data=True)

    assert np.array_equal(result_1, stacked_data_result_1)
    assert np.array_equal(result_2, stacked_data_result_2)


def test_commit_with_transfer_function(primitive):
    import numpy as np

    def transfer_function(data, basis):
        return np.mean(data, axis=0).reshape([-1, 1, 1])

    data_1 = np.array([1, 2, 3]).reshape([3, 1])
    data_2 = np.array([4, 5, 6]).reshape([3, 1])

    # The shape should be [sweep_shape, result_ids ,data_shape]
    # For this example, it should be [2, 1, 3]

    primitive.allocate_measurement_buffer(
        sweep_shape=[2, ], number_of_measurements=1, data_shape=[3, 1]
    )

    primitive.set_transform_function(transfer_function)

    primitive.commit_measurement(data=data_1.reshape([3, 1]), indices=(0,))
    primitive.commit_measurement(data=data_2.reshape([3, 1]), indices=(1,))

    stacked_data = np.array([data_1, data_2])

    assert primitive._transformed_measurement_buffer.shape == (2, 1, 1, 1)

    assert np.array_equal(
        primitive.result(
            result_id=0,
            raw_data=True),
        stacked_data)

    result = primitive.result(
        result_id=0, raw_data=False)
    assert np.array_equal(result,
                          np.asarray([2, 5]).reshape([2, 1, 1]))
