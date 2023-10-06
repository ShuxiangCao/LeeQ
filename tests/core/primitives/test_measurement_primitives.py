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
    data = np.array([1, 2, 3])
    primitive.commit_measurement(data=data)
    assert np.array_equal(primitive.result(result_id=0, raw_data=True), data)


def test_clear_results_method(primitive):
    import numpy as np
    data = np.array([1, 2, 3])
    primitive.commit_measurement(data=data)
    primitive.clear_results()
    with pytest.raises(RuntimeError):
        primitive.result(result_id=0, raw_data=True)
