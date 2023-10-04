import pytest
import inspect

from leeq.compiler.utils.pulse_shape_utils import PulseShapeFactory


class MockPulseShapeFactory(PulseShapeFactory):

    def __init__(self):
        super().__init__()

    @classmethod
    def kill_singleton(cls):
        cls._instance = None


def test_singleton_behavior():
    # Given/When
    factory_1 = MockPulseShapeFactory()
    factory_2 = MockPulseShapeFactory()

    # Then
    assert factory_1 is factory_2, "Singleton instances should be the same object"

    factory_1.kill_singleton()


def sample_pulse_shape(sampling_rate):
    return None


def test_register_pulse_shape():
    # Given
    factory = MockPulseShapeFactory()
    pulse_shape_name = "sample_shape"
    pulse_shape_function = sample_pulse_shape

    # When
    factory.register_pulse_shape(pulse_shape_name, pulse_shape_function)

    # Then
    assert factory._pulse_shape_functions[
               pulse_shape_name] == pulse_shape_function, "Pulse shape should be registered correctly"

    factory.kill_singleton()


def test_register_pulse_shape_function_not_callable():
    # Given
    factory = MockPulseShapeFactory()
    pulse_shape_name = "invalid_shape"
    pulse_shape_function = "Not a function"

    # When/Then
    with pytest.raises(RuntimeError, match="The pulse shape function must be callable."):
        factory.register_pulse_shape(pulse_shape_name, pulse_shape_function)

    factory.kill_singleton()


def test_register_pulse_shape_without_sampling_rate_parameter():
    # Given
    factory = MockPulseShapeFactory()
    pulse_shape_name = "incompatible_shape"

    def invalid_pulse_shape():  # No parameters
        pass

    # When/Then
    with pytest.raises(RuntimeError,
                       match="The pulse shape function invalid_pulse_shape is not compatible. It must accept 'sampling_rate' parameter."):
        factory.register_pulse_shape(pulse_shape_name, invalid_pulse_shape)

    factory.kill_singleton()


def test_register_already_registered_pulse_shape(caplog):
    # Given
    factory = MockPulseShapeFactory()
    pulse_shape_name = "sample_shape"
    pulse_shape_function = sample_pulse_shape

    # When
    factory.register_pulse_shape(pulse_shape_name, pulse_shape_function)  # Register once
    factory.register_pulse_shape(pulse_shape_name, pulse_shape_function)  # Register twice

    # Then
    assert "The pulse shape name sample_shape has already been registered." in caplog.text, "Warning should be logged"

    factory.kill_singleton()


from unittest.mock import patch, MagicMock


def mock_pulse_shape_function(sampling_rate):
    pass


@pytest.fixture
def mock_pulse_shapes_module():
    with patch('leeq.compiler.utils.pulse_shapes',
               autospec=True) as mock_module:  # Adjust your_module_path accordingly
        mock_module.mock_pulse_shape = mock_pulse_shape_function
        yield mock_module


def test_load_built_in_pulse_shapes(mock_pulse_shapes_module):
    # Given
    factory = MockPulseShapeFactory()

    # Mock is_valid_pulse_shape_function to always return True
    factory.is_valid_pulse_shape_function = MagicMock(return_value=True)
    factory.register_pulse_shape = MagicMock()

    # Mock inspect.getmembers to return our mocked pulse shape function
    with patch.object(inspect, 'getmembers',
                      return_value=[('mock_pulse_shape', mock_pulse_shapes_module.mock_pulse_shape)]):
        factory._load_built_in_pulse_shapes()  # If _load_built_in_pulse_shapes is protected

        # Then
        factory.register_pulse_shape.assert_called_once_with('mock_pulse_shape',
                                                             mock_pulse_shapes_module.mock_pulse_shape)

    factory.kill_singleton()


def test_integrate_load_builtin_pulse_shapes():
    pulse_shape_built_in_list = ['blackman', 'blackman_drag', 'clear_square', 'gaussian', 'gaussian_drag',
                                 'soft_square', 'square']

    factory = MockPulseShapeFactory()
    pulse_shapes = factory.get_available_pulse_shapes()
    assert len(pulse_shapes) > 0, "There should be some built-in pulse shapes"

    # Make sure all the listed shapes are in the available pulse shapes
    for pulse_shape in pulse_shape_built_in_list:
        assert pulse_shape in pulse_shapes, f"{pulse_shape} should be in the available as a built-in pulse shape"

    assert 'get_t_list' not in pulse_shapes, "get_t_list should not be in the available pulse shapes"

    factory.kill_singleton()
