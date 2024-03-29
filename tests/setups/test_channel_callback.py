import pytest

from leeq.setups.setup_base import SetupStatusParameters


# Test Fixture for providing a common SetupStatusParameters instance for tests
@pytest.fixture
def setup_status():
    return SetupStatusParameters(name="TestSetup")


# Test if channel is added successfully
def test_add_channel(setup_status):
    setup_status.add_channel("ch1", prop="value")
    assert "ch1" in setup_status._channel_dict
    assert setup_status._channel_dict["ch1"] == {"prop": "value"}
    assert "ch1" in setup_status._channel_callbacks
    assert setup_status._channel_callbacks["ch1"] is None


# Test if channel cannot be added twice
def test_add_channel_twice(setup_status):
    setup_status.add_channel("ch1")
    with pytest.raises(ValueError, match=r".*already configured channel ch1.*"):
        setup_status.add_channel("ch1")


# Test if callback can be registered successfully
def test_register_compile_lpb_callback(setup_status):
    setup_status.add_channel("ch1")

    def callback(parameters):
        return parameters

    setup_status.register_compile_lpb_callback("ch1", callback)
    assert setup_status._channel_callbacks["ch1"] == callback


# Test if callback cannot be registered for non-existent channel
def test_register_compile_lpb_callback_non_existent_channel(setup_status):
    with pytest.raises(ValueError, match=r".*does not have channel ch1.*"):
        setup_status.register_compile_lpb_callback("ch1", lambda x: x)


# Test if callback cannot be registered when it's not callable
def test_register_compile_lpb_callback_non_callable(setup_status):
    setup_status.add_channel("ch1")
    with pytest.raises(ValueError, match=r".*Callback is not callable.*"):
        setup_status.register_compile_lpb_callback("ch1", "not_callable")


# Test if get_modified_lpb_parameters_from_channel_callback returns
# modified parameters
def test_get_modified_lpb_parameters_from_channel_callback(setup_status):
    setup_status.add_channel("ch1")

    def callback(parameters):
        parameters["test_key"] = "modified_value"
        return parameters

    setup_status.register_compile_lpb_callback("ch1", callback)

    parameters = {"test_key": "original_value"}
    modified_parameters = setup_status.get_modified_lpb_parameters_from_channel_callback(
        "ch1", parameters)

    assert modified_parameters["test_key"] == "modified_value"


# Test if get_modified_lpb_parameters_from_channel_callback returns
# original parameters when no callback
def test_get_modified_lpb_parameters_no_callback(setup_status):
    setup_status.add_channel("ch1")
    parameters = {"test_key": "original_value"}
    returned_parameters = setup_status.get_modified_lpb_parameters_from_channel_callback(
        "ch1", parameters)

    assert returned_parameters == parameters
