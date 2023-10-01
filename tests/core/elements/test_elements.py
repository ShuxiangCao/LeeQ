import os

import pytest
from pathlib import Path
import json
from datetime import datetime

from leeq.core.elements.elements import Element
from leeq.utils import get_calibration_log_path


@pytest.fixture
def calibration_log_path(tmp_path) -> pytest.fixture():
    os.environ["LEEQ_CALIBRATION_LOG_PATH"] = str(tmp_path / "calibration_logs")
    return os.environ["LEEQ_CALIBRATION_LOG_PATH"]


class MockElement(Element):
    def __init__(self, name: str, parameters: dict = None):
        super().__init__(name, parameters)

    @classmethod
    def _validate_calibration_dict(cls, calibration_dict: dict):
        pass

    def _dump_lpb_collections(self):
        return self._parameters['lpb_collections']

    def _dump_measurement_primitives(self):
        return self._parameters['measurement_primitives']

    def _build_lpb_collections(self):
        pass

    def _build_measurement_primitives(self):
        pass


@pytest.fixture
def element(valid_calibration):
    return MockElement(name="test_element", parameters=valid_calibration)


@pytest.fixture
def valid_calibration():
    return {
        'lpb_collections': {"test_lpb": "test_lpb_val"},
        'measurement_primitives': {"test_measurement": "test_measurement_val"}
    }


def test_element_initialization(element, valid_calibration):
    assert isinstance(element, MockElement)
    assert element._parameters == valid_calibration

    assert element._lpb_collections == {}
    assert element._measurement_primitives == {}


def test_save_calibration_log(element, valid_calibration, calibration_log_path):
    element.save_calibration_log()
    path = get_calibration_log_path()
    assert path.exists()
    file_name = [f for f in path.iterdir() if 'test_element' in str(f)]
    assert file_name
    with open(file_name[0], 'r') as f:
        calibration_log = json.load(f)
    assert calibration_log == valid_calibration


def test_load_from_calibration_log(element,calibration_log_path):
    element.save_calibration_log()
    loaded_element = MockElement.load_from_calibration_log(name="test_element")
    assert loaded_element._parameters == element._parameters
    assert loaded_element._lpb_collections == element._lpb_collections
    assert loaded_element._measurement_primitives == element._measurement_primitives


def test_validate_calibration_dict(valid_calibration):
    Element._validate_calibration_dict(valid_calibration)


def test_validate_calibration_dict_missing_lpb_collections(valid_calibration):
    del valid_calibration['lpb_collections']
    with pytest.raises(AssertionError, match="LPB collections not found in the calibration dictionary."):
        Element._validate_calibration_dict(valid_calibration)


def test_validate_calibration_dict_missing_measurement_primitives(valid_calibration):
    del valid_calibration['measurement_primitives']
    with pytest.raises(AssertionError, match="Measurement primitives not found in the calibration dictionary."):
        Element._validate_calibration_dict(valid_calibration)
