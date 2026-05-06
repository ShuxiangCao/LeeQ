import pytest

from leeq.core.elements.elements import CalibrationEncoder, Element


def test_element_default_calibration_is_empty_and_valid():
    element = Element(name="q0")

    assert element.get_calibrations() == {
        "lpb_collections": {},
        "measurement_primitives": {},
    }


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"measurement_primitives": {}}, "LPB collections not found"),
        ({"lpb_collections": {}}, "Measurement primitives not found"),
    ],
)
def test_element_requires_calibration_sections(parameters, message):
    with pytest.raises(ValueError, match=message):
        Element(name="invalid", parameters=parameters)


def test_dump_dict_filters_private_nested_keys():
    element = Element(name="q0")

    dumped = element._dump_dict(
        {
            "visible": 1,
            "_private": "hidden",
            "nested": {
                "kept": 2,
                "_dropped": 3,
            },
        }
    )

    assert dumped == {"visible": 1, "nested": {"kept": 2}}


def test_missing_measurement_primitive_reports_requested_name():
    element = Element(name="q0")

    with pytest.raises(KeyError, match="Measurement primitive 0 not found"):
        element.get_measurement_primitive(0)


def test_calibration_encoder_serializes_callables_by_repr():
    encoded = CalibrationEncoder().encode({"callback": len})

    assert "callback" in encoded
    assert "len" in encoded
