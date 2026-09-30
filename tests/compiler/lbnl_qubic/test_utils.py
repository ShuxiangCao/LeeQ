import inspect

import numpy as np

from leeq.compiler.lbnl_qubic.utils import (
    wrap_envelope_leeq_function_to_qubic_format,
)


def test_qubic_adapter_exposes_dt_interface_and_evaluates_square():
    adapter = wrap_envelope_leeq_function_to_qubic_format("square")

    assert "dt" in inspect.signature(adapter).parameters
    time, envelope = adapter(dt=2e-9, width=0.008, phase=0.0)
    assert len(time) == len(envelope)
    np.testing.assert_allclose(envelope, 1.0)
