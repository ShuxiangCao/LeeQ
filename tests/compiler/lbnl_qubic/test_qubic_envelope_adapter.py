import inspect

import numpy as np

from leeq.compiler.lbnl_qubic.utils import wrap_envelope_leeq_function_to_qubic_format


def test_adapter_signature_and_sample_units():
    adapter = wrap_envelope_leeq_function_to_qubic_format("square")
    assert "dt" in inspect.signature(adapter).parameters
    time, envelope = adapter(dt=2e-9, width=0.008, phase=0.0, amp=0.2)
    assert len(time) == len(envelope) == 4
    # QubiC applies the amplitude separately; the adapter returns a unit envelope.
    np.testing.assert_allclose(envelope, 1.0)
