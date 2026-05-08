def create_two_qubit_gate_collection(*_args, **_kwargs):
    """
    Return a lightweight placeholder collection for tutorial notebooks.

    LeeQ does not currently ship a built-in two-qubit gate collection factory,
    but older tutorials import this helper while demonstrating the intended API.
    """
    return {}


__all__ = ["create_two_qubit_gate_collection"]
