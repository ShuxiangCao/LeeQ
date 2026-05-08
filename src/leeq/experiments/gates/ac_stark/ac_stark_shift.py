"""Compatibility wrapper for :mod:`leeq.experiments.gates.ac_stark.stark_shift`."""

from leeq.experiments.gates.ac_stark.stark_shift import *


class ACStarkShiftCalibration:
    """
    Lightweight compatibility class for notebook AC Stark examples.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.parameters = kwargs

    def run(self):
        return {
            "status": "configured",
            "parameters": self.parameters,
        }


__all__ = [name for name in globals() if not name.startswith("_")]
