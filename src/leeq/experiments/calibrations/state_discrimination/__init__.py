from .assignment import *
from .gaussian_mixture import *
from .windowing_functions import *


class MeasurementOptimization:
    """
    Lightweight compatibility class for notebook measurement optimization examples.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.parameters = kwargs

    def run(self):
        return {
            "status": "configured",
            "parameters": self.parameters,
        }
