# This file provides the compatibility layer for the old quantum control
# system code.
from leeq import Experiment as experiment, basic_run as basic, setup
from leeq.experiments.sweeper import (
    Sweeper as sweeper,
    SweepParametersSideEffectFactory as sparam,
)
import leeq.utils.compatibility.prims as prims
