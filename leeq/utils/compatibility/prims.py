from leeq.core.primitives import *
from leeq.core.primitives.built_in.common import Delay
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep, \
    LogicalPrimitiveBlockParallel as ParallelLPB, LogicalPrimitiveBlockSerial as SerialLPB

SeriesLPB = SerialLPB


class SweepLPB:
    """
    For compatibility reasons
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and (isinstance(args[0], list) or isinstance(args[0], tuple)):
            return LogicalPrimitiveBlockSweep(children=args[0])
        else:
            return LogicalPrimitiveBlockSweep(children=args)
