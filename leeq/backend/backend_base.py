from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import Sweeper


class BackendBase(LeeQObject):


    def compile_lpb(self, lpb: LogicalPrimitiveBlock):
        """
        Compile the logical primitive block to instructions that going to be passed to the backend.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.

        Returns:
            Any: The compiled instructions.
        """
        raise NotImplementedError()

