from leeq.core.context import ExperimentContext
from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock


class CompilerBase(LeeQObject):

    def compile_lpb(self, context: ExperimentContext, lpb: LogicalPrimitiveBlock):
        """
        Compile the logical primitive block to instructions that going to be passed to the compiler.

        Parameters:
            context (ExperimentContext): The context between setup and compiler.
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.

        Returns:
            Any: The compiled instructions.
        """
        raise NotImplementedError()

    def commit_measurement(self, context: ExperimentContext, lpb: LogicalPrimitiveBlock):
        """
        Commit the measurement primitives to the compiler.

        Parameters:
            context (ExperimentContext): The context between setup and compiler.
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.
        """
        raise NotImplementedError()
