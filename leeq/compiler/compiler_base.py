from leeq.core.context import ExperimentContext
from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock


class MeasurementSequence:
    """
    The MeasurementSequence class is used to annotate when to start the aquisition of the signal.
    """

    def __init__(self):
        """
        Initialize the MeasurementSequence class.
        """
        self._measurements = []

    def add_measurement(self, position, channel, tags):
        """
        Add a measurement to the measurement sequence.
        """
        self._measurements.append((position, channel, tags))

    def get_measurements(self):
        """
        Get the measurements.
        """
        return self._measurements


class LPBCompiler(LeeQObject):
    """
    The CompilerBase class defines a compiler that is used to compile the logical primitive block to instructions that
    going to be passed to the compiler.
    """

    def commit_measurement(
        self, context: ExperimentContext, lpb: LogicalPrimitiveBlock
    ):
        """
        Commit the measurement result to the measurement primitives.
        """

        measurement_keys = [k for k in context.results.keys()]
        measurement_keys.sort(key=lambda x: x[1])

        for i, (uuid, position_point) in enumerate(measurement_keys):
            measurement_primitive = lpb.nodes[uuid]
            measurement_primitive.commit_result(
                context.results[(uuid, position_point)])

        return context

    def clear(self):
        """
        Clear the compiler.
        """
        pass


# Dummy compiler simply returns the LPB as the compiled instructions
DummyCompiler = LPBCompiler
