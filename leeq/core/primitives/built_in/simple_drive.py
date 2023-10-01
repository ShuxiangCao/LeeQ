from leeq.core.primitives.logical_primitives import LogicalPrimitive
from leeq.core.primitives.collections import LogicalPrimitiveCollection


class SimpleDrive(LogicalPrimitive):

    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)


class SimpleDriveCollection(LogicalPrimitiveCollection):

    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)
        self._build_primitives()
