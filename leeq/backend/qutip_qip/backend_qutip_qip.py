from leeq.backend.backend_base import BackendBase
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock


class BackendQutipQIP(BackendBase):
    """
    The BackendQutipQIP class defines a backend for using the qutip_qip package to simulate the experiment at pulse
    level.
    """

    def __init__(self):
        """
        Initialize the BackendQutipQIP class.
        """
        name = 'qutip_qip'
        super().__init__(name)

    def compile_lpb(self, lpb: LogicalPrimitiveBlock):
        """
        Compile the logical primitive block to instructions.
        """
        pass
