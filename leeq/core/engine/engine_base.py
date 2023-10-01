from typing import Any

from leeq.backend.backend_base import BackendBase
from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import Sweeper


class EngineBase(LeeQObject):
    """
    The GridSweepEngine class is a class that is used to drive the experiment. It simply takes a lpb and a sweep, and
    implement the sweep in series, find those modified lpbs and call the backend to implement the experiment.
    """

    def __init__(self, name: str, backend: Any):
        """
        Initialize the GridSweepEngine class.

        Parameters:
            name (str): The name of the engine.
            backend (Any): The backend to use.
        """

        assert isinstance(backend, BackendBase), "The backend should be a subclass of BackendBase."

        self._backend = backend

        super().__init__(name)

    def run(self, lpb: LogicalPrimitiveBlock, sweep: Sweeper):
        """
        Run the experiment.

        The experiment run iterates all the parameters described by the sweeper. Each iteration can be break into
         four steps:

        1. Compile the measurement lpb to instructions that going to be passed to the backend.
        2. Upload the instruction to the backend, including changing frequencies of the generators etc. Get everything
            ready for the experiment.
        3. Fire the experiment and wait for it to finish.
        4. Collect the data from the backend and commit it to the measurement primitives.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.
            sweep (Sweeper): The sweeper to use.
        """

        raise NotImplementedError()

    def _compile_lpb(self, lpb: LogicalPrimitiveBlock):
        """
        Compile the logical primitive block to instructions that going to be passed to the backend.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.

        Returns:
            Any: The compiled instructions.
        """
        return self._backend.compile_lpb(lpb)

    def _update_setup_parameters(self, instructions):
        """
        Update the setup parameters of the backend.

        Parameters:
            instructions (Any): The instructions to be executed.
        """
        self._backend.update_setup_parameters(instructions)

    def _fire_experiment(self, instructions=None):
        """
        Fire the experiment and wait for it to finish.

        Parameters:
            instructions (Any, Optional): The instructions to be executed.
        """
        self._backend.fire_experiment(instructions)

    def _collect_data(self):
        """
        Collect the data from the backend and commit it to the measurement primitives.
        """
        return self._backend.collect_data()
