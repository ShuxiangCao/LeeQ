from typing import Any

from leeq.backend.backend_base import BackendBase
from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import Sweeper
from leeq.setups.setup_base import ExperimentalSetup


class EngineBase(LeeQObject):
    """
    The GridSweepEngine class is a class that is used to drive the experiment. It simply takes a lpb and a sweep, and
    implement the sweep in series, find those modified lpbs and call the backend to implement the experiment.
    """

    def __init__(self, name: str, backend: Any, setup: Any):
        """
        Initialize the GridSweepEngine class.

        Parameters:
            name (str): The name of the engine.
            backend (Any): The backend to use.
            setup (Any): The instrument setup to use.
        """

        assert isinstance(backend, BackendBase), "The backend should be a subclass of BackendBase."
        assert isinstance(setup, ExperimentalSetup), "The setup should be a subclass of LeeQObject."

        self._backend = backend
        self._setup = setup

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
        self._setup.update_setup_parameters(instructions)

    def _fire_experiment(self, instructions=None):
        """
        Fire the experiment and wait for it to finish.

        Parameters:
            instructions (Any, Optional): The instructions to be executed.
        """
        self._setup.fire_experiment(instructions)

    def _collect_data(self, context=None):
        """
        Collect the data from the backend and commit it to the measurement primitives.
        """
        return self._setup.collect_data(context)
