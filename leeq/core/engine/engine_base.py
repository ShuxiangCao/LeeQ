from typing import Any

from leeq.compiler.compiler_base import LPBCompiler
from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import Sweeper
from leeq.setups.setup_base import ExperimentalSetup


class EngineBase(LeeQObject):
    """
    The GridSweepEngine class is a class that is used to drive the experiment. It simply takes a lpb and a sweep, and
    implement the sweep in series, find those modified lpbs and call the compiler to implement the experiment.
    """

    def __init__(self, name: str, compiler: Any, setup: Any):
        """
        Initialize the GridSweepEngine class.

        Parameters:
            name (str): The name of the engine.
            compiler (Any): The compiler to use.
            setup (Any): The instrument setup to use.
        """

        assert isinstance(
            compiler, LPBCompiler
        ), "The compiler should be a subclass of LPBCompiler."
        assert isinstance(
            setup, ExperimentalSetup
        ), "The setup should be a subclass of LeeQObject."

        self._compiler = compiler
        self._setup = setup

        super().__init__(name)

    def get_live_status(self):
        """
        Get the live status of the engine.

        Returns:
            dict: The live status of the engine.
        """

        if self._context is None:
            return None

        return {
            'step_no': self._context.step_no,
        }

    def run(self, lpb: LogicalPrimitiveBlock, sweep: Sweeper):
        """
        Run the experiment.

        The experiment run iterates all the parameters described by the sweeper. Each iteration can be break into
         four steps:

        1. Compile the measurement lpb to instructions that going to be passed to the compiler.
        2. Upload the instruction to the compiler, including changing frequencies of the generators etc. Get everything
            ready for the experiment.
        3. Fire the experiment and wait for it to finish.
        4. Collect the data from the compiler and commit it to the measurement primitives.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.
            sweep (Sweeper): The sweeper to use.
        """

        raise NotImplementedError()

    def _compile_lpb(self, lpb: LogicalPrimitiveBlock):
        """
        Compile the logical primitive block to instructions that going to be passed to the compiler.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.

        Returns:
            Any: The compiled instructions.
        """

        if self._compiler is None:
            # No compling, directly use the lpb
            self._context.instructions = lpb

        return self._compiler.compile_lpb(self._context, lpb)

    def _update_setup_parameters(self):
        """
        Update the setup parameters of the compiler.
        """
        self._setup.update_setup_parameters(self._context)

    def _fire_experiment(self):
        """
        Fire the experiment and wait for it to finish.
        """
        self._setup.fire_experiment(self._context)

    def _collect_data(self):
        """
        Collect the data from the compiler and commit it to the measurement primitives.
        """
        return self._setup.collect_data(self._context)
