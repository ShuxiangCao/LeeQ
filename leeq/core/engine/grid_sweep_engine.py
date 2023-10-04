from functools import reduce
from typing import Any

from leeq.core.context import ExperimentContext
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import Sweeper
from leeq.core.engine.engine_base import EngineBase
import itertools

from leeq.utils import is_running_in_jupyter

_in_jupyter = is_running_in_jupyter()

if _in_jupyter:
    from tqdm import notebook as tqdm
else:
    from tqdm import tqdm


class GridSerialSweepEngine(EngineBase):
    """
    The GridSweepEngine class is a class that is used to drive the experiment. It simply takes a lpb and a sweep, and
    implement the sweep in series, find those modified lpbs and call the compiler to implement the experiment.
    """

    def __init__(self, name: str, backend: Any, setup: Any):
        """
        Initialize the GridSweepEngine class.

        Parameters:
            name (str): The name of the engine.
            backend (Any): The compiler to use.
            setup (Any): The instrument setup to use.
        """

        super().__init__(name=name, backend=backend, setup=setup)
        self._context = ExperimentContext(self._name + '.context')

    def run(self, lpb: LogicalPrimitiveBlock, sweep: Sweeper):
        """
        Run the experiment.

        This engine runs the experiment in series. It iterates over all the parameter combinations, for each of them,
        it will call the compiler to compile, upload, fire, commit.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.
            sweep (Sweeper): The sweeper to use.
        """

        def _run_single_step(step_no):
            """ Run a single step of the experiment. Should be fairly clear :)"""
            self._context.reset()
            self._context.set_step_no(step_no)
            self._compile_lpb(lpb=lpb)
            self._update_setup_parameters()
            self._fire_experiment()
            self._collect_data()
            self._commit_measurement()

        if sweep is None:
            _run_single_step(0)
        else:
            shape = sweep.shape

            iterator_list = [range(shape[i]) for i in range(len(shape))]

            # Calculate the total size by shape
            total_size = reduce(lambda x, y: x * y, shape)

            with tqdm(total=total_size) as pbar:
                for i, indices in enumerate(itertools.product(*iterator_list)):
                    # Call the side effect callbacks
                    sweep.execute_side_effects_by_step_no(indices)
                    _run_single_step(indices)

                    # Update the progress bar
                    pbar.update(1)

    def _compile_lpb(self, lpb: LogicalPrimitiveBlock):
        """
        Compile the logical primitive block to instructions that going to be passed to the compiler.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.

        Returns:
            Any: The compiled instructions.
        """
        return self._backend.compile_lpb(self._context, lpb)

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

    def _commit_measurement(self):
        """
        Commit the measurement primitives to the compiler.
        """
        self._backend.commit_measurement(self._context)
