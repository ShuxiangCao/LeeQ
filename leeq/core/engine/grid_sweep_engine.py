from functools import reduce
from typing import Any

from leeq.backend.backend_base import BackendBase
from leeq.core import LeeQObject
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
    implement the sweep in series, find those modified lpbs and call the backend to implement the experiment.
    """

    def run(self, lpb: LogicalPrimitiveBlock, sweep: Sweeper):
        """
        Run the experiment.

        This engine runs the experiment in series. It iterates over all the parameter combinations, for each of them,
        it will call the backend to compile, upload, fire, commit.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.
            sweep (Sweeper): The sweeper to use.
        """

        def _run_single_step():
            context = self._compile_lpb(lpb)
            self._update_setup_parameters(context)
            self._fire_experiment(context)
            data = self._collect_data(context)

            # TODO: Commit the data to the measurement primitives

        if sweep is None:
            _run_single_step()
        else:
            shape = sweep.shape

            iterator_list = [range(shape[i]) for i in range(len(shape))]

            # Calculate the total size by shape
            total_size = reduce(lambda x, y: x * y, shape)

            with tqdm(total=total_size) as pbar:
                for i, indices in enumerate(itertools.product(*iterator_list)):
                    # Call the side effect callbacks
                    sweep.execute_side_effects_by_step_no(indices)
                    _run_single_step()
                    # Update the progress bar
                    pbar.update(1)
