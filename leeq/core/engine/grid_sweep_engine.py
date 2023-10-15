from functools import reduce
from typing import Any, List

import numpy as np

from leeq.core.context import ExperimentContext
from leeq.core.engine.measurement_result import MeasurementResult
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock, MeasurementPrimitive
from leeq.experiments.sweeper import Sweeper
from leeq.core.engine.engine_base import EngineBase
import itertools

from leeq.utils import is_running_in_jupyter

_in_jupyter = is_running_in_jupyter()

if _in_jupyter:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class GridSerialSweepEngine(EngineBase):
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

        super().__init__(name=name, compiler=compiler, setup=setup)
        self._context = ExperimentContext(self._name + ".context")
        self._sweep_shape = None

    def run(self, lpb: LogicalPrimitiveBlock, sweep: Sweeper):
        """
        Run the experiment.

        This engine runs the experiment in series. It iterates over all the parameter combinations, for each of them,
        it will call the compiler to compile, upload, fire, commit.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.
            sweep (Sweeper): The sweeper to use.
        """
        self._measurement_results = {}
        self._compiler.clear()

        def _run_single_step(step_no):
            """Run a single step of the experiment. Should be fairly clear :)"""
            self._context.reset()
            self._context.set_lpb(lpb=lpb)
            self._context.set_step_no(step_no)
            self._compile_lpb(lpb=lpb)
            self._update_setup_parameters()
            self._fire_experiment()
            self._collect_data()
            self._commit_measurement(lpb=lpb)
            self._context.reset()

        if sweep is None:
            self._sweep_shape = [1]
            _run_single_step(0)
        else:
            shape = sweep.shape
            self._sweep_shape = shape

            iterator_list = [range(shape[i]) for i in range(len(shape))]

            # Calculate the total size by shape
            total_size = reduce(lambda x, y: x * y, shape)

            with tqdm(total=total_size) as pbar:
                for i, indices in enumerate(itertools.product(*iterator_list)):
                    self._step_no = indices
                    annotations = {'index': str(indices)}
                    # Call the side effect callbacks
                    annotations.update(sweep.execute_side_effects_by_step_no(indices))
                    pbar.set_postfix(annotations)
                    _run_single_step(indices)

                    # Update the progress bar
                    pbar.update(1)
                    self._progress = (i+1) / total_size

    def _commit_measurement(self, lpb):
        """
        Commit the measurement primitives to the compiler.

        First the measurement buffer in the engine will be cleared, then the measurement results will be committed to
        the engine. If the memory buffer is not allocated yet, it will be allocated first with the size infered from
        the first measurement commit and the sweep shape. Then each new commit will be written to the buffer.

        Finally each measurement result will be committed to the measurement primitives.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.
        """

        measurement_results: List[MeasurementResult] = self._context.results

        sweep_shape = self._sweep_shape

        for measurement_result in measurement_results:
            measurement_primitive: MeasurementPrimitive = lpb.nodes[measurement_result.mprim_uuid]

            if not measurement_primitive.is_buffer_allocated():
                # Allocate new buffer
                buffer_shape = list(sweep_shape) + \
                               list(measurement_result.shape)
                assert len(measurement_result.shape) > 1, (
                    f"The shape of the measurement result {measurement_result.shape} should be at least 2D,"
                    f" one dimension for the result id another one for the data."
                )
                measurement_primitive.allocate_measurement_buffer(
                    sweep_shape=sweep_shape,
                    number_of_measurements=measurement_result.shape[0],
                    data_shape=measurement_result.shape[1:],
                    dtype=measurement_result.data.dtype)

            # Write to buffer
            indices = self._context.step_no
            measurement_primitive.commit_measurement(indices=self._context.step_no, data=measurement_result.data)
