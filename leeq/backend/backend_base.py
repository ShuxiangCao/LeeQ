from leeq.core import LeeQObject
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import Sweeper


class BackendBase(LeeQObject):
    pass

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

        So the backend should implement the following methods:
        1. `compile_lpb`: Compile the logical primitive block to instructions that going to be passed to the backend.
        2. `update_setup_parameters`: Update the setup parameters of the backend. Usually this function calculates
            the frequencies of the generators etc and pass it to specific experiment setup class to further upload.
        3. `fire_experiment`: Fire the experiment and wait for it to finish.
        4. `collect_data`: Collect the data from the backend and commit it to the measurement primitives.

        Note that the collected data will be committed to the measurement primitives by the engine, so the backend
            should not commit the data to the measurement primitives.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.
            sweep (Sweeper): The sweeper to use.
        """
        pass

    def compile_lpb(self, lpb: LogicalPrimitiveBlock):
        """
        Compile the logical primitive block to instructions that going to be passed to the backend.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.

        Returns:
            Any: The compiled instructions.
        """
        raise NotImplementedError()

    def update_setup_parameters(self, instructions):
        """
        Update the setup parameters of the backend.

        Parameters:
            instructions (Any): The instructions to be executed.
        """
        raise NotImplementedError()

    def fire_experiment(self, instructions):
        """
        Fire the experiment and wait for it to finish.

        Parameters:
            instructions (Any): The instructions to be executed.
        """
        raise NotImplementedError()

    def collect_data(self, instructions):
        """
        Collect the data from the backend and commit it to the measurement primitives.

        Parameters:
            instructions (Any): The instructions to be executed.
        """
        raise NotImplementedError()
