from leeq.core import LeeQObject, LogicalPrimitive


class ExperimentContext(LeeQObject):
    """
    The ExperimentContext class is used to store the context of the experiment between the compiler, setup and engine.
    """

    def __init__(self, name):
        """
        Initialize the ExperimentContext class.
        """
        super().__init__(name)
        self._instructions = None
        self._results = None
        self._step_no = None
        self._lpb = None

    @property
    def instructions(self):
        """
        Get the instructions.
        """
        return self._instructions

    @instructions.setter
    def instructions(self, instructions):
        self._instructions = instructions

    @property
    def results(self):
        """
        Get the results.
        """
        return self._results

    @results.setter
    def results(self, results):
        self._results = results

    def reset(self):
        """
        Reset the context.
        """
        self._instructions = None
        self._results = None
        self._step_no = None

    def set_step_no(self, step_no):
        """
        Set the step number.
        """
        self._step_no = step_no

    @property
    def step_no(self):
        """
        Get the step number.
        """
        return self._step_no

    @property
    def lpb(self):
        """
        Get the logical primitive block.
        """
        return self._lpb

    def set_lpb(self, lpb):
        """
        Set the logical primitive block.
        """
        self._lpb = lpb
