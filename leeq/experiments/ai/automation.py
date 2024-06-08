from typing import Optional, Dict, Any, Union
import numpy as np

from labchronicle import register_browser_function, log_and_record
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep

from leeq.utils import setup_logging

logger = setup_logging(__name__)

__all__ = ["AIInstructionExperiment", "FullyAutomatedExperiment", "AIRun", "AutoRun"]


class AIInstructionExperiment(Experiment):
    """
    An experiment that contains one instruction (step) to be run. The instructions are powered by language model.
    """

    def run(self, prompt: str, var_table: Dict[str, Any], **kwargs):
        """
        Run the experiment powered by language model.

        Parameters
        ----------
        prompt: str
            The prompt to run the experiment. Contains the experiment design and instructions.
        var_table: Dict[str, Any]
            The variable table containing the variables to be used in the experiment.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        """
        pass


class FullyAutomatedExperiment(Experiment):
    """
    A fully automated experiment that contains multiple steps. Automatically runs the experiment based on the instructions
    provided.
    """

    def run(self, prompt: str, var_table: Dict[str, Any], **kwargs):
        """
        Run the automated experiment powered by language model.

        Parameters
        ----------
        prompt: str
            The prompt to run the experiment. Contains the experiment design and instructions.
        var_table: Dict[str, Any]
            The variable table containing the variables to be used in the experiment.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        """
        pass


AIRun = AIInstructionExperiment
AutoRun = FullyAutomatedExperiment
