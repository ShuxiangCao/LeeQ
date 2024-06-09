from typing import Optional, Dict, Any, Union, List
import numpy as np

from labchronicle import register_browser_function, log_and_record

from leeq import Experiment
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep

from leeq.utils import setup_logging
from leeq.utils.ai.staging.stage_execution import Stage, get_exp_from_var_table, get_codegen_wm, get_code_from_wm

logger = setup_logging(__name__)

__all__ = ["AIInstructionExperiment", "FullyAutomatedExperiment", "AIRun", "AutoRun"]


class AIStagedExperiment(Experiment):
    """
    An experiment that contains multiple stages to be run.
    """

    def run(self, stages: List[Stage], **kwargs):
        """
        Run the staged experiment powered by language model.

        Parameters
        ----------
        stages: List[Stage]
            The stages of the experiment.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        """
        from leeq.utils.ai.variable_table import VariableTable
        from leeq.utils.ai.code_indexer import build_leeq_code_ltm
        from ideanet.codegen.codegen_cog_model import CodegenModel
        from leeq.utils.ai.staging.stage_transition import get_next_stage_label
        from leeq.utils.ai.staging.stage_transition import generate_new_stage_description

        var_table = VariableTable()
        for key, value in kwargs.items():
            var_table.add_variable(key, value)

        self.stages: List[Stage] = stages

        leeq_code_ltm, exps_var_table = build_leeq_code_ltm()
        self.code_cog_model = CodegenModel(leeq_code_ltm)
        self.code_cog_model.n_recall_items = 5  # Number of items to recall in cognitive model
        self.var_table: VariableTable = VariableTable()
        self.var_table.add_parent_table(exps_var_table)
        self.var_table.add_parent_table(var_table)
        self.n_step_multiplier = 6  # Multiplier to control the number of execution steps

        curr_stage = self.stages[0]
        for step in range(len(self.stages) * self.n_step_multiplier):
            new_var_table = self.run_stage_description(curr_stage)
            experiment = get_exp_from_var_table(new_var_table)
            next_stage_info = get_next_stage_label(curr_stage, experiment)
            next_stage_label = next_stage_info["next"]
            additional_info = next_stage_info["additional_info"]
            if next_stage_label == "Complete":
                return
            elif next_stage_label == "Fail":
                return
            next_stage: Stage
            for stage in self.stages:
                if stage.label == next_stage_label:
                    next_stage = stage
                    break
            else:
                assert False, f"Next stage label {next_stage_label} not found in stages"
            new_description = generate_new_stage_description(next_stage, additional_info)
            next_stage.description = new_description
            curr_stage = next_stage

    def run_stage_description(self, stage: Stage):
        """
        Run the stage description powered by language model.

        Parameters
        ----------
        stage: Stage
            The stage to run.
        """
        codegen_wm = get_codegen_wm(stage.description, self.input_var_table)
        recall_res = self.code_cog_model.recall(codegen_wm)
        new_codegen_wm = self.code_cog_model.act(codegen_wm, recall_res)
        new_var_table = self.var_table.new_child_table()
        new_var_table.interpret(get_code_from_wm(new_codegen_wm))
        return new_var_table


class AIInstructionExperiment(AIStagedExperiment):
    """
    An experiment that contains one instruction (step) to be run. The instructions are powered by language model.
    """

    @log_and_record
    def run(self, prompt: str, **kwargs):
        """
        Run the experiment powered by language model.

        Parameters
        ----------
        prompt: str
            The prompt to run the experiment. Contains the experiment design and instructions.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        """
        from leeq.utils.ai.staging.stage_execution import Stage
        stage = Stage("Stage1", "Experiment", prompt, "Go to Complete unless there is an error.")
        stage_complete = Stage("Complete", "Complete", "The experiment is complete.", "End of experiment.")
        stage_fail = Stage("Fail", "Fail", "The experiment has failed.", "End of experiment.")
        stages = [stage, stage_complete, stage_fail]

        super().run(stages, **kwargs)


class FullyAutomatedExperiment(AIStagedExperiment):
    """
    A fully automated experiment that contains multiple steps. Automatically runs the experiment based on the instructions
    provided.
    """

    @log_and_record
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

        from leeq.utils.ai.staging.stage_generation import get_stages_from_description
        stages = get_stages_from_description(prompt)
        super().run(stages, **kwargs)


AIRun = AIInstructionExperiment
AutoRun = FullyAutomatedExperiment
