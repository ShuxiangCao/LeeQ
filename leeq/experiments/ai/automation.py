from typing import Optional, Dict, Any, Union, List
import numpy as np

from labchronicle import register_browser_function, log_and_record

from ..experiments import Experiment
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep

from leeq.utils import setup_logging
from leeq.utils.notebook import show_spinner, hide_spinner

from ideanet.core.logger import IdeaBaseLogger

logger = setup_logging(__name__)

__all__ = ["AIInstructionExperiment", "FullyAutomatedExperiment", "AIRun", "AutoRun"]


class AIStagedExperiment(Experiment):
    """
    An experiment that contains multiple stages to be run.
    """

    def run(self, stages: List['Stage'], **kwargs):
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
        from leeq.utils.ai.staging.stage_execution import Stage, get_exp_from_var_table, get_codegen_wm, \
            get_code_from_wm
        from leeq.utils.ai.staging.stage_generation import stages_to_html
        from leeq.utils.ai.display_chat.notebooks import display_chat, code_to_html, dict_to_html

        stages_html = stages_to_html(stages)
        display_chat("Stage planning AI", '#fff0f5', "The planned experiments are:<br>" + stages_html)

        input_var_table = VariableTable()
        for key, value in kwargs.items():
            input_var_table.add_variable(key, value)

        self.stages: List[Stage] = stages

        leeq_code_ltm, exps_var_table = build_leeq_code_ltm()
        code_cog_model = CodegenModel(leeq_code_ltm)
        code_cog_model.n_recall_items = 5  # Number of items to recall in cognitive model
        var_table: VariableTable = VariableTable()
        var_table.add_parent_table(exps_var_table)
        var_table.add_parent_table(input_var_table)
        self.n_step_multiplier = 6  # Multiplier to control the number of execution steps

        def run_stage_description(stage: 'Stage'):
            """
            Run the stage description powered by language model.

            Parameters
            ----------
            stage: Stage
                The stage to run.
            """
            spinner_id = show_spinner(f"Generating code for {stage.label}")

            prompt = f"""
            Overview of the funcationality: {stage.overview}
            Current stage: {stage.label}
            """


            codegen_wm = get_codegen_wm(stage.description, input_var_table, hint=prompt)
            recall_res = code_cog_model.recall(codegen_wm)
            new_codegen_wm = code_cog_model.act(codegen_wm, recall_res)
            new_var_table = var_table.new_child_table()
            with IdeaBaseLogger():
                codes = get_code_from_wm(new_codegen_wm)

            hide_spinner(spinner_id)
            code_html = code_to_html(codes)
            display_chat("Code generation AI", '#fff0f5', f"Generated code are as follows:<br>{code_html}")
            new_var_table.interpret(codes)
            return new_var_table

        curr_stage = self.stages[0]
        for step in range(len(self.stages) * self.n_step_multiplier):
            new_var_table = run_stage_description(curr_stage)

            exp_object = get_exp_from_var_table(new_var_table)
            experiment_result = exp_object.get_ai_inspection_results()

            experiment_analysis_html = dict_to_html(experiment_result)

            display_chat("Inspection AI",
                         '#f0f8ff', f"Experiment analysis results are as follows:<br>{experiment_analysis_html}")

            spinner_id = show_spinner(f"Considering the next stage...")
            next_stage_info = get_next_stage_label(curr_stage, experiment_result)
            next_stage_label = next_stage_info["next"]
            additional_info = next_stage_info["additional_info"]
            if next_stage_label == "Complete":

                display_chat("Stage Planning AI", '#fff0f5',
                             f"Experiment complete.<br>"
                             f"{next_stage_info['analysis']}")
                hide_spinner(spinner_id)
                return
            elif next_stage_label == "Fail":
                display_chat("Stage Planning AI", '#fff0f5',
                             f"Experiment failed.<br>"
                             f"{next_stage_info['analysis']}")
                hide_spinner(spinner_id)
                return
            next_stage: Stage
            for stage in self.stages:
                if stage.label == next_stage_label:
                    next_stage = stage
                    break
            else:
                assert False, f"Next stage label {next_stage_label} not found in stages"

            new_description = generate_new_stage_description(next_stage, additional_info)
            hide_spinner(spinner_id)
            display_chat("Stage Planning AI", '#fff0f5', f"Transitioning to the next"
                                                         f" stage {next_stage.label} with the following description:<br>{new_description}<br>"
                                                         f"{next_stage_info['analysis']}")
            next_stage.description = new_description
            curr_stage = next_stage


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
        stage = Stage("Stage1", "Experiment", prompt, "Go to Complete if success, otherwise go to Fail.")
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
    def run(self, prompt: str, **kwargs):
        """
        Run the automated experiment powered by language model.

        Parameters
        ----------
        prompt: str
            The prompt to run the experiment. Contains the experiment design and instructions.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        """

        from leeq.utils.ai.staging.stage_generation import get_stages_from_description
        spinner_id = show_spinner("AI is designing the experiment...")
        stages = get_stages_from_description(prompt)
        hide_spinner(spinner_id)
        super().run(stages, **kwargs)


AIRun = AIInstructionExperiment
AutoRun = FullyAutomatedExperiment
