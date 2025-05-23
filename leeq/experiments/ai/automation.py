from typing import Optional, Dict, Any, Union, List
import numpy as np

from labchronicle import register_browser_function, log_and_record

from leeq.utils import setup_logging
from leeq.utils.notebook import show_spinner, hide_spinner
from leeq.utils.ai.staging.stage_execution import CodegenModel
from leeq.utils.ai.staging.stage_generation import find_the_stage_label_based_on_description

from IPython.display import display, HTML
from leeq.experiments import Experiment

logger = setup_logging(__name__)

__all__ = ["AIInstructionExperiment", "FullyAutomatedExperiment", "AIRun", "AutoRun"]


def execute_experiment_from_prompt(prompt: str, **kwargs):
    """
    Execute an experiment from a prompt.

    Parameters
    ----------
    prompt: str
        The prompt to run the experiment.
    kwargs
        Additional keyword arguments.

    Returns
    -------
    The variable table after the experiment is run.

    """
    from leeq.utils.ai.variable_table import VariableTable
    from leeq.utils.ai.code_indexer import build_leeq_code_ltm
    from leeq.utils.ai.staging.stage_transition import get_next_stage_label
    from leeq.utils.ai.staging.stage_transition import generate_new_stage_description
    from leeq.utils.ai.staging.stage_execution import Stage, get_exp_from_var_table, get_codegen_wm, \
        get_code_from_wm, check_if_needed_to_break_down
    from leeq.utils.ai.staging.stage_generation import stages_to_html
    from leeq.utils.ai.display_chat.notebooks import display_chat, code_to_html, dict_to_html

    spinner_id = show_spinner(f"Interpreting experiment...")
    input_var_table = VariableTable()
    for key, value in kwargs.items():
        input_var_table.add_variable(key, value)

    leeq_code_ltm, exps_var_table = build_leeq_code_ltm()
    code_cog_model = CodegenModel()
    for idea in leeq_code_ltm.ideas:
        code_cog_model.lt_memory.add_idea(idea)
    code_cog_model.n_recall_items = 5  # Number of items to recall in cognitive model
    var_table: VariableTable = VariableTable()
    var_table.add_parent_table(exps_var_table)
    var_table.add_parent_table(input_var_table)

    codegen_wm = get_codegen_wm(prompt, input_var_table)

    recall_res = code_cog_model.recall(codegen_wm)
    codes = code_cog_model.codegen(codegen_wm, recall_res)

    new_var_table = var_table.new_child_table()

    hide_spinner(spinner_id)
    code_html = code_to_html(codes)
    display_chat("Code generation AI", 'light_purple', f"Here is the generated code:<br>{code_html}")
    new_var_table.interpret(codes)
    return new_var_table


class AIStagedExperiment(Experiment):
    """
    An experiment that contains multiple stages to be run.
    """

    def run(self, stages: List['Stage'], sub_experiment=False, **kwargs):
        """
        Run the staged experiment powered by language model.

        Parameters
        ----------
        stages: List[Stage]
            The stages of the experiment.
        sub_experiment: bool
            Whether the experiment is a sub-experiment. If it is we do not allow it to be further splitted.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        """
        from leeq.utils.ai.variable_table import VariableTable
        from leeq.utils.ai.code_indexer import build_leeq_code_ltm
        from leeq.utils.ai.staging.stage_transition import get_next_stage_label
        from leeq.utils.ai.staging.stage_transition import generate_new_stage_description
        from leeq.utils.ai.staging.stage_execution import Stage, get_exp_from_var_table, get_codegen_wm, \
            get_code_from_wm, check_if_needed_to_break_down
        from leeq.utils.ai.staging.stage_generation import stages_to_html
        from leeq.utils.ai.display_chat.notebooks import display_chat, code_to_html, dict_to_html

        input_var_table = VariableTable()
        for key, value in kwargs.items():
            input_var_table.add_variable(key, value)
        # input_var_table.add_variable("np", np)

        self.stages: List[Stage] = stages

        leeq_code_ltm, exps_var_table = build_leeq_code_ltm()
        code_cog_model = CodegenModel()
        for idea in leeq_code_ltm.ideas:
            code_cog_model.lt_memory.add_idea(idea)
        code_cog_model.n_recall_items = 5  # Number of items to recall in cognitive model
        var_table: VariableTable = VariableTable()

        moduler_var_table = VariableTable()
        moduler_var_table.add_variable("np", np)
        moduler_var_table.add_variable("numpy", np)
        var_table.add_parent_table(moduler_var_table)

        var_table.add_parent_table(exps_var_table)
        var_table.add_parent_table(input_var_table)
        self.n_step_multiplier = 6  # Multiplier to control the number of execution steps

        coding_ltm_cache = {}

        self.experiment_history = []

        def run_stage_description(stage: 'Stage'):
            """
            Run the stage description powered by language model.

            Parameters
            ----------
            stage: Stage
                The stage to run.
            """
            spinner_id = show_spinner(f"Executing {stage.label}: {stage.title}...")

            prompt = f"""
            Overview of the funcationality: {stage.overview}
            Current stage: {stage.label}
            """

            html = stages_to_html([stage])
            display(HTML(html))

            if sub_experiment:
                single_step = True
            else:
                breakdown_requirement = check_if_needed_to_break_down(stage.description)
                single_step = breakdown_requirement['single_step'] or len(breakdown_requirement['steps']) == 1

            if not single_step:
                hide_spinner(spinner_id)
                display_chat("Stage Planning AI", 'light_blue',
                             f"Stage {stage.label} is too complex to be processed in one step. Planning to break down the stage into smaller steps. {breakdown_requirement['reason']}.")
                exp = FullyAutomatedExperiment(stage.description, sub_experiment=True, **input_var_table.variable_objs)
                new_var_table = var_table.new_child_table()
                new_var_table.add_variable("exp", exp)

                return new_var_table

            codegen_wm = get_codegen_wm(stage.description, input_var_table)

            if stage.title not in coding_ltm_cache:
                recall_res = code_cog_model.recall(codegen_wm)
                coding_ltm_cache[stage.title] = recall_res
            else:
                recall_res = coding_ltm_cache[stage.title]

            # with display_chats():
            codes = code_cog_model.codegen(codegen_wm, recall_res)

            new_var_table = var_table.new_child_table()

            hide_spinner(spinner_id)
            code_html = code_to_html(codes)
            display_chat("Code generation AI", 'light_purple', f"Here is the generated code:<br>{code_html}")
            new_var_table.interpret(codes)
            return new_var_table

        curr_stage = self.stages[0]
        for step in range(len(self.stages) * self.n_step_multiplier):
            numbers_of_retry = 0

            while numbers_of_retry < 3:
                try:
                    new_var_table = run_stage_description(curr_stage)
                    exp_object = get_exp_from_var_table(new_var_table)
                    if exp_object is None:
                        logger.warning(f"Experiment object not found in the variable table.")
                        continue
                    self.experiment_history.append(exp_object)
                    experiment_result = exp_object.get_ai_inspection_results()
                    break
                except Exception as e:
                    raise e
                    numbers_of_retry += 1
                    if numbers_of_retry == 3:
                        raise e

            experiment_analysis_html = dict_to_html(experiment_result)

            color = 'light_green' if experiment_result['Experiment success'] else 'light_red'
            display_chat("Inspection AI",
                         color, f"Experiment analysis results are as follows:<br>{experiment_analysis_html}")

            spinner_id = show_spinner(f"Considering the next stage...")
            next_stage_info = get_next_stage_label(curr_stage, experiment_result)
            next_stage_label = next_stage_info["next"]
            additional_info = next_stage_info["additional_info"]
            if next_stage_label == "Complete":

                display_chat("Stage Planning AI", 'light_green',
                             f"Experiment complete.<br>"
                             f"{next_stage_info['analysis']}")
                hide_spinner(spinner_id)
                break
            elif next_stage_label == "Fail":
                display_chat("Stage Planning AI", 'light_red',
                             f"Experiment failed.<br>"
                             f"{next_stage_info['analysis']}")
                hide_spinner(spinner_id)
                break
            next_stage: Stage
            for stage in self.stages:
                if stage.label in next_stage_label:
                    next_stage = stage
                    break
            else:
                next_stage = find_the_stage_label_based_on_description(self.stages, next_stage_label)
                if next_stage is None:
                    assert False, f"Next stage label {next_stage_label} not found in stages"

            if curr_stage.label in next_stage.label:
                new_description = generate_new_stage_description(next_stage, additional_info)
                next_stage.description = new_description

            hide_spinner(spinner_id)
            display_chat("Stage Planning AI", 'light_blue', f"Transitioning to the next"
                                                            f" stage {next_stage.label} with the following description:<br>{next_stage.description}<br>"
                                                            f"{next_stage_info['analysis']}")
            curr_stage = next_stage

        self.final_result = {
            "success": next_stage_label == "Complete",
            "analysis": self.experiment_history[-1].get_ai_inspection_results()
        }

    def get_experiment_history(self) -> List[Experiment]:
        """
        Get the history of experiments run in the staged experiment.

        Returns
        -------
        List[Experiment]: The history of experiments run in the staged experiment.
        """
        return self.experiment_history

    def get_last_experiment(self) -> Experiment:
        """
        Get the last experiment run in the staged experiment.

        Returns
        -------
        Experiment: The last experimen't run in the staged experiment.
        """
        return self.experiment_history[-1]

    def get_ai_inspection_results(self) -> Dict[str, Any]:
        return self.get_last_experiment().get_ai_inspection_results()

        # return {
        #    "Analysis": self.final_result["analysis"],
        #    "Suggested parameter updates": None,
        #    'Experiment success': self.final_result["success"],
        # }


class AIInstructionExperiment(AIStagedExperiment):
    """
    An experiment that contains one instruction (step) to be run. The instructions are powered by language model.
    """

    @log_and_record(overwrite_func_name='AIInstructionExperimen.run')
    def run_simulated(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @log_and_record
    def run(self, prompt: str, next_stage_guide=None, **kwargs):
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
        # label: str, title: str, overview: str, description: str, next_stage_guide: str

        if next_stage_guide is None:
            next_stage_guide = """Go to Complete if success. 
                                Go back to the same stage if the experiment failed and the parameters should be adjusted.
                                Go to Fail if the experiment failed and the parameters cannot be adjusted.
                                Go to Fail if the experiment failed and there is no suggestion for how to adjust the parameters.
                                Follow the instructions on how to transit to the next stage from the report of the experiment if there is any.
                                Go to Fail if the experiment has failed after 3 attempts."""

        stage = Stage(label="Stage1", title="Implement experiment",
                      overview='You are requested to implement one experiment and modify the parameter to make it success.',
                      description=prompt, next_stage_guide=next_stage_guide
                      )
        stage_complete = Stage("Complete", "Complete", "The experiment is complete.", "End of experiment.",
                               next_stage_guide='None')
        stage_fail = Stage("Fail", "Fail", "The experiment has failed.", "End of experiment.", next_stage_guide='None')
        stages = [stage, stage_complete, stage_fail]

        super().run(stages, **kwargs)


class FullyAutomatedExperiment(AIStagedExperiment):
    """
    A fully automated experiment that contains multiple steps. Automatically runs the experiment based on the instructions
    provided.
    """

    @log_and_record(overwrite_func_name='FullyAutomatedExperiment.run')
    def run_simulated(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @log_and_record
    def run(self, prompt: str, sub_experiment=False, **kwargs):
        """
        Run the automated experiment powered by language model.

        Parameters
        ----------
        prompt: str
            The prompt to run the experiment. Contains the experiment design and instructions.
        sub_experiment: bool
            Whether the experiment is a sub-experiment. If it is we do not allow it to be further splitted.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        """

        from leeq.utils.ai.staging.stage_generation import get_stages_from_description
        from leeq.utils.ai.staging.stage_generation import stages_to_html
        from leeq.utils.ai.display_chat.notebooks import display_chat, code_to_html, dict_to_html
        spinner_id = show_spinner("AI is designing the experiment...")
        stages = get_stages_from_description(prompt)
        hide_spinner(spinner_id)

        stages_html = stages_to_html(stages)
        display_chat("Stage planning AI", 'light_blue', "The planned experiments are:<br>" + stages_html)

        super().run(stages, sub_experiment=sub_experiment, **kwargs)


AIRun = AIInstructionExperiment
AutoRun = FullyAutomatedExperiment

if __name__ == '__main__':
    from leeq.utils.ai.variable_table import VariableTable
    from leeq.utils.ai.staging.stage_execution import get_codegen_wm, CodegenModel
    from leeq.utils.ai.code_indexer import build_leeq_code_ltm
    from leeq.utils.ai.ideanet.recall_logger import RecallLogger

    prompt = "Do qubit measurement calibration to update the GMM model."
    wm = get_codegen_wm(prompt, VariableTable())
    leeq_code_ltm, exps_var_table = build_leeq_code_ltm()
    code_cog_model = CodegenModel(rounds=1)
    code_cog_model.n_recall_items = 5
    for idea in leeq_code_ltm.ideas:
        code_cog_model.lt_memory.add_idea(idea)

    with RecallLogger():
        code = code_cog_model.codegen(wm)
