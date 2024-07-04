from typing import Optional, Dict, Any, Union, List
import numpy as np

from labchronicle import register_browser_function, log_and_record
from mllm import Chat

from ideanet.codegen.code_wmemory import CodeWMemoryItem
from ideanet.core.lt_memory import LongTermMemory, RecallResult, IdeaResult, Idea
from ideanet.core.w_memory import WorkingMemory
from ideanet.utils.logger import RecallLogger

from leeq.utils import setup_logging
from leeq.utils.notebook import show_spinner, hide_spinner

from IPython.core.display import display, HTML
from leeq.experiments import Experiment

logger = setup_logging(__name__)

__all__ = ["AIInstructionExperiment", "FullyAutomatedExperiment", "AIRun", "AutoRun"]


class CodegenIdea(Idea):
    """
    Generate the code based on the working memory
    Will put the generated code in the working memory
    """

    def __init__(self):
        super().__init__("CodegenIdea")

    def get_score(self, w_memory: WorkingMemory):
        if not w_memory.has_tag("code_suggestion"):
            return -1.0
        return 1.0

    def run_idea(self, w_memory: WorkingMemory) -> IdeaResult:
        chat = Chat()
        chat += """
        Your task is to generate new code for the context described below.
        """
        chat += w_memory.get_in_prompt_format(tag="context", tags_to_ignore=["comment"])
        chat += """
        <instruction>
        You are required to generate new code that can be used to replace the <code_to_complete> based on <code_suggestion>.
        The new code should absolutely just be what should appear in the place of # [slot]. You should not output the full code. Just the slot.
        Your code should start with no indentation.
        Some of the <code_suggestion> might be misleading. But you should pick the most relevant one.
        The new code should not import any external modules.
        Notice that
        - The existing attempted code might be totally wrong. For example, it might have some additional parts other than the given slot of <code_to_complete>.
        - You should never use function/object/module that is not mentioned in the context
        - If you think current context is not enough to generate the code, you can write the comment of what to do and put ... as a placeholder
        Output a JSON dict with the following keys:
        "analysis" (string): an analysis of the current situation. Especially, focusing on how to generate the code.
        "code" (string): the new code that can fill the slot in <code_to_complete>.
        </instruction>
        """
        res = chat.complete(parse="dict", expensive=True)["code"]
        idea_res = IdeaResult(self, True)
        code_item = CodeWMemoryItem(res, tag="attempted_code")
        idea_res.add_new_wm_item(code_item)
        idea_res.tags_to_remove = ["attempted_code", "code_suggestion"] # remove the old attempted code
        return idea_res


class CodegenModel:
    lt_memory: LongTermMemory
    n_recall_items: int

    def __init__(self, rounds=2):
        self.lt_memory = LongTermMemory()
        self.codegen_idea = CodegenIdea()
        self.n_recall_items = 10
        self.rounds = rounds

    def recall(self, wm: WorkingMemory) -> RecallResult:
        """
        Recall ideas from long term memory, using what is currently in the working memory.

        :param wm: the working memory to stimuli ideas
        :return: the result of triggered ideas
        """
        res = self.lt_memory.recall_by_wm(wm, top_k=self.n_recall_items)
        return res

    def codegen(self, wm: WorkingMemory) -> str:
        """
        Generate code from working memory, updates working memory with recalled ideas in the process.

        :param wm: the working memory to generate code from
        :return: source code

        Preconditions:
            - there exists an item in wm tagged with 'completed_code' after at most 100 recalls.
        """
        for i in range(self.rounds):
            recall_res = self.recall(wm)
            wm.update_by_recall_res(recall_res, to_tick=True)
            #print("Generating code...")
            idea_res = self.codegen_idea.run_idea(wm)
            recall_res = RecallResult([idea_res])
            wm.update_by_recall_res(recall_res, to_tick=False)
        code = wm.extract_tag_contents("attempted_code")
        if len(code) > 0:
            return code[0]


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

        self.stages: List[Stage] = stages

        leeq_code_ltm, exps_var_table = build_leeq_code_ltm()
        code_cog_model = CodegenModel(rounds=2)
        for idea in leeq_code_ltm.ideas:
            code_cog_model.lt_memory.add_idea(idea)
        code_cog_model.n_recall_items = 5  # Number of items to recall in cognitive model
        var_table: VariableTable = VariableTable()
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

            codegen_wm = get_codegen_wm(stage.description, input_var_table, hint=prompt)

            #if stage.title not in coding_ltm_cache:
            #    recall_res = code_cog_model.recall(codegen_wm)
            #    coding_ltm_cache[stage.title] = recall_res
            #else:
            #    recall_res = coding_ltm_cache[stage.title]

            codes = code_cog_model.codegen(codegen_wm)
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
                        continue
                    self.experiment_history.append(exp_object)
                    experiment_result = exp_object.get_ai_inspection_results()
                    break
                except Exception as e:
                    raise e
                    print(e)
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
                if stage.label == next_stage_label:
                    next_stage = stage
                    break
            else:
                assert False, f"Next stage label {next_stage_label} not found in stages"

            new_description = generate_new_stage_description(next_stage, additional_info)
            hide_spinner(spinner_id)
            display_chat("Stage Planning AI", 'light_blue', f"Transitioning to the next"
                                                            f" stage {next_stage.label} with the following description:<br>{new_description}<br>"
                                                            f"{next_stage_info['analysis']}")
            next_stage.description = new_description
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
        Experiment: The last experiment run in the staged experiment.
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
        # label: str, title: str, overview: str, description: str, next_stage_guide: str
        stage = Stage(label="Stage1", title="Implement experiment",
                      overview='You are requested to implement one experiment and modify the parameter to make it success.',
                      description=prompt, next_stage_guide="Go to Complete if success. "
                                                           "Go back to the same stage if the experiment failed and the parameters should be adjusted."
                                                           "Go to Fail if the experiment failed and the parameters cannot be adjusted."
                                                           "Go to Fail if the experiment has failed after 3 attempts."
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
    from leeq.utils.ai.staging.stage_execution import get_codegen_wm
    from leeq.utils.ai.code_indexer import build_leeq_code_ltm
    prompt = "Do qubit measurement calibration to update the GMM model."
    wm = get_codegen_wm(prompt, VariableTable())
    leeq_code_ltm, exps_var_table = build_leeq_code_ltm()
    code_cog_model = CodegenModel(rounds=2)
    code_cog_model.n_recall_items = 5
    for idea in leeq_code_ltm.ideas:
        code_cog_model.lt_memory.add_idea(idea)
    with RecallLogger():
        code = code_cog_model.codegen(wm)


