from typing import List, Tuple, Optional

import mllm.debug

from leeq.utils.ai.code_indexer import build_leeq_code_ltm
from ideanet.codegen.code_wmemory import CodeWMemoryItem
from ideanet.codegen.codegen_cog_model import CodegenModel
from ideanet.core.idea import WorkingMemory, WMemoryNoStimuliItem
from leeq import Experiment
from stage_transition import get_next_stage_label, generate_new_stage_description
from var_table import VariableTable

class Stage:
    """Represents a stage in an experimental workflow."""

    def __init__(self, label: str, title: str, description: str, next_stage_guide: str):
        self.title = title  # Title of the stage
        self.label = label  # Unique identifier for the stage
        self.description = description  # Description of what happens in this stage
        self.next_stage_guide = next_stage_guide  # Guidance for transitioning to the next stage
        self.var_table = None  # Variable table specific to this stage, initialized later


class StageRunner:
    """Manages the execution of multiple stages in an experiment."""

    def __init__(self, stages: List[Stage], input_var_table: Optional[VariableTable] = None):
        """
        Initialize the StageRunner with a list of stages and an optional input variable table.

        Parameters:
            stages (List[Stage]): The list of stages to execute.
            input_var_table (Optional[VariableTable]): The input variable table to use in the execution.
        """
        self.stages: List[Stage] = stages
        leeq_code_ltm, exps_var_table = build_leeq_code_ltm()
        self.code_cog_model = CodegenModel(leeq_code_ltm)
        self.code_cog_model.n_recall_items = 5  # Number of items to recall in cognitive model
        self.var_table: VariableTable = VariableTable()
        self.var_table.add_parent_table(exps_var_table)
        if input_var_table is not None:
            self.var_table.add_parent_table(input_var_table)
        self.input_var_table = input_var_table
        self.n_step_multiplier = 6  # Multiplier to control the number of execution steps

    def make_var_table(self) -> Tuple[VariableTable, str]:
        """
        Generates a consolidated variable table and a string prompt from all stages.

        Returns:
            Tuple[VariableTable, str]: A tuple containing the variable table and the string prompt.
        """
        var_table = VariableTable()
        in_prompt_var_table = []
        for stage in self.stages:
            if stage.var_table is not None:
                in_prompt_var_table.append(f"{stage.label}: {stage.title}")
                in_prompt_var_table.append(stage.var_table.get_local_prompt())
                var_table.update_by_other_table(stage.var_table)
        in_prompt_var_table = "\n".join(in_prompt_var_table)
        return var_table, in_prompt_var_table

    def run(self):
        """
        Executes the stages sequentially until completion or failure.
        """
        curr_stage = self.stages[0]
        for step in range(len(self.stages) * self.n_step_multiplier):
            with mllm.debug.display_chats(1):
                new_var_table = self.run_stage_description(curr_stage)
            with mllm.debug.display_chats():
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

    def run_stage_description(self, stage: Stage) -> VariableTable:
        """
        Generates a new variable table by processing a stage's description with cognitive models.

        Parameters:
            stage (Stage): The stage to process.

        Returns:
            VariableTable: The new variable table generated after processing the stage description.
        """
        codegen_wm = get_codegen_wm(stage.description, self.input_var_table)
        recall_res = self.code_cog_model.recall(codegen_wm)
        new_codegen_wm = self.code_cog_model.act(codegen_wm, recall_res)
        new_var_table = self.var_table.new_child_table()
        new_var_table.interpret(get_code_from_wm(new_codegen_wm))
        return new_var_table

def get_exp_from_var_table(var_table: VariableTable) -> Optional[Experiment]:
    """Searches the variable table for an Experiment object."""
    for name, obj in var_table.variable_objs.items():
        if isinstance(obj, Experiment):
            return obj

def get_code_from_wm(wm: WorkingMemory) -> str:
    """
    Extracts code from the working memory that contains code items.

    Parameters:
        wm (WorkingMemory): The working memory to extract code from.

    Returns:
        str: The extracted code.
    """
    code = ""
    for item in wm._items:
        if isinstance(item, CodeWMemoryItem):
            code = item.content
            break
    return code

def get_codegen_wm(description: str, var_table: VariableTable) -> WorkingMemory:
    """
    Prepares working memory for code generation based on a description and variable table.

    Parameters:
        description (str): The description of the code to generate.
        var_table (VariableTable): The variable table to use in the code generation.

    Returns:
        WorkingMemory: The working memory prepared for code generation.
    """
    wm = WorkingMemory()
    if not var_table.is_empty():
        var_table_in_prompt = var_table.get_local_prompt()
        prompt = f'{var_table_in_prompt}\ndo("""{description}""")'
        wm.add_item(WMemoryNoStimuliItem(prompt, "available_variables"))
    return wm
