from typing import List, Tuple, Optional
from ideanet.codegen.code_wmemory import CodeWMemoryItem
from ideanet.core.idea import WorkingMemory, WMemoryNoStimuliItem, WMemoryHiddenItem
from leeq import Experiment
from leeq.utils.ai.variable_table import VariableTable


class Stage:
    """Represents a stage in an experimental workflow."""

    def __init__(self, label: str, title: str, overview: str, description: str, next_stage_guide: str):
        self.title = title  # Title of the stage
        self.label = label  # Unique identifier for the stage
        self.overview = overview  # Overview of the stage
        self.description = description  # Description of what happens in this stage
        self.next_stage_guide = next_stage_guide  # Guidance for transitioning to the next stage
        self.var_table = None  # Variable table specific to this stage, initialized later

    def to_dict(self) -> dict:
        """Converts the stage to a dictionary."""
        return {
            "Title": self.title,
            "ExperimentDescription": self.description,
            "Next": self.next_stage_guide,
            "Overview": self.overview,
        }

    def display(self):
        """
        Display information about the stage in a jupyter notebook.
        First converts the stage to a dictionary into a markdown format.
        then display it using IPython
        """
        from IPython.display import display, Markdown

        stage_markdown = f"""##{self.title}
        **Label**: {self.label}
        **Description**: {self.description}
        **Next Stage Guide**: {self.next_stage_guide}
        """
        display(Markdown(stage_markdown))


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


def get_codegen_wm(description: str, var_table: VariableTable, hint: str = None) -> WorkingMemory:
    """
    Prepares working memory for code generation based on a description and variable table.

    Parameters:
        description (str): The description of the code to generate.
        var_table (VariableTable): The variable table to use in the code generation.
        hint (str): The hint to display in the working memory.
    Returns:
        WorkingMemory: The working memory prepared for code generation.
    """

    if hint is None:
        hint = ""
    wm = WorkingMemory()
    if not var_table.is_empty():
        var_table_in_prompt = var_table.get_local_prompt()
        prompt = f'''
{var_table_in_prompt}
'''
    wm.add_item(WMemoryNoStimuliItem(prompt, "available_variables"))
    wm.add_item(WMemoryNoStimuliItem('Call exactly one time to the experiment function / class in this edit.'))
    wm.add_item(WMemoryNoStimuliItem('Every class or function call will include the data analysis inside'
                                     'the call automatically so there is no need to do data analysis separately',
                                     "data_analyze"))
    wm.add_item(WMemoryNoStimuliItem('Always use named arguments to call functions or classes.', "argument_name"))
    wm.add_item(WMemoryNoStimuliItem(
        ('Some of the calls accept `ai_inspection` parameter. Note that since you are an AI and if the '
         'document suggest to set it to true when an AI is writing the code, set it to True explicitly.'),
        "ai_inspection"))
    # wm.add_item(WMemoryNoStimuliItem('The result of the experiment run should be saved in the exp_run variable.',
    #                                 "return_values"))
    wm.add_content(hint, "Background")
    # wm.add_content(""""
    #               If you need to execute more than one experiment at this stage, please use the following to break it into multiple steps:
    #               ```
    #               exp_run = FullyAutomatedExperiment("<The description of all the steps, including the arguments described>",
    #                    all the available variables passed in with keyword arguments that not described previously )
    #               ```
    #               """,'Multiple steps')
    prompt = f'''
do("""{description}""")
'''
    wm.add_item(CodeWMemoryItem(prompt))
    return wm


def check_if_needed_to_break_down(description: str):
    """
    Check if the stage described by the description needs to be broken down into multiple steps.

    Parameters:
        description (str): The description of the code to generate.
    """

    # Check if we need to split the stage described by the description into multiple steps
    prompt = f"""
    Evaluate whether a stage description in an experimental process should be executed as a single step or needs to be divided into multiple steps. Consider the following criteria for your evaluation:

    - Multiple Experiments: If the stage includes multiple distinct experiments, each requiring unique parameters, divide it into multiple steps.
    - Repeated Experiments with Specified Parameters: If the stage involves repeating one experiment multiple times with different parameters that are explicitly specified, divide it into multiple steps.
    - Repeated Experiments without Specified Parameters: If the stage involves repeating one experiment multiple times without explicitly specified parameters, treat it as a single step.
    - Parameter Variations Within an Experiment: If a single experiment includes variations in parameters, consider these as part of the internal management of the experiment and do not divide the stage.
    - Single Experiment, No Specified Parameters: If the stage comprises only one experiment with no parameters specified, execute it as a single step.
    
    If division into multiple steps is necessary, provide a detailed description of each step, including the parameters for each.

    Stage Description:
    {description}

    Expected return format:
    {{
        "reason": str,
        "single_step": bool,
        "steps": List[dict]
    }}
    """

    import mllm
    chat = mllm.Chat(prompt, "You are a very smart and helpful assistant who only reply in JSON dict")
    res = chat.complete(parse="dict", expensive=True, cache=True)

    return res
