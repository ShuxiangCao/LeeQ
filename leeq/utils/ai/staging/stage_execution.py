from typing import List, Tuple, Optional
from leeq.utils.ai.ideanet.code_wmemory import CodeWMemoryItem
from leeq.utils.ai.ideanet.lt_memory import RecallResult, LongTermMemory, IdeaResult, Idea
from leeq.utils.ai.ideanet.w_memory import WorkingMemory, WMemoryNoStimuliItem, WMemoryHiddenItem
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
            "Label": self.label,
            "Title": self.title,
            "ExperimentDescription": self.description,
            "Next": self.next_stage_guide,
            "Overview": self.overview,
        }

    def to_xml(self) -> str:
        """Converts the stage to an XML string."""
        stage_dict = self.to_dict()
        xml_str = "<Stage>\n"
        for key, value in stage_dict.items():
            xml_str += f"  <{key}>{value}</{key}>\n"
        xml_str += "</Stage>"
        return xml_str

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


def get_exp_from_var_table(var_table: VariableTable) -> Optional['Experiment']:
    """Searches the variable table for an Experiment object."""
    from leeq import Experiment
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
        wm.add_item(CodeWMemoryItem(prompt, "available_variables").set_no_stimuli())
    notices = """
- Call exactly one time to the experiment function / class in this edit.
- Every class or function call will include the data analysis inside the call automatically so there is no need to do data analysis separately.
- Always use named arguments to call functions or classes.
- Store the return value of the call functions or classes to a variable.
"""
    wm.add_item(CodeWMemoryItem(notices, tag="notices").set_no_stimuli())
    # wm.add_item(WMemoryNoStimuliItem('The result of the experiment run should be saved in the exp_run variable.',
    #                                 "return_values"))
    if hint:
        wm.add_content(CodeWMemoryItem(hint, "background"))
    # wm.add_content(""""
    #               If you need to execute more than one experiment at this stage, please use the following to break it into multiple steps:
    #               ```
    #               exp_run = FullyAutomatedExperiment("<The description of all the steps, including the arguments described>",
    #                    all the available variables passed in with keyword arguments that not described previously )
    #               ```
    #               """,'Multiple steps')
    prompt = f'''
# [slot: {description}]
'''
    wm.add_item(CodeWMemoryItem(prompt, tag="code_to_complete").set_no_stimuli())
    wm.add_item(WMemoryHiddenItem([description]))
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
    - If this is not the first attempt at the stage, do not divide it into multiple steps.
    
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
        from mllm import Chat
        chat = Chat()
        chat += """
        Your task is to generate new code for the context described below.
        """
        chat += w_memory.get_in_prompt_format(tag="context", tags_to_ignore=["comment"])
        chat += """
        <instruction>
        You are required to adopt code that can be used to replace the <code_to_complete> from <code_suggestion>.
        The adopted code should absolutely just be what should appear in the place of # [slot]. 
        You should just choose the code and fill it into the slot.
        Some of the <code_suggestion> might be misleading. But you should pick the most relevant one. You have to pick one of the suggestions unless there is no suggestion.
        If no suggestion exist, you can write the comment of what to do and put ... as a placeholder
        Output a JSON dict with the following keys:
        "analysis" (string): an analysis of the current situation. Especially, focusing on how to generate the code.
        "code" (string): the new code that can fill the slot in <code_to_complete>.
        </instruction>
        """
        res = chat.complete(parse="dict", expensive=True)["code"]
        idea_res = IdeaResult(self, True)
        code_item = CodeWMemoryItem(res, tag="attempted_code")
        idea_res.add_new_wm_item(code_item)
        idea_res.tags_to_remove = ["attempted_code", "code_suggestion"]  # remove the old attempted code
        return idea_res


class CodegenModel:
    lt_memory: LongTermMemory
    n_recall_items: int

    def __init__(self):
        self.lt_memory = LongTermMemory()
        self.codegen_idea = CodegenIdea()
        self.n_recall_items = 10
        self._cached_recall_res = None

    def recall(self, wm: WorkingMemory) -> RecallResult:
        """
        Recall ideas from long term memory, using what is currently in the working memory.

        :param wm: the working memory to stimuli ideas
        :return: the result of triggered ideas
        """
        res = self.lt_memory.recall_by_wm(wm, top_k=self.n_recall_items)
        self._cached_recall_res = res
        return res

    def codegen(self, wm: WorkingMemory, recall_res: dict = None) -> str:
        """
        Generate code from working memory, updates working memory with recalled ideas in the process.

        Parameters:
        - wm: the working memory to generate code from
        - recall_res: the recall result from the long term memory

        Preconditions:
            - there exists an item in wm tagged with 'completed_code' after at most 100 recalls.
        """

        if recall_res is None:
            if self._cached_recall_res is None:
                recall_res = self.recall(wm)
            else:
                recall_res = self._cached_recall_res

        wm.update_by_recall_res(recall_res, to_tick=True)
        idea_res = self.codegen_idea.run_idea(wm)
        recall_res = RecallResult([idea_res])
        wm.update_by_recall_res(recall_res, to_tick=False)
        code = wm.extract_tag_contents("attempted_code")
        if len(code) > 0:
            return code[0]
