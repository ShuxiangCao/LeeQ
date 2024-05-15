import inspect
from typing import Optional, Any
from IPython.display import display, Markdown, Code

import asteval
from IPython.core.display_functions import display
from fibers.data_loader.module_to_tree import get_tree_for_module
from fibers.tree.node_attr.code_node import get_type, get_obj
from mllm import Chat, display_chats

from ideanet.codegen.code_diffuse import CodeDiffuser
from ideanet.codegen.code_diffuse_ideas import init_code_for_wm
from ideanet.core.idea import IdeaResult, Idea, WorkingMemory
from ideanet.core.ideabase import Ideabase

from leeq.utils import is_running_in_jupyter

_leeq_idea_base = None
_aeval = None


class LeeQExpIdea(Idea):
    def __init__(self, exp_name, exp_cls):
        super().__init__(f"{exp_name} suggestion")
        self.exp_name = exp_name
        self.exp_cls = exp_cls
        self.receptor.add_or_src([exp_name])

    def run_idea(self, w_memory: WorkingMemory):
        prompt = f"""
You are trying to use some knowledge to rewrite some Python code.
<knowledge>
Whenever you need to run experiment `{self.exp_name}`, you should create a new instance of the experiment. The experiment
will be carried out when the experiment object is created.
To create new instance: `experiment_<experiment_name> = {self.exp_cls.__name__}(argument1,argument2, ...)`
Signature:
{inspect.signature(self.exp_cls.run)}
Documentation:
{inspect.getdoc(self.exp_cls.run)}
</knowledge>
The knowledge might be useful to implement some `do` functions in the code.
{w_memory.get_in_prompt_format(tag="code_to_edit")}
<instruction>
You should notice that:
- The provided knowledge might be totally irrelevant to the `do` function.
You should output a JSON dict. The keys should be
- "code_fragment": a string of the code fragment that contains a `do` function that can be converted to python code based on the knowledge.
- "new_code_fragment": a string of code that accurately implement the code_fragment by generalizing from the knowledge. The new code should ignore the irrelevant parts of the knowledge. If part of the code_fragment cannot be implemented, use `do` functions to represent the remaining parts.
- "analysis": a string that analyze whether the code_fragment is implemented by the new_code_fragment and whether the knowledge is useful in decomposing the `do` functions in code_fragment.
- "is_proper": a boolean that indicates whether the new_code_fragment implements the required functionality described in "do" function.
</instruction>
"""
        chat = Chat(prompt,
                    "You are a very smart and helpful assistant who only reply in JSON dict")
        res = chat.complete(parse="dict", expensive=True)

        if not res["is_proper"]:
            return IdeaResult(self, False)
        suggestion = \
            f"""Original: {res["code_fragment"]}
        New: {res["new_code_fragment"]}"""
        w_memory.add_content(suggestion, "editing suggestion")

        return IdeaResult(self, True, prevent_trigger_again=True)


def build_leeq_idea_base(refresh=False) -> Ideabase:
    """
    Build the idea base for leeq. It will scan the builtin experiments and create ideas for them.

    Parameters:
        refresh (bool): Optional. Whether to refresh the idea base.

    Returns:
        Tuple[Ideabase, Intepreter]: The idea base for leeq and the loaded interpreter.
    """
    global _leeq_idea_base
    global _aeval

    if refresh:
        _leeq_idea_base = None

    if _leeq_idea_base is not None:
        return _leeq_idea_base

    aeval = asteval.Interpreter()

    from leeq.experiments import builtin

    module_root = get_tree_for_module(builtin)
    ideas = []
    from leeq import Experiment
    for node in module_root.iter_subtree_with_dfs():
        if get_type(node) == "class":
            class_obj = get_obj(node)
            if not issubclass(class_obj, Experiment):
                continue
            ideas.append(LeeQExpIdea(class_obj.__name__, class_obj))
            aeval.symtable[class_obj.__name__] = class_obj
    ideabase = Ideabase()
    for idea in ideas:
        ideabase.add_idea(idea)
    ideabase.compile()

    _leeq_idea_base = ideabase
    _aeval = aeval
    return ideabase, aeval


def do(instruction: str, symtable: Optional[dict[str, Any]] = None, display_chat: bool = False,
       display_code: bool = True):
    """
    Ask the AI to implement an instruction.

    Parameters:
        instruction (str): The instruction to implement.
        symtable (dict[str, Any]): Optional. The symbol table to use for the implementation.
        display_chat (bool): Optional. Whether to display the chat for debugging.
        display_code (bool): Optional. Whether to display the code for debugging.

    Returns:
        Any: The result of the implementation.
    """
    idea_base, aeval = build_leeq_idea_base()
    instruction_in_code = init_code_for_wm(instruction)
    diffuser = CodeDiffuser(idea_base)

    generated_code, _ = diffuser.build_functions(instruction_in_code)

    if display_code:
        if is_running_in_jupyter():
            display(Code(generated_code, language='python'))
        else:
            print(generated_code)

    aeval.eval(generated_code)

    return aeval.symtable


if __name__ == '__main__':
    instruction = """
    Do a normalized Rabi experiment with amp = 0.05. This is not MultiQubitRabi
    """
    instruction = """
    Do a drag calibration with default parameters, on dut=duts_dict['Q1']
    """

    do(instruction, display_chat=True, display_code=True)
