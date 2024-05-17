import inspect
from typing import Optional, Any
from IPython.display import display, Markdown, Code

import asteval
from IPython.core.display_functions import display
from fibers.data_loader.module_to_tree import get_tree_for_module
from fibers.tree.node_attr.code_node import get_type, get_obj
from mllm import Chat, display_chats
from mllm.utils import parallel_map

from ideanet.codegen.code_diffuse import CodeDiffuser
from ideanet.codegen.code_diffuse_ideas import init_code_for_wm
from ideanet.core.idea import IdeaResult, Idea, WorkingMemory
from ideanet.core.ideabase import Ideabase

from leeq.utils import is_running_in_jupyter

_leeq_idea_base = None
_aeval = None


def imagine_applications(exp_cls):
    doc_string = inspect.getdoc(exp_cls.run)
    prompt = f"""
You are trying to produce imperative sentences that would invoke the execution of experiment `{exp_cls.__name__}` based on its documentation.
<docstring>
{doc_string}
</docstring>
<example>
Here are a few of im:
- Run the calibration experiment with duts=duts and start=0.0
- Calibrate `duts` 
- Please execute the CrossAllXYDragMultiSingleQubitMultilevel experiment
- Do the AllXY drag experiment.
</example>
<instruction>
You should output a JSON dict. The keys should be string of indices of the sentences and the values should be the sentences. 
Each sentence should be complete and independent. Name of the experiment should be transformed to natural language and be mentioned.
The sentences should be imperative and should be based on the documentation.
You should output 4 sentences.
</instruction>
"""
    chat = Chat(prompt)
    res = chat.complete(parse="dict", expensive=True, cache=True)
    values = list(res.values())
    return values



class LeeQExpIdea(Idea):
    def __init__(self, exp_cls):
        exp_name = exp_cls.__name__
        super().__init__(f"{exp_name} suggestion")
        self.exp_name = exp_name
        self.exp_cls = exp_cls
        self.receptor.add_or_src([exp_name])
        embedding_src = imagine_applications(exp_cls)
        self.receptor.add_or_src(embedding_src)

    def run_idea(self, w_memory: WorkingMemory):
        prompt = f"""
You are trying to use some knowledge to rewrite some Python code.
<knowledge>
Whenever you need to run experiment `{self.exp_name}`, you should create a new instance of the experiment. The experiment
will be carried out when the experiment object is created.
To create new instance: `experiment_<name> = {self.exp_cls.__name__}(argument1,argument2, ...)`
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


def build_leeq_idea_base(refresh=False) -> (Ideabase, Any):
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
        return _leeq_idea_base, _aeval

    aeval = asteval.Interpreter()

    from leeq.experiments.builtin.basic import calibrations

    module_root = get_tree_for_module(calibrations)
    classes = []
    from leeq import Experiment
    for node in module_root.iter_subtree_with_dfs():
        if get_type(node) == "class":
            class_obj = get_obj(node)
            if not issubclass(class_obj, Experiment):
                continue
            classes.append(class_obj)
            aeval.symtable[class_obj.__name__] = class_obj

    ideas = []
    for i, idea in parallel_map(LeeQExpIdea, [cls for cls in classes]):
        ideas.append(idea)

    ideabase = Ideabase()
    for idea in ideas:
        ideabase.add_idea(idea)
    ideabase.compile()

    _leeq_idea_base = ideabase
    _aeval = aeval
    return ideabase, aeval


def run_inst(instruction: str, symtable: Optional[dict[str, Any]] = None, display_chat: bool = False,
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

    if symtable is not None:
        symtable_prompt = f"""You must use only use following symbols in the function call: {",".join(list(symtable.keys()))}"""
        instruction_in_code = f"{symtable_prompt}\n{instruction_in_code}"

    instruction_in_code, _ = diffuser.build_structure(instruction_in_code)
    with display_chats():
        instruction_in_code, _ = diffuser.build_functions(instruction_in_code)

    if display_code:
        if is_running_in_jupyter():
            display(Code(instruction_in_code, language='python'))
        else:
            print(instruction_in_code)

    if symtable is not None:
        aeval.symtable.update(symtable)

    aeval.eval(instruction_in_code)

    return aeval.symtable['instruct_experiment']


if __name__ == '__main__':
    instruction = """
    Do a normalized Rabi experiment with amp = 0.05. This is not MultiQubitRabi
    """
    instruction = """
    Do a drag calibration with default parameters, on dut=duts_dict['Q1']. If fail, run the experiment again.
    """

    instruction = """
Run Ramsey experiment three times with different set_offset. Start from the default set_offset. Amplify the set_offset by 10 times each time.
"""

    run_inst(instruction, display_chat=True, display_code=True)
