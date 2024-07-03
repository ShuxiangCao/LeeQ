import inspect
import os
from typing import Any, Tuple, Type, List

from fibers.data_loader.module_to_tree import get_tree_for_module
from fibers.tree.node_attr.code_node import get_type, get_obj
from mllm import Chat
from mllm.utils import parallel_map

from ideanet.codegen.code_wmemory import CodeEditingItem
from ideanet.core.idea import IdeaResult, WorkingMemory, LongTermMemory, EmbedIdea
from .variable_table import VariableTable


def imagine_applications(exp_name, exp_docs) -> List[str]:
    """
    Generate a list of imperative sentences based on the documentation of an experiment class method.

    Args:
        exp_cls (Type[Any]): The experiment class.

    Returns:
        List[str]: A list of imperative sentences derived from the experiment's documentation.
    """
    # Retrieve the docstring for the `run` method of the experiment class
    doc_string = exp_docs
    # Construct the prompt for the Chat model
    prompt = f"""
You are trying to produce imperative sentences that would invoke the execution of experiment `{exp_name}` based on its documentation.
<docstring>
{doc_string}
</docstring>
<example>
Here are a few of examples of imperative sentences:
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
    # Instantiate a Chat model and get responses
    chat = Chat(prompt)
    res = chat.complete(parse="dict", expensive=True, cache=True)
    # Extract the values from the response dictionary
    values = list(res.values())
    return values


class LeeQExpIdea(EmbedIdea):
    def __init__(self, exp_cls: Type[Any]):
        """
        Initialize an idea for triggering and embedding experiment-based sentences.

        Args:
            exp_cls (Type[Any]): The experiment class to be considered.
        """
        exp_name = exp_cls.__name__
        self.exp_name = exp_name
        self.exp_cls = exp_cls
        if "needing_situation" in exp_cls.__dict__:
            embedding_src = exp_cls.needing_situations
        else:
            # Generating sentences for the idea
            embedding_src = imagine_applications(exp_cls.__name__, inspect.getdoc(exp_cls.run))
        triggering_src = [exp_name] + embedding_src
        super().__init__(f"{exp_name} suggestion", triggering_src)

    def run_idea(self, w_memory: WorkingMemory) -> IdeaResult:
        """
        Execute the idea using the provided working memory, returning an IdeaResult.

        Args:
            w_memory (WorkingMemory): The current working memory instance.

        Returns:
            IdeaResult: The result of executing the idea, possibly modifying working memory.
        """
        # Create a detailed prompt for the Chat model
        prompt = f"""
You are trying to use some knowledge to rewrite some Python code.
<knowledge>
Whenever you need to run experiment `{self.exp_name}`, you should create a new instance of the experiment. The experiment
will be carried out when the experiment object is created.
To create new instance: `experiment_<name> = {self.exp_cls.__name__}(argument1,argument2, ...)`
Signature:
<singature> {inspect.signature(self.exp_cls.run)} </signature>
Documentation:
<documentation> 
{inspect.getdoc(self.exp_cls.run)} 
</documentation>
</knowledge>
The knowledge might be useful to implement some `do` functions in the code.
{w_memory.get_in_prompt_format(tag="code_to_edit")}
<instruction>
You should notice that:
- The provided knowledge might be totally irrelevant to the `do` function.
You should output a JSON dict. The keys should be
- "relation": a string of the relation between the code to edit and the experiment, including the similarity and difference.
- "applicable": a boolean that indicates whether the experiment is useful in implementing the `do` functions in the code fragment. If not useful, output empty string in "code_fragment".
- "code_fragment": a string of the code fragment that contains a `do` function that can be converted to python code based on the knowledge.
- "new_code_fragment": a string of code that accurately implement the code_fragment by generalizing from the knowledge. The new code should ignore the irrelevant parts of the knowledge. If part of the code_fragment cannot be implemented, use `do` functions to represent the remaining parts.
- "note": a string that describes why the replacement might not be proper.
</instruction>
"""
        # Generate and handle the response
        chat = Chat(prompt, "You are a very smart and helpful assistant who only reply in JSON dict")
        res = chat.complete(parse="dict", expensive=True)

        if not res["applicable"]:
            return IdeaResult(self, False)

        idea_res = IdeaResult(self, True, suppress_ticks=0)
        idea_res.add_new_wm_item(CodeEditingItem(res["code_fragment"], res["new_code_fragment"], note=res["note"]))

        return idea_res


def add_leeq_exp_to_ltm(lt_memory: LongTermMemory, var_table: VariableTable, exp_cls: Type[Any]) -> None:
    """
    Add an experiment class to the long term memory and variable table for leeq.

    Args:
        lt_memory (LongTermMemory): The long term memory for leeq.
        var_table (VariableTable): The variable table for leeq.
        exp_cls (Type[Any]): The experiment class to be added.
    """
    idea = LeeQExpIdea(exp_cls)
    lt_memory.add_idea(idea)
    var_table.add_variable(exp_cls.__name__, exp_cls, exp_cls.__name__)


def build_leeq_code_ltm() -> Tuple[LongTermMemory, VariableTable]:
    """
    Build the idea base for leeq. It scans built-in experiments and creates ideas for them.

    Returns:
        Tuple[LongTermMemory, VariableTable]: The long term memory for leeq and the loaded variable table.
    """
    from leeq.experiments import builtin

    lt_memory = LongTermMemory()
    var_table = VariableTable()

    # Load the module root and scan for experiment classes
    module_root = get_tree_for_module(builtin)
    classes = []
    from leeq import Experiment
    for node in module_root.iter_subtree_with_dfs():
        if get_type(node) == "class":
            class_obj = get_obj(node)
            if not issubclass(class_obj, Experiment):
                continue
            elif not class_obj.is_ai_compatible():
                continue
            classes.append(class_obj)

    # Load the AI automated experiment class for nested execution.
    from leeq.experiments import FullyAutomatedExperiment
    classes.append(FullyAutomatedExperiment)

    def _add_leeq_exp_to_ltm(exp_cls: Type[Any]):
        add_leeq_exp_to_ltm(lt_memory, var_table, exp_cls)

    for i, idea in parallel_map(_add_leeq_exp_to_ltm, [cls for cls in classes], title = "Adding experiment to memory"):
        ...

    from leeq import experiments as exp
    from .procedure_indexer import extract_procedures_to_lt_memory
    root = os.path.dirname(exp.__file__)
    #extract_procedures_to_lt_memory(root + "/procedures/calibration.md", var_table,
    #                                lt_memory)

    return lt_memory, var_table
