import os
from typing import Tuple

from k_agents.memory.lt_memory import LongTermMemory
from k_agents.translation.agent import init_translation_agent, build_code_ltm
from k_agents.variable_table import VariableTable


def init_leeq_translation_agent():
    from leeq.experiments import builtin
    from leeq.experiments import experiments as exp
    root = os.path.dirname(exp.__file__)
    document_root = root + "/procedures"
    init_translation_agent(builtin, document_root)


def build_leeq_code_ltm(add_document_procedures=True) -> Tuple[LongTermMemory, VariableTable]:
    """
    Build the long term memory and variable table for leeq.

    Args:
        add_document_procedures (bool): Whether to add document procedures to the long term memory.

    Returns:
        Tuple[LongTermMemory, VariableTable]: The long term memory and variable table for leeq.
    """
    from leeq.experiments import builtin
    from leeq.experiments import experiments as exp
    root = os.path.dirname(exp.__file__)
    # get all the markdown files in the procedures directory
    document_paths = []
    if add_document_procedures:
        for file in os.listdir(root + "/procedures"):
            if file.endswith(".md"):
                document_paths.append(root + "/procedures/" + file)
    return build_code_ltm(builtin, document_paths)
