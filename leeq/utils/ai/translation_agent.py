import os
from typing import Tuple

from k_agents.agent_group.agent_group import AgentGroup
from k_agents.translation.agent import init_translation_agent, build_code_ltm
from k_agents.variable_table import VariableTable


def init_leeq_translation_agent(document_root: str = None):
    from leeq.experiments import builtin
    from leeq.experiments import experiments as exp
    root = os.path.dirname(exp.__file__)
    document_root = document_root or root + "/procedures"
    init_translation_agent(builtin, document_root)


def build_leeq_code_ltm(add_document_procedures=True) -> Tuple[
    AgentGroup, VariableTable]:
    """
    Build the long term memory and variable table for leeq.

    Args:
        add_document_procedures (bool): Whether to add document procedures to the long term memory.

    Returns:
        Tuple[AgentGroup, VariableTable]: The long term memory and variable table for leeq.
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
