import os
from typing import Tuple

from k_agents.agent_group.agent_group import AgentGroup
from k_agents.translation.agent import init_translation_agents, build_code_ltm
from k_agents.variable_table import VariableTable


def init_leeq_translation_agents(document_root: str = None, n_agents_to_call: int = 3):
    from leeq.experiments import builtin
    from leeq.experiments import experiments as exp
    root = os.path.dirname(exp.__file__)
    document_root = document_root or root + "/procedures"
    init_translation_agents(builtin, document_root, n_agents_to_call)


def build_leeq_translation_agent_group(add_document_procedures=True) -> Tuple[
    AgentGroup, VariableTable]:
    """
    Build the group of translation agents for leeq.

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
    lambda_is_leeq_ai_exp_class = lambda x: issubclass(x, exp.Experiment) and x.class_obj._experiment_result_analysis_instructions is not None
    return build_code_ltm(builtin, document_paths, lambda_is_leeq_ai_exp_class)
