import os

from k_agents.translation.agent import prepare_translation_agents
from k_agents.translation.env import TranslationAgentEnv


def init_leeq_translation_agents(document_root: str = "/procedures", n_agents_to_call: int = 3):
    from leeq.experiments import builtin
    from leeq.experiments import experiments as exp
    root = os.path.dirname(exp.__file__)
    if document_root is not None:
        document_root = root + document_root

    def lambda_is_leeq_ai_exp_class(x): return issubclass(x,
                                                          exp.Experiment) and x._experiment_result_analysis_instructions is not None

    def add_class_to_var_table(var_table, exp_cls):
        var_table.add_variable(exp_cls.__name__, exp_cls, exp_cls.__name__)

    translation_agents, translation_var_table = prepare_translation_agents(builtin, document_root, n_agents_to_call, lambda_is_leeq_ai_exp_class, add_class_to_var_table)
    env = TranslationAgentEnv()
    env.translation_agents = translation_agents
    env.translation_var_table = translation_var_table
