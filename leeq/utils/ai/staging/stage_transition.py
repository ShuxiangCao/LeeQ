from __future__ import annotations
from typing import List, TYPE_CHECKING

import mllm
from leeq.experiments import Experiment

if TYPE_CHECKING:
    from .stage_execution import Stage


def generate_new_stage_description(stage_jumped_to: Stage, additional_information: str)->str:
    """
    Generate a new stage of the stage jumped to and additional information provided.

    Parameters:
        stage_jumped_to (Stage): The stage that was jumped to.
        additional_information (str): The additional information provided.

    Returns:
        new_description (str): The new description of the stage.
    """

    prompt = f"""
You have jumped to a new stage based on the provided information. The new stage is {stage_jumped_to.label}.

 The original description of the stage is:
<description> 
{stage_jumped_to.description}
 </description>
the additional information is as follows:
<additional_information>
{additional_information}
</additional_information>
Please write a new description of the stage as the additional information requires by changing the value of arguments
of the original description only. If there is no modification to the argument, return the original description.

Return in JSON format, example:

{{
"analysis": "The thinking process of writing the new description of the stage",
"new_description": "The new description of the stage"
}}
"""

    chat = mllm.Chat(prompt)
    res = chat.complete(parse="dict", expensive=True, cache=True)

    new_description = res["new_description"]

    return new_description


def get_next_stage_label(current_stage: Stage, experiment_result: dict[str,str])->dict[str, str]:
    """
    Get the next stage label based on the current stage and the experiment object.

    Parameters:
        current_stage (Stage): The current stage.
        experiment_result (dict[str,str]): The experiment results.

    Returns:
        next_stage_label (dict[str,str]): The next stage label and the additional information for executing
        the next stage.
    """

    rules = current_stage.next_stage_guide

    result_prompt = ""

    for key, value in experiment_result.items():
        result_prompt += f"Result from {key}: {value}\n\n"

    prompt = f"""
You are operating a state machine and the current node has produced some results. You must analyze these results and use the specific rules to determine the next state of the machine.

Current stage:
{current_stage.label}:{current_stage.description}

Here are the rules:
{rules}

Here are the results from the experiments. Note that results from fitting provides the accurate 
values, but it is not valid if the inspection from the plots suggest the experiment has failed.
{result_prompt}

Based on the rules and the result provided, determine the next stage of the state machine. Return your decision in
 JSON format, including what the next state is and any additional information that modifies the arguments of the next
  stage in natural language that will be necessary for operating in the next state.
""" + """
Example JSON Output:
{
    "analysis": "<Your analysis of the results and the rules>",
    "next": "<Name of the next stage>",
    "additional_info": "<Natural language description of any information about changing the argument value of the next stage>"
}
"""

    chat = mllm.Chat(prompt)
    res = chat.complete(parse="dict", expensive=True, cache=True)
    return res
