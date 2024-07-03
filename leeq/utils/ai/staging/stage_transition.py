from __future__ import annotations
from typing import List, TYPE_CHECKING

import mllm
from leeq.experiments import Experiment

if TYPE_CHECKING:
    from .stage_execution import Stage


def generate_new_stage_description(stage_jumped_to: Stage, additional_information: str) -> str:
    """
    Generate a new stage of the stage jumped to and additional information provided.

    Parameters:
        stage_jumped_to (Stage): The stage that was jumped to.
        additional_information (str): The additional information provided.

    Returns:
        new_description (str): The new description of the stage.
    """

    prompt = f"""
Based on the information provided, you have transitioned to a new stage, identified as {stage_jumped_to.label}.

The current description of this stage is:
{stage_jumped_to.description}

If there is any additional information, it is detailed below:
{additional_information}

Using the details provided, write an updated description for this stage. Specifically, if the number of attempts is
mentioned in the additional information, include this in your description. Furthermore, if there are instructions in the
additional information to adjust certain parameters, select specific values for each parameter as requested and justify
these choices based on the analysis provided.  Include your analysis only in the analysis field and aim for conciseness
and clarity in your revised description. Do not include the objective into the description.

Example of the description:
"Conduct the <experiment name> with parameters <parameter list for experiment>."

Return the response in the following JSON format:

{{
    "analysis": "Describe your thought process for updating the stage description.",
    "new_description": "Provide the updated description of the stage here."
}}

"""

    chat = mllm.Chat(prompt)
    res = chat.complete(parse="dict", expensive=True, cache=True)

    new_description = res["new_description"]

    return new_description


def get_next_stage_label(current_stage: Stage, experiment_result: dict[str, str]) -> dict[str, str]:
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
        if key == 'Suggested parameter updates':
            if experiment_result['Experiment success']:
                result_prompt += f"Result from {key}: {None}\n\n"
                continue

        result_prompt += f"Result from {key}: {value}\n\n"

    prompt = f"""
You are operating a state machine and the current node has produced some results. You must analyze these results and use the specific rules to determine the next state of the machine.

Current stage:
{current_stage.label}:{current_stage.description}

Here are the rules:
{rules}

Here are the results from the experiments. Note that results from fitting and the inspection must be consistant to indicate
the validity. Otherwise they are both invalid.
{result_prompt}

Based on the rules and the result provided, determine the next stage of the state machine. If the current stage has 
been tried 3 times and still failed, the next stage should be "Fail". Do not retry the current stage if the experiment
has been successful.
 
Return your decision in JSON format, including what the next state is and any additional information such as the results
from the current experiment that indicates the arguments of the next stage in natural language that will be necessary
for operating in the next state. The next stage does not posses the information of the current stage.

Do not pass any additional information if the next stage is different from the current stage. 

   If the next stage is the same as the current stage, include the number of tries in the additional info.
""" + """
Example JSON Output:
{
    "analysis": "<Your analysis of the results and the rules>",
    "next": "<Name of the next stage>",
    "additional_info": "<Additional information to be passed to the next stage>"
}
"""

    chat = mllm.Chat(prompt)
    res = chat.complete(parse="dict", expensive=True, cache=True)
    return res
