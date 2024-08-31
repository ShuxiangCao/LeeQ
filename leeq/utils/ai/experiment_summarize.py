import json
from typing import Any


def get_experiment_summary(description: str, run_parameters: dict[str, Any], results_list: dict[str, str]):
    """
    Summarize the experiment results.
    """

    results_str = "".join([f"{key}: {value}" + '\n' for key, value in results_list.items()])

    prompt = f"""
    Summarize the experiment results and report the key results. Indicate if the experiment was successful or failed.
    If failed, suggest possible updates to the parameters or the experiment design if the experiment fails. The suggestion
    needs to be specific on how much of the quantity needs to be changed on the parameters. Otherwise return None for the
    parameter updates.
    
    <Run parameters>
    {json.dumps(run_parameters)}
    </Run parameters>
    
    <Experiment description> 
    {description}
    </Experiment description>
   
    <Results>
    {results_str}
    </Results>
    
    Return in json format:
    <Return>
    {{
        "analysis": str,
        "parameter_updates": "<parameter 1> should be updated by <value> and <parameter 2> should be updated by <value>",
        "success": bool,
    }}
    </Return>
    """

    import mllm
    chat = mllm.Chat(prompt, "You are a very smart and helpful assistant who only reply in JSON dict")
    res = chat.complete(parse="dict", expensive=True, cache=True)
    return res
