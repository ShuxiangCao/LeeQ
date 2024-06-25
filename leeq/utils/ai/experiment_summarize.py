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
    
    Run parameters for this experiment:
    {run_parameters}
     
    Experiment description: 
    {description}
   
    Results:
    {results_str}
    
    Return in json format:
    {{
        "analysis": str,
        "parameter_updates": str,
        "success": bool,
    }}
    """

    import mllm
    chat = mllm.Chat(prompt, "You are a very smart and helpful assistant who only reply in JSON dict")
    res = chat.complete(parse="dict", expensive=True, cache=True)
    return res
