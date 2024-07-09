from typing import List

from tqdm.notebook import tqdm
from leeq.utils.ai.display_chat.notebooks import display_chat, code_to_html
from leeq.utils.ai.experiment_generation.data_analysis import generate_data_analysis
from leeq.utils.ai.experiment_generation.data_visualization import generate_data_visualization
from leeq.utils.ai.experiment_generation.load_documents import load_document_file
from leeq.utils.ai.experiment_generation.pulse_sequences import generate_pulse_sequences


def summarize_experiment(experiment_summary: str, code_fragments: List[str]):
    """
    Summarize the given code fragments into a single experiment.

    Args:
        experiment_summary (str): The summary of the experiment.
        code_fragments (List[str]): The code fragments to summarize.

    return str: The summarized experiment code
    """

    prompt = f"""
    There are a few python code snippit that have been generated for the experiment. Please assemble them into a 
    single working experiment class.
    
    <experiment_summary>
    {experiment_summary}
    </experiment_summary>
    
    The code fragments are:
    <code_fragments>
    {code_fragments}
    </code_fragments>
    
    Please assemble the code fragments into a single working experiment class.
    
    Return format:
    {{
        "code": <The experiment class code>
    }}
    """

    import mllm
    chat = mllm.Chat(prompt, "You are a very smart and helpful assistant who only reply in JSON dict")
    res = chat.complete(parse="dict", cache=True)
    return res["code"]


def add_comments_annotations_and_gagets(summary: str, code: str):
    """
    Add comments, annotations, and gadgets to the given code.
    """

    leeq_example = load_document_file('full_class_example.md')
    leeq_loading = load_document_file('imports_and_loading.md')

    prompt = f"""
    There is a code snippet that has been generated for implementing a quantum experiment. We wish to modify the code 
    to make it compatible with the LeeQ package. The code syntax is correct, but it lacks comments, annotations, and 
    decorators. It also may not inherent from the `Experiment` class from leeq.  
    
    <leeq_example>
    {leeq_example}
    </leeq_example>
    
    <leeq_imports>
    {leeq_loading}
    </leeq_imports>
    
    <summary> 
    {summary}
    </summary>
    
    <code>
    {code}
    </code>
    
    - Please add decorators to the code to make it compatible with the LeeQ package. 
    - Make it also inherent from the `Experiment` class from leeq. 
    - Do not use the '__init__' function, instead use the 'run' function to initialize the class.
    - Add comments to the code to make it more readable.
    - Add type annotations to all class arguments to the code to make it more understandable.
    - Add docstrings to the class and all class functions to make it more understandable.
    - Do not change the structure of the code, add or remove class functions.
    - You have to the full code after modification and redy to be executed. Do not return the code in parts.
    
    Return format:
    {{
        "analysis": <Your thoughts on how to implement the code>,
        "code": <The modified code in string>
    }}
    """

    import mllm
    chat = mllm.Chat(prompt, "You are a very smart and helpful assistant who only reply in JSON dict")
    res = chat.complete(parse="dict", cache=True)

    return res["code"]


def break_down_description(description: str):
    """
    Break down the given description into code fragments.

    Args:
        description (str): The description to break down.

    return dict[str,str]: requirements
    """
    prompt = f"""
    Please read the following information about implementing an quantum computing experiment.
    <description>
    {description}
    </description>
    
    Please extract the following information from the description and provide the following information in JSON format:
    1. Summary of the experiment
    2. The description of what pulse sequences should be used
    3. The description of how to implement data analysis
    4. The description of how to do data visualization
    
    If the information about certain quantities such as the sweep period is missing, consider it is an input parameter 
    in the later stage and do not consider it is missing.
    And reflect that in the description. The details about how to operate the experiment will be provided in in the later stage.
    
    If the information is not provided in the description, please suggest the information is missing in the field.
    
    Please provide the information in the following format:
    {{
        "summary": <The summary of the experiment>,
        "pulse_sequences": <The description of what pulse sequences should be used, should contain what gate to apply, is there a delay and what to measure>,
        "data_analysis": "The description of how to implement data analysis>,
        "data_visualization": <The description of how to do data visualization>,
    }}
    """

    import mllm
    chat = mllm.Chat(prompt, "You are a very smart and helpful assistant who only reply in JSON dict")
    res = chat.complete(parse="dict", cache=True)

    return res


def generate_experiment(description: str, display_progress=True):
    """
    Generate an experiment based on the given description.

    Args:
        description (str): The description of the experiment.
        display_progress (bool): Whether to display a progress bar while generating the experiment.

    return cls: The generated experiment class
    """
    steps = [
        "Analyzing the experiment description",
        "Generating pulse sequences",
        "Generating data analysis",
        "Generating data visualization",
        "Summarizing the experiment"
    ]

    if display_progress:
        progress_bar = tqdm(total=len(steps), desc="Experiment Generation Progress")

    descriptions = break_down_description(description)

    if display_progress:
        progress_bar.set_description(steps[0])
        progress_bar.update(1)

    pulse_sequences = generate_pulse_sequences(overview=descriptions['summary'],
                                               description=descriptions["pulse_sequences"])

    if display_progress:
        progress_bar.set_description(steps[1])
        progress_bar.update(1)

    data_analysis = generate_data_analysis(descriptions["data_analysis"], context=pulse_sequences)

    if display_progress:
        progress_bar.set_description(steps[2])
        progress_bar.update(1)

    data_visualization = generate_data_visualization(descriptions["data_visualization"], context=data_analysis)

    code_fragments = {
        "pulse_sequences": pulse_sequences,
        "data_analysis": data_analysis,
        "data_visualization": data_visualization
    }

    if display_progress:
        progress_bar.set_description(steps[3])
        progress_bar.update(1)

    experiment_code = summarize_experiment(experiment_summary=descriptions["summary"], code_fragments=code_fragments)

    print(experiment_code)

    experiment_code = add_comments_annotations_and_gagets(summary=descriptions["summary"], code=experiment_code)

    print(experiment_code)

    if display_progress:
        progress_bar.set_description(steps[4])
        progress_bar.update(1)
        progress_bar.close()

    html = code_to_html(experiment_code)
    display_chat(agent_name="Experiment generation agent", background_color='light_purple', content=html)
    return experiment_code
