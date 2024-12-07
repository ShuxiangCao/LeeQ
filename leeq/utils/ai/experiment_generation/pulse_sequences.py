from leeq.utils.ai.experiment_generation.load_documents import load_document_file


def generate_pulse_sequences(overview: str, description: str):
    """
    Generate pulse sequences based on the given description.

    Args:
        overview (str): The overview of the pulse sequences
        description (str): The description of the pulse sequences.

    return str: The generated pulse sequence code
    """

    # load the pulse_sqeuences.md file in the directory of this file
    # read the file and return the content
    knowledge = load_document_file('pulse_sequences.md')

    prompt = f"""
    You are requested to write a run function that generates a pulse sequence for a quantum experiment based on the 
    overview of the experiment and the given description of the pulse sequences. You also have access to the knowledge 
    of a software package LeeQ stated below.
    
    <LeeQ Knowledge>
    {knowledge}
    </LeeQ Knowledge>
    
    Your task:
    <overview>
    {overview}
    </overview>
    
    Description of the pulse sequences:
    <description>
    {description}
    </description>
    
    Please write the code implementing the "run" class function, store all your results inside the class attributes.
    
    Reply in the following format:
    
    ```python
        <The code implementing the "run" class function>
    ```
    """

    import mllm
    chat = mllm.Chat(prompt, "You are a very smart and helpful coding assistant.", dedent=True)
    res = chat.complete(parse="quotes", cache=True)

    return {'code': res}
