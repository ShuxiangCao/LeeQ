from typing import Any
from pprint import pprint


def generate_data_analysis(description: str, context: dict[str, Any]):
    """
    Generate data analysis based on the given description.

    Args:
        description (str): The description of the data analysis.

    return str: The generated data analysis code
    """
    prompt: str = f"""
    Suppose some data is collected with the following code and description:
    
    <context>
    {context}
    </context>
    
    Please write a Python code snippet that performs the data analysis based on the description provided.
    
    <description>
    {description}
    </description>
    
    The code should be a class function called `data_analysis` that takes the data saved in the class attributes as 
    input and save the analysis results into class attributes.
    
    Return the `data_analysis` function only.
    
    The format of the return should be:
    {{
    'analysis': <how did you implement the data analysis>,
    'code': <the code snippet that performs the data analysis>,
    'parameter_updates': <what class attribute have you updated to store the analyzed results>
    }}
    """

    import mllm
    chat = mllm.Chat(prompt, "You are a very smart and helpful assistant who only reply in JSON dict")
    res = chat.complete(parse="dict", cache=True)
    from pprint import pprint
    return res['code']
