from pprint import pprint
from typing import Any


def generate_data_visualization(description: str, context: dict[str, Any]):
    """
    Generate data visualization based on the given description.

    Args:
        description (str): The description of the data visualization.
        context (dict[str, Any]): The context of the data visualization.

    return str: The generated data visualization code
    """
    pass

    prompt: str = f"""
    Suppose some data is collected and analyzed with the following code and description:

    <context>
    {context}
    </context>

    Please write a Python code snippet that visualize the data based on the description provided.

    <description>
    {description}
    </description>

    The code should be a class function called `data_visualization` that takes the data saved in the class attributes as 
    input. At the begining of the function, you should call the `data_analysis` function to make sure the data is analyzed.
    
    Return a plotly figure or a matplotlib figure as the output, instead of show them directly.

    Return the `data_visualization` function only.

    The format of the return should be:
    ```python
    <The data visualization code>
    ```
    """

    import mllm
    chat = mllm.Chat(prompt, "You are a very smart and helpful coding assistant.")
    res = chat.complete(parse="quotes", cache=False)
    return res
