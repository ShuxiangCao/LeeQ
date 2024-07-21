from typing import Any
from fibers.tree.node_attr.code_node import get_type, get_obj


def _load_fitting_function_docstrings():
    """
    Load the docstrings of the fitting functions from the leeq package.
    """
    from leeq.theory.fits import fit_exp
    from fibers.data_loader.module_to_tree import get_tree_for_module
    module_root = get_tree_for_module(fit_exp)

    available_functions = {}

    for node in module_root.iter_subtree_with_dfs():
        if get_type(node) == "function":
            func_obj = get_obj(node)
            if func_obj.__doc__ is None:
                continue
            if func_obj.__name__.startswith("_"):
                continue
            available_functions[func_obj.__name__] = func_obj.__doc__

    return available_functions


def function_signature_to_xml(key, val):
    """
    Convert function signature to xml format
    """
    return f"<function><name>{key}</name>\n<document>{val}</document></function>\n"


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
    
    You can use the following fitting functions from the leeq package:
    <fitting_functions>
    {"".join([function_signature_to_xml(x, y) for x, y in _load_fitting_function_docstrings().items()])}
    </fitting_functions>
    
    If the above functions are not enough, you have to create your own fitting functions.
    
    Return the `data_analysis` function only.
   
    The format of the return should be:
    ```python
    <The data analysis code>
    ```
    """

    import mllm
    chat = mllm.Chat(prompt, "You are a very smart and helpful coding assistant.")
    res = chat.complete(parse="quotes", cache=False)

    return revise_data_analysis(description, res)


def revise_data_analysis(description: str, code: dict[str, Any]):
    """
    Revise the data analysis based on the given description.

    Args:
        description (str): The description of the data analysis.

    return str: The revised data analysis code
    """
    prompt: str = f"""
    Please revise the Python code snippet that performs the data analysis based on the description provided.
    
    <description>
    {description}
    </description>
    
    The code should be a class function called `data_analysis` that takes the data saved in the class attributes as 
    input and save the analysis results into class attributes.
    
    <code>
    {code}
    </code>
    
    Note that the code for fitting the data may be incorrect. It may have used functions does not exists.
    You should revise the code to make it correct.
    
    Here are the only functions from the leeq package that exists and can be used (from the leeq.theory.fits module):
    <fitting_functions>
    {"".join([function_signature_to_xml(x, y) for x, y in _load_fitting_function_docstrings().items()])}
    </fitting_functions>
    
    Note that these function may return `ufloat` objects from the `uncertainties` package. If needed, you can convert
    them to `float` using the `nominal_value` attribute.
    
    If the above functions are not enough, you have to create your own fitting functions.
    
    Return the `data_analysis` function only.
   
    The format of the return should be:
    ```python
    <the code snippet that performs the data analysis>
    ```
    """

    import mllm
    chat = mllm.Chat(prompt, "You are a very smart and helpful coding assistant.")
    res = chat.complete(parse="quotes", cache=False)
    return res
