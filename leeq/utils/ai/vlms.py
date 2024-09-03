import inspect
import os
import re
from functools import partial

from mllm import Chat
from plotly import graph_objects as go
from matplotlib import pyplot as plt
from IPython.display import display


def visual_analyze_prompt(prompt: str):
    """
    Decorator function for the functions that used to visualize data of the class.
    It is used to register the prompt to analyze the data.

    Parameters:
        prompt (str): The prompt to be registered.

    Returns:
        Any: The return value of the function.
    """
    calling_path = inspect.stack()[1].filename

    def inner_func(func):
        """
        Decorator function for the functions that used to visualize data of the class.
        It is used to register the prompt to analyze the data.
        The function must be a method of a LoggableObject.

        Parameters:
            func (function): The function to be registered.

        Returns:
            Any: The same function.
        """
        func._visual_prompt = prompt
        func._file_path = calling_path
        func.ai_inspect = partial(_fast_visual_inspection, func)

        return func

    return inner_func


def get_visual_analyze_prompt(func):
    """
    Get the prompt to analyze the data.

    Parameters:
        func (function): The function to get the prompt.

    Returns:
        str: The prompt to analyze the data.
    """
    return getattr(func, '_visual_prompt', None)


def has_visual_analyze_prompt(func):
    """
    Check if the function has a prompt to analyze the data.

    Parameters:
        func (function): The function to check.

    Returns:
        bool: True if the function has a prompt to analyze the data.
    """
    return hasattr(func, '_visual_prompt')


import matplotlib.pyplot as plt
from PIL import Image


def visual_inspection(image: "Image", prompt: str, func_file_path, rescale=0.5, **kwargs) -> dict:
    """
    Ask a question about the data shape. For example, the number of peaks, or whether the data is nearly periodic.
    The answer is either True or False.
    You should use it when you need to analyze the data.

    Parameters:
        image (Image): The image to analyze.
        prompt (str): The prompt to ask.
        rescale (float): Optional. The rescale factor for the image.
        kwargs (dict): Optional. The keyword arguments to pass to the function.

    Returns:
        dict: The result of the analysis.
    """
    from leeq.utils.ai.utils import matplotlib_plotly_to_pil

    if not isinstance(image, Image.Image):
        assert isinstance(image, plt.Figure) or isinstance(image,
                                                           go.Figure), "The image must be a PIL image or a Matplotlib or Plotly figure."
        image = matplotlib_plotly_to_pil(image)

    from mllm import Chat
    chat = prepare_visual_inspection_chat(prompt, func_file_path)
    chat.system_message = ("You are a helpful visual assistant that able to provide analysis on images or plots. "
                       "Please return your message in a json format with keys analysis(str, single paragraph) "
                       "and 'success'(boolean)")
    chat.add_user_message("Here is the input image:")
    chat.add_image_message(image)
    res = chat.complete(parse="dict", **kwargs)

    return res

def prepare_visual_inspection_chat(raw_prompt, func_file_path):
    # extract all the Image("...") or Image('...') patterns and break the raw_prompt into parts
    # for example "123 Image("dasda") 123" should be broken into ['123 ', 'Image("dasda")', ' 123']
    func_dir = os.path.dirname(func_file_path)
    image_pattern = re.compile(r'Image\(["\'].*?["\']\)')
    image_matches = image_pattern.findall(raw_prompt)
    parts = image_pattern.split(raw_prompt)
    path_pattern = re.compile(r'Image\(["\'](.*?)["\']\)')

    merged_parts = []
    for part, image_match in zip(parts, image_matches):
        merged_parts.append(part)
        #
        path = path_pattern.search(image_match).group(1)
        merged_parts.append({"src": os.path.join(func_dir, path), "type": "image"})
    if len(parts) > len(image_matches):
        merged_parts.append(parts[-1])

    chat = Chat()
    for part in merged_parts:
        if isinstance(part, str):
            trimmed = part.strip()
            if len(trimmed) > 0:
                chat.add_user_message(trimmed)
        else:
            chat.add_image_message(part["src"])

    return chat


def _fast_visual_inspection(func, image=None, prompt=None, func_kwargs=None, llm_kwargs=None):
    """
    Fast version of visual inspection. It will not ask the user for the prompt.

    Parameters:
        func (function): The function to analyze.
        prompt (str): Optional. The prompt to ask.
        image (Image): Optional. The image to analyze.
        func_kwargs (dict): Optional. The keyword arguments to pass to the function.
        llm_kwargs (dict): Optional. The keyword arguments to pass to the function.

    Returns:
        dict: The result of the analysis.
    """

    if func_kwargs is None:
        func_kwargs = {}
    if llm_kwargs is None:
        llm_kwargs = {}

    if prompt is None:
        prompt = get_visual_analyze_prompt(func)
        if prompt is None:
            raise ValueError(f"No default prompt for function {func.__qualname__}.")

    if image is None:
        if hasattr(func, '_image'):
            image = func._image
        else:
            image = func(**func_kwargs)
            func.__dict__['_image'] = image

    res = visual_inspection(image, prompt, func._file_path, **llm_kwargs)

    func.__dict__['_ai_inspect_result'] = res

    return res