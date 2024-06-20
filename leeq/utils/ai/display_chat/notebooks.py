from IPython.display import display, HTML
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


def dict_to_html(data_dict) -> str:
    """
    Converts a dictionary into an HTML string with keys in bold and values formatted with repr(),
    styled to fit within a chat bubble.

    Args:
    data_dict (dict): The dictionary to convert to HTML.

    Returns:
    str: The HTML formatted string.
    """
    html_parts = []
    for key, value in data_dict.items():
        formatted_value = repr(value)
        html_parts.append(f"<strong>{key}</strong>: {formatted_value}<br>")
    return ''.join(html_parts)

def code_to_html(code:str):
    """
    Convert Python code to HTML using Pygments.

    Args:
    code (str): Python code to convert.

    Returns:
    str: HTML formatted string representing the Python code.
    """
    formatter = HtmlFormatter()
    formatted_code = highlight(code, PythonLexer(), formatter)
    return f"""
        <div style="background: #f8f8f8; padding: 10px; border-radius: 5px; overflow: auto; margin-top: 8px;">
            {formatted_code}
        </div>
"""


def display_chat(agent_name:str, background_color:str, content:str):
    """
    Display a chat message formatted with the given parameters.

    Args:
    agent_name (str): Name of the agent speaking.
    background_color (str): Background color for the chat bubble.
    content (str): Content of the message, which may include plain text or HTML.

    Returns:
    str: HTML formatted string representing the chat message.
    """
    html = f'''
    <p style="background-color: {background_color}; padding: 20px; border-radius: 8px; color: #333;">
        <strong>{agent_name}:</strong> {content}
    </p>
    '''
    display(HTML(html))