import time
import uuid
from IPython.display import display, HTML, Javascript

def show_spinner(text: str = "Processing..."):
    """
    Display a spinner to indicate that the notebook is processing.

    Parameters
    ----------
    text: str
        The text to display next to the spinner.

    Returns
    -------
    spinner_id: str
        The unique identifier of the spinner.
    """
    spinner_id = str(uuid.uuid4())
    spinner_html = HTML(f"""
    <div id="{spinner_id}" style="font-size:16px;">
        <i class="fa fa-spinner fa-spin"></i> {text}
    </div>
    """)
    display(spinner_html)
    return spinner_id  # Return the UUID to the caller for later use

def hide_spinner(spinner_id):
    """
    Hide the spinner with the given unique identifier.

    Parameters
    ----------
    spinner_id: str
        The unique identifier of the spinner to hide.
    """
    hide_spinner_js = Javascript(f"""
    document.getElementById('{spinner_id}').remove();
    """)
    display(hide_spinner_js)