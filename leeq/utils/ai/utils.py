from typing import Union
import matplotlib.pyplot as plt
from PIL import Image
import io
import plotly.graph_objects as go
import os


def matplotlib_plotly_to_pil(fig: Union[go.Figure, plt.Figure]):
    """
    Convert a Matplotlib or Plotly figure to a PIL image.

    Parameters:
        fig (Union[go.Figure, plt.Figure]): The Matplotlib or Plotly figure.

    Returns:
        Image: The PIL image.
    """

    # Save the Matplotlib figure to a BytesIO object

    buf = io.BytesIO()
    if isinstance(fig, go.Figure):
        if os.name == 'nt':
            engine = "orca"
        else:
            engine = "kaleido"
        fig.write_image(buf,format='png',engine=engine)
    elif isinstance(fig, plt.Figure):
        fig.savefig(buf, format='png')
    else:
        raise ValueError(f"The input must be a Matplotlib or Plotly figure. Got {type(fig)}.")

    buf.seek(0)

    # Open the image with PIL
    image = Image.open(buf)
    return image
