import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from flask import request
import json

from leeq.utils import setup_logging

logger = setup_logging(__name__)

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
app.title = "LeeQ Experiment Monitor"

server = app.server

# Define the app layout
app.layout = html.Div(
    style={"height": "100%"},
    children=[
        # Banner display
        html.Div(
            [
                html.H2(
                    "LeeQ Experiment Monitor",
                    id="title",
                    className="eight columns",
                    style={"margin-left": "3%"},
                ),
                html.H3("Experiment details:"),
                html.Pre(id='live-json', style={"border": "thin lightgrey solid", "padding": 10}),
                html.H3("Figure display:"),
                dcc.Graph(id='live-figure'),
                dcc.Interval(
                    id='interval-component',
                    interval=1 * 1000,  # in milliseconds
                    n_intervals=0
                ),
            ]
        ),
    ]
)

global experiment_manager
experiment_manager = None
@app.callback(
    Output('live-json', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_experiment_details(n: int):
    """
    Update the experiment details.

    Parameters:
        n (int): The number of intervals.

    Returns:
        str: The experiment details in json format.
    """
    status = experiment_manager.get_live_status()
    return json.dumps(status, indent=4)


@app.callback(
    Output('live-figure', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_figure(n: int):
    """
    Update the figure.

    Parameters:
        n (int): The number of intervals.

    Returns:
        (Any): The figure.
    """

    manager = experiment_manager
    return None


def start_app(**kwargs):
    """
    Start the app.
    
    Parameters:
        **kwargs: The arguments to pass to `app.run_server`.
    """
    global  experiment_manager
    experiment_manager = kwargs.pop('experiment_manager')
    arguments = {
        'jupyter_mode': 'external',
        'host': '0.0.0.0',
        'debug': True
    }
    arguments.update(kwargs)
    app.run_server(**arguments)


def stop_app():
    """
    Stop the app.
    """
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        msg = 'Not running with the Werkzeug Server'
        logger.error(msg)
        raise RuntimeError(msg)
    func()
