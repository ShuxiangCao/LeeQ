import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from flask import request
import json

from leeq.utils import setup_logging

logger = setup_logging(__name__)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "LeeQ Experiment Monitor"

server = app.server

title_div = html.H2(
    "LeeQ Experiment Monitor",
    id="title",
    className="eight columns",
    style={"margin-left": "3%"},
)

interval_dcc = dcc.Interval(
    id='interval-component',
    interval=1 * 1000,  # in milliseconds
    n_intervals=0
)

# Define the app layout
app.layout = dbc.Container(
    [
        html.H1(
            "LeeQ Experiment Monitor",
        ),
        html.Hr(),
        dbc.Progress(
            value=80,
            id="animated-progress",
            animated=True,
            striped=False),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        children=[
                            dcc.Graph(
                                id='live-figure',
                                animate=False),
                        ]),
                    md=8),
                dbc.Col(
                    html.Div(
                        children=[
                            html.Pre(
                                id='live-json',
                                style={
                                    "border": "thin lightgrey solid",
                                    "padding": 10}),
                        ]),
                    md=4),
            ],
            align="center",
        ),
        interval_dcc])

global experiment_manager
experiment_manager = None


def _build_table(dict_data: dict):
    table_body = [html.Tbody([
        html.Tr([html.Td(key), html.Td(repr(val))]) for key, val in dict_data.items()
    ])]

    table = dbc.Table(table_body, bordered=True)


@app.callback(
    [Output('live-json', 'children'),
     Output('animated-progress', 'value'),
     Output('animated-progress', 'label'),
     ],
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

    percentage = status['engine_status']['progress'] * 100
    label = str(status['engine_status']['step_no']) + f'/{percentage:.2f}%'

    del status['engine_status']

    return json.dumps(status, indent=4), percentage, label


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

    return experiment_manager.get_live_plots()


def start_app(**kwargs):
    """
    Start the app.

    Parameters:
        **kwargs: The arguments to pass to `app.run_server`.
    """
    global experiment_manager
    experiment_manager = kwargs.pop('experiment_manager')
    arguments = {
        'jupyter_mode': 'external',
        # 'host': '0.0.0.0',
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
