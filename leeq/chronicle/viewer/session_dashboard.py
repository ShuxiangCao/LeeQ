"""
Chronicle Session Viewer - Live monitoring dashboard for active Chronicle sessions.

This module provides a web-based dashboard for viewing experiments from the current
Chronicle session in real-time. It polls the Chronicle singleton every 5 seconds
to display newly completed experiments with full visualization capabilities.

Features:
    - Live polling of active Chronicle session (5-second intervals)
    - Manual refresh capability as backup
    - Hierarchical tree view of experiments
    - Interactive plot generation and visualization
    - Experiment details and attributes display
    - Resizable panels for optimal layout
    - Graceful handling of no active session

Usage:
    Called from Chronicle.launch_viewer() method:
        from leeq.chronicle import Chronicle
        chronicle = Chronicle()
        chronicle.launch_viewer()

Author: LeeQ Development Team
Version: 2.0.0
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json

# Import shared components from common module
from leeq.chronicle.viewer.common import (
    get_html_template,
    convert_figure_to_plotly,
    create_tree_view_items,
    render_tree_nodes,
    create_experiment_attributes_panel,
    create_header_layout,
    create_plot_display_section
)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css"
])

# Global chronicle instance (similar to experiment_manager pattern in live_dash_app)
chronicle_instance = None

# Use common HTML template with CSS and JavaScript
app.index_string = get_html_template(title="Active Chronicle Session Viewer")

# Main app layout with full-height sidebar design
app.layout = dbc.Container([
    # Header
    create_header_layout(
        "Active Chronicle Session Viewer",
        "Live monitoring of active Chronicle recording session"
    ),
    
    # Main layout with resizable three panels
    html.Div([
        # Left Sidebar - Experiment Selection (Resizable)
        html.Div([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-diagram-3 me-2"),
                    html.Strong("Experiment Selection"),
                    html.Small(" (drag right edge to resize)", className="text-muted ms-2")
                ]),
                dbc.CardBody([
                    # Session controls section
                    html.Div([
                        dbc.Label("Session Controls:", className="fw-bold mb-2"),
                        dcc.Interval(
                            id='session-interval',
                            interval=5000,  # Update every 5 seconds
                            n_intervals=0
                        ),
                        dbc.ButtonGroup([
                            dbc.Button(
                                [
                                    html.I(className="bi bi-arrow-clockwise me-1"),
                                    "Refresh"
                                ],
                                id="manual-refresh",
                                color="primary",
                                size="sm",
                                n_clicks=0
                            )
                        ], className="mb-3 w-100"),
                        html.Div(id="refresh-status", className="text-muted small mb-3"),
                    ], className="file-input-section"),
                    
                    # Experiment tree section
                    html.Div([
                        html.Hr(className="my-3"),
                        html.Div(id="session-tree", className="experiment-tree"),
                    ], className="experiment-section")
                ])
            ], className="sidebar-card"),
            # Resize handle
            html.Div(className="resize-handle", title="Drag to resize sidebar")
        ], className="sidebar-resizable"),
        
        # Right Content Area - Plot & Attributes
        html.Div([
            dbc.Row([
                # Main Content Area - Plot & Controls
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            # Experiment info section
                            dcc.Loading(
                                id="loading-exp-info",
                                type="circle",
                                children=html.Div(id="session-experiment-info", className="mb-3"),
                                color="#0d6efd"
                            ),
                            
                            # Plot controls section
                            dcc.Loading(
                                id="loading-plot-controls",
                                type="dot",
                                children=html.Div(id="plot-controls", className="mb-3"),
                                color="#0d6efd"
                            ),
                            
                            # Plot display section
                            create_plot_display_section(),
                        ])
                    ], className="main-content-card")
                ], width=8, className="main-content-column pe-2"),
                
                # Right Panel - Experiment Attributes
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="bi bi-info-circle me-2"),
                            html.Strong("Experiment Attributes")
                        ]),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-attributes",
                                type="dot",
                                children=html.Div(
                                    id="session-experiment-attributes", 
                                    children=[
                                        dbc.Alert(
                                            [
                                                html.I(className="bi bi-arrow-left me-2"),
                                                "Select an experiment to view its attributes"
                                            ],
                                            color="info",
                                            className="text-center"
                                        )
                                    ],
                                    className=""
                                ),
                                color="#0d6efd"
                            )
                        ])
                    ], className="attributes-card")
                ], width=4, className="attributes-column")
            ], className="g-0 h-100")
        ], className="main-content-resizable")
    ], className="resizable-container content-row"),
    
    # Hidden stores for state management
    dcc.Store(id='selected-experiment-id', storage_type='memory'),
    dcc.Store(id='experiment-data-store', storage_type='memory'),
    dcc.Store(id='plot-functions-store', storage_type='memory')
], fluid=True, className="main-container")


# ============================================================================
# Session-Specific Functions
# ============================================================================

def load_session_experiments():
    """
    Load experiments from the active Chronicle session.
    Returns a tuple of (experiments_list, tree_structure).
    
    This function is specific to live session monitoring.
    """
    global chronicle_instance
    
    if not chronicle_instance:
        return [], {}
    
    try:
        # Check if Chronicle is recording
        if not chronicle_instance.is_recording():
            return [], {}
        
        # Access the active record book
        record_book = chronicle_instance._active_record_book
        if not record_book:
            return [], {}
        
        # Get all experiments from the session
        root = record_book.get_root_entry()
        experiments = []
        
        def traverse_entry(entry, path_prefix=""):
            """Recursively traverse the record tree to collect all experiments."""
            # Get entry path
            entry_path = str(entry.get_path())
            
            # Add this entry as an experiment
            if entry != root:  # Don't include root itself
                experiments.append({
                    'record_id': entry.record_id,
                    'entry_path': entry_path,
                    'timestamp': entry.timestamp,
                    'name': entry.name,
                    'path_parts': entry_path.strip('/').split('/')[1:] if '/' in entry_path else []
                })
            
            # Recursively traverse children
            try:
                for child in entry.children:
                    traverse_entry(child, entry_path)
            except Exception:
                # Some entries may not have accessible children
                pass
        
        # Start traversal from root
        traverse_entry(root)
        
        # Sort by timestamp (older first)
        experiments.sort(key=lambda x: x['timestamp'])
        
        # Build tree structure using the common function
        tree_structure = create_tree_view_items(experiments)
        
        return experiments, tree_structure
        
    except Exception as e:
        print(f"Error loading session experiments: {str(e)}")
        return [], {}


# ============================================================================
# Session-Specific Callbacks
# ============================================================================

@app.callback(
    [Output("session-tree", "children"),
     Output("refresh-status", "children"),
     Output("experiment-data-store", "data")],
    [Input('session-interval', 'n_intervals'),
     Input("manual-refresh", "n_clicks")]
)
def update_session_view(n_intervals, n_clicks):
    """
    Update the session view with current experiments.
    Triggered by both automatic interval and manual refresh button.
    
    This callback handles:
    - Automatic updates every 5 seconds via interval component
    - Manual refresh via button click
    - Error cases: no chronicle instance, no active session, no experiments
    """
    # Determine trigger source
    ctx = callback_context
    trigger_source = "auto" if not ctx.triggered else ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Check if chronicle instance is available
    if not chronicle_instance:
        no_chronicle_alert = dbc.Alert(
            [
                html.I(className="bi bi-exclamation-circle me-2"),
                "No Chronicle instance available. Please launch viewer from Chronicle.launch_viewer()"
            ],
            color="warning"
        )
        return no_chronicle_alert, "Chronicle not initialized", {}
    
    try:
        experiments, tree_structure = load_session_experiments()
        
        if not experiments:
            # Check if recording is active
            if not chronicle_instance.is_recording():
                # No active recording session
                tree_content = dbc.Alert(
                    [
                        html.I(className="bi bi-pause-circle me-2"),
                        "Chronicle is not recording. Start a recording session to see experiments."
                    ],
                    color="warning"
                )
                status_msg = "No active recording session"
            else:
                # Recording active but no experiments yet
                tree_content = dbc.Alert(
                    [
                        html.I(className="bi bi-hourglass-split me-2"),
                        "Session active - waiting for experiments to complete..."
                    ],
                    color="info"
                )
                status_msg = "Session active, no experiments yet"
            
            return tree_content, status_msg, []
        
        # Render the tree view using common function
        tree_content = render_tree_nodes(tree_structure)
        
        # Create informative status message
        if trigger_source == "manual-refresh":
            status_msg = f"✓ Manual refresh - {len(experiments)} experiment(s) found"
        else:
            # Show time since last update
            interval_seconds = (n_intervals * 5) if n_intervals else 0
            minutes = interval_seconds // 60
            seconds = interval_seconds % 60
            time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
            status_msg = f"Auto-refresh active ({time_str}) - {len(experiments)} experiment(s)"
        
        # Store experiment data for other callbacks
        experiment_data = {exp['record_id']: exp for exp in experiments}
        
        return tree_content, status_msg, experiment_data
        
    except AttributeError as e:
        # Handle missing Chronicle attributes
        error_alert = dbc.Alert(
            [
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Chronicle API error: {str(e)}. Check Chronicle version compatibility."
            ],
            color="danger"
        )
        return error_alert, f"API Error: {str(e)}", {}
        
    except Exception as e:
        # Generic error handling
        error_alert = dbc.Alert(
            [
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Unexpected error: {str(e)}"
            ],
            color="danger"
        )
        return error_alert, f"Error: {str(e)}", {}


@app.callback(
    Output("selected-experiment-id", "data"),
    Input({"type": "experiment-btn", "index": ALL}, "n_clicks"),
    State({"type": "experiment-btn", "index": ALL}, "id"),
    prevent_initial_call=True
)
def select_experiment(n_clicks, button_ids):
    """Handle experiment selection from the tree view."""
    if not any(n_clicks):
        raise PreventUpdate
    
    # Find which button was clicked
    for i, clicks in enumerate(n_clicks):
        if clicks:
            return button_ids[i]["index"]
    
    raise PreventUpdate


@app.callback(
    [Output("session-experiment-info", "children"),
     Output("plot-controls", "children"),
     Output("session-experiment-attributes", "children"),
     Output("plot-functions-store", "data")],
    [Input("selected-experiment-id", "data")],
    [State("experiment-data-store", "data")]
)
def update_experiment_display(selected_id, experiment_data):
    """
    Update the experiment info and attributes panels when an experiment is selected.
    
    This callback handles:
    - Displaying experiment details when selected from tree
    - Loading experiment object and attributes from Chronicle
    - Error handling for missing or inaccessible experiments
    """
    if not selected_id:
        # No experiment selected - show helpful prompt
        info_content = dbc.Alert(
            [
                html.I(className="bi bi-arrow-left me-2"),
                "Select an experiment from the tree to view details"
            ],
            color="info",
            className="text-center"
        )
        plot_controls = []
        attributes_content = create_experiment_attributes_panel()
        return info_content, plot_controls, attributes_content, None
    
    # Check chronicle instance is available
    if not chronicle_instance:
        error_content = dbc.Alert(
            [
                html.I(className="bi bi-exclamation-circle me-2"),
                "Chronicle instance not available"
            ],
            color="warning"
        )
        return error_content, [], create_experiment_attributes_panel(error_msg="Chronicle not available"), None
    
    try:
        # Verify Chronicle is still recording
        if not chronicle_instance.is_recording():
            error_content = dbc.Alert(
                [
                    html.I(className="bi bi-stop-circle me-2"),
                    "Chronicle session has ended. Historical data may be unavailable."
                ],
                color="warning"
            )
            return error_content, [], create_experiment_attributes_panel(error_msg="Session ended"), None
        
        # Access the record book and get the specific record
        record_book = chronicle_instance._active_record_book
        if not record_book:
            raise Exception("Record book not accessible")
            
        record = record_book.get_record_by_id(selected_id)
        if not record:
            raise Exception(f"Record {selected_id} not found")
        
        # Attempt to load experiment object
        experiment_obj = None
        obj_error = None
        try:
            experiment_obj = record.get_object()
        except Exception as e:
            obj_error = f"Could not load object: {str(e)}"
        
        # Attempt to load attributes
        attributes = {}
        attr_error = None
        try:
            attributes = record.load_all_attributes()
        except Exception as e:
            attr_error = f"Could not load attributes: {str(e)}"
        
        # Build info panel content
        info_sections = [
            html.H5(f"Experiment: {record.name}"),
            html.Hr()
        ]
        
        # Basic information
        info_sections.extend([
            html.P([
                html.Strong("Record ID: "),
                html.Code(selected_id, style={'fontSize': '0.9em'})
            ]),
            html.P([
                html.Strong("Path: "),
                html.Code(str(record.get_path()), style={'fontSize': '0.9em'})
            ]),
            html.P([
                html.Strong("Timestamp: "),
                f"{record.timestamp:.2f}s from session start"
            ])
        ])
        
        # Add experiment metadata if available from store
        if experiment_data and selected_id in experiment_data:
            exp_meta = experiment_data[selected_id]
            if 'entry_path' in exp_meta:
                info_sections.append(
                    html.P([
                        html.Strong("Entry Path: "),
                        html.Code(exp_meta['entry_path'], style={'fontSize': '0.9em'})
                    ])
                )
        
        info_sections.append(html.Hr())
        
        # Experiment object section
        info_sections.append(html.H6("Experiment Object"))
        if experiment_obj:
            info_sections.append(
                html.Pre(
                    str(experiment_obj),
                    style={'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px'}
                )
            )
        elif obj_error:
            info_sections.append(
                dbc.Alert(obj_error, color="warning", className="small")
            )
        else:
            info_sections.append(
                html.P("Object data not available", className="text-muted")
            )
        
        info_content = html.Div(info_sections)
        
        # Generate plot controls if experiment object is available
        plot_controls = []
        plot_functions_data = None
        
        if experiment_obj:
            try:
                # Get available plot functions
                plot_functions = experiment_obj.get_browser_functions()
                if plot_functions:
                    # Create buttons for each plot function
                    buttons = []
                    for name, method in plot_functions:
                        btn = dbc.Button(
                            name.replace("_", " ").title(),
                            id={"type": "plot-btn", "index": name},
                            color="primary",
                            className="me-2 mb-2",
                            size="sm"
                        )
                        buttons.append(btn)
                    
                    plot_controls = dbc.Card([
                        dbc.CardBody([
                            html.H5([
                                html.I(className="bi bi-graph-up me-2"),
                                f"Available Plots ({len(plot_functions)})"
                            ], className="card-title mb-3 text-primary"),
                            html.Div(buttons)
                        ])
                    ], color="light", className="shadow-sm")
                    
                    # Store plot functions for later use
                    plot_functions_data = {
                        "record_id": selected_id,
                        "functions": {name: True for name, _ in plot_functions}
                    }
                else:
                    plot_controls = dbc.Alert(
                        [
                            html.I(className="bi bi-info-circle me-2"),
                            "No plots available for this experiment type"
                        ],
                        color="info"
                    )
            except AttributeError:
                plot_controls = dbc.Alert(
                    [
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        "This experiment does not support browser functions"
                    ],
                    color="warning"
                )
        
        # Create attributes panel using common function
        if attr_error:
            attributes_content = create_experiment_attributes_panel(error_msg=attr_error)
        elif experiment_obj:
            attributes_content = create_experiment_attributes_panel(
                experiment=experiment_obj,
                record_id=selected_id
            )
        elif attributes:
            attributes_content = create_experiment_attributes_panel(
                experiment=attributes,
                record_id=selected_id
            )
        else:
            attributes_content = create_experiment_attributes_panel(
                error_msg="No attributes available"
            )
        
        return info_content, plot_controls, attributes_content, plot_functions_data
        
    except Exception as e:
        # Handle unexpected errors gracefully
        error_content = dbc.Alert(
            [
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Error loading experiment: {str(e)}"
            ],
            color="danger"
        )
        return error_content, [], create_experiment_attributes_panel(error_msg=str(e)), None


# Callback for displaying plots
@app.callback(
    Output("plot-display", "figure"),
    Input({"type": "plot-btn", "index": ALL}, "n_clicks"),
    [State("plot-functions-store", "data"),
     State("selected-experiment-id", "data")],
    prevent_initial_call=True
)
def display_plot(n_clicks, plot_functions_data, selected_id):
    """
    Generate and display a plot based on the selected browser function.
    
    This callback generates plots from live Chronicle session experiments.
    Unlike the file-based viewer, this accesses experiments directly from memory.
    """
    if not any(n_clicks):
        return go.Figure()
    
    # Check chronicle instance and selected experiment
    if not chronicle_instance or not selected_id:
        return go.Figure().add_annotation(
            text="No experiment selected or Chronicle not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Get which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        return go.Figure()
    
    # Extract the button ID that was clicked
    button_id = ctx.triggered[0]['prop_id']
    if '"index":' not in button_id:
        return go.Figure()
    
    # Parse the method name from the button ID
    button_dict = json.loads(button_id.split('.')[0])
    method_name = button_dict['index']
    
    try:
        # Verify Chronicle is still recording
        if not chronicle_instance.is_recording():
            return go.Figure().add_annotation(
                text="Chronicle session has ended",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Access the record book and get the specific record
        record_book = chronicle_instance._active_record_book
        if not record_book:
            raise Exception("Record book not accessible")
            
        record = record_book.get_record_by_id(selected_id)
        if not record:
            raise Exception(f"Record {selected_id} not found")
        
        # Load the experiment object
        experiment = record.get_object()
        
        # Get plot functions and find the selected one
        plot_functions = experiment.get_browser_functions()
        
        for name, method in plot_functions:
            if name == method_name:
                try:
                    # Generate the plot
                    fig = method()
                    # Convert figure to Plotly format using common function
                    plotly_fig = convert_figure_to_plotly(fig)
                    return plotly_fig
                except Exception as plot_error:
                    return go.Figure().add_annotation(
                        text=f"Error generating plot: {str(plot_error)[:200]}",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
        
        return go.Figure().add_annotation(
            text=f"Plot method '{method_name}' not found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error: {str(e)[:200]}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )


def start_viewer(**kwargs):
    """
    Start the session viewer dashboard with a Chronicle instance.
    
    Args:
        chronicle_instance: The Chronicle singleton instance
        debug: Whether to run in debug mode (default: True)
        port: Port to run the server on (default: 8051)
        **kwargs: Additional arguments passed to app.run()
    """
    global chronicle_instance
    
    # Extract Chronicle instance from kwargs
    chronicle_instance = kwargs.pop('chronicle_instance', None)
    
    if not chronicle_instance:
        raise ValueError("Chronicle instance must be provided to start_viewer()")
    
    # Set default arguments
    default_args = {
        'debug': True,
        'port': 8051,
        'host': '127.0.0.1'
    }
    default_args.update(kwargs)
    
    print(f"Starting Chronicle Session Viewer on http://{default_args['host']}:{default_args['port']}")
    print("Monitoring active Chronicle session for completed experiments...")
    print("Press Ctrl+C to stop the viewer")
    
    # Start the Dash server
    app.run(**default_args)


# For testing import
if __name__ == "__main__":
    print("Session dashboard module loaded. Use Chronicle.launch_viewer() to start.")