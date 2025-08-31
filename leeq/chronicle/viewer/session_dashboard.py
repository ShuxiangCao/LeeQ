"""
Chronicle Session Viewer - Live monitoring dashboard for active Chronicle sessions.

This module provides a web-based dashboard for viewing experiments from the current
Chronicle session in real-time. It polls the Chronicle singleton every 5 seconds
to display newly completed experiments.

Features:
    - Live polling of active Chronicle session (5-second intervals)
    - Manual refresh capability as backup
    - Hierarchical tree view of experiments
    - Experiment details and attributes display
    - Graceful handling of no active session

Usage:
    Called from Chronicle.launch_viewer() method:
        from leeq.chronicle import Chronicle
        chronicle = Chronicle()
        chronicle.launch_viewer()

Author: LeeQ Development Team
Version: 1.0.0
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from leeq.chronicle.viewer.dashboard import (
    create_tree_view_items,
    render_tree_nodes,
    create_experiment_attributes_panel
)
import json

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css"
])

# Global chronicle instance (similar to experiment_manager pattern in live_dash_app)
chronicle_instance = None

# Custom CSS for layout (reuse from existing dashboard)
custom_css = """
<style>
body {
    height: 100vh;
    overflow: hidden;
}

.main-container {
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.content-row {
    flex: 1;
    min-height: 0;
}

.sidebar-column {
    height: 100%;
    display: flex;
    flex-direction: column;
}

.sidebar-card {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
}

.sidebar-card .card-body {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
}

.main-content-column {
    height: 100%;
    display: flex;
    flex-direction: column;
}

.main-content-card {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
}

.main-content-card .card-body {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
}

.attributes-column {
    height: 100%;
    display: flex;
    flex-direction: column;
}

.attributes-card {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
}

.attributes-card .card-body {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
}

.experiment-tree {
    overflow-y: auto !important;
}

.experiment-tree details summary {
    cursor: pointer;
    padding: 4px 0;
}

.experiment-tree details summary:hover {
    background-color: rgba(0, 123, 255, 0.1);
    border-radius: 3px;
}

.experiment-tree .btn-outline-primary {
    border-width: 1px;
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
}

.experiment-tree .btn-outline-primary:hover {
    background-color: rgba(0, 123, 255, 0.1);
    border-color: #0d6efd;
}
</style>
"""

# Set custom HTML template with CSS
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        """ + custom_css + """
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Application layout
app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Active Chronicle Session Viewer", className="mb-3"),
                html.Hr()
            ])
        ]),
        
        # Auto-refresh interval component (5-second polling)
        dcc.Interval(
            id='session-interval',
            interval=5000,  # 5 seconds in milliseconds
            n_intervals=0
        ),
        
        # Manual refresh button and status
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise me-2"), "Manual Refresh"],
                        id="manual-refresh",
                        color="primary",
                        className="mb-3"
                    )
                ]),
                html.Div(id="refresh-status", className="text-muted small mb-2")
            ])
        ]),
        
        # Main content area with three columns
        dbc.Row([
            # Left panel: Experiment tree view
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-diagram-3 me-2"),
                        "Experiments"
                    ]),
                    dbc.CardBody([
                        html.Div(id="session-tree", className="experiment-tree")
                    ])
                ], className="sidebar-card h-100")
            ], width=4, className="sidebar-column"),
            
            # Middle panel: Experiment information
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-info-circle me-2"),
                        "Experiment Information"
                    ]),
                    dbc.CardBody([
                        html.Div(id="session-experiment-info")
                    ])
                ], className="main-content-card h-100")
            ], width=5, className="main-content-column"),
            
            # Right panel: Experiment attributes
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-list-ul me-2"),
                        "Experiment Attributes"
                    ]),
                    dbc.CardBody([
                        html.Div(id="session-experiment-attributes")
                    ])
                ], className="attributes-card h-100")
            ], width=3, className="attributes-column")
        ], className="content-row h-100"),
        
        # Hidden stores for state management
        dcc.Store(id='selected-experiment-id', storage_type='memory'),
        dcc.Store(id='experiment-data-store', storage_type='memory')
    ], fluid=True, className="main-container")


def load_session_experiments():
    """
    Load experiments from the active Chronicle session.
    Returns a tuple of (experiments_list, tree_structure).
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
        
        # Build tree structure using the existing function
        tree_structure = create_tree_view_items(experiments)
        
        return experiments, tree_structure
        
    except Exception as e:
        print(f"Error loading session experiments: {str(e)}")
        return [], {}


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
        
        # Render the tree view
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
     Output("session-experiment-attributes", "children")],
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
        attributes_content = create_experiment_attributes_panel()
        return info_content, attributes_content
    
    # Check chronicle instance is available
    if not chronicle_instance:
        error_content = dbc.Alert(
            [
                html.I(className="bi bi-exclamation-circle me-2"),
                "Chronicle instance not available"
            ],
            color="warning"
        )
        return error_content, create_experiment_attributes_panel(error_msg="Chronicle not available")
    
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
            return error_content, create_experiment_attributes_panel(error_msg="Session ended")
        
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
        
        # Create attributes panel
        if attr_error:
            attributes_content = create_experiment_attributes_panel(error_msg=attr_error)
        elif attributes:
            attributes_content = create_experiment_attributes_panel(
                experiment=attributes,
                record_id=selected_id
            )
        else:
            attributes_content = create_experiment_attributes_panel(
                error_msg="No attributes available"
            )
        
        return info_content, attributes_content
        
    except Exception as e:
        # Handle unexpected errors gracefully
        error_content = dbc.Alert(
            [
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Error loading experiment: {str(e)}"
            ],
            color="danger"
        )
        return error_content, create_experiment_attributes_panel(error_msg=str(e))


def start_viewer(**kwargs):
    """
    Start the session viewer dashboard with a Chronicle instance.
    
    Args:
        chronicle_instance: The Chronicle singleton instance
        debug: Whether to run in debug mode (default: True)
        port: Port to run the server on (default: 8051)
        **kwargs: Additional arguments passed to app.run_server()
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
    app.run_server(**default_args)


# For testing import
if __name__ == "__main__":
    print("Session dashboard module loaded. Use Chronicle.launch_viewer() to start.")