"""
Common components for Chronicle viewer dashboards.
Shared between file-based (dashboard.py) and live session (session_dashboard.py) viewers.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.tools
import base64
import io
import inspect


# ============================================================================
# CSS and JavaScript Assets
# ============================================================================

def get_custom_css():
    """Return the custom CSS for dashboard styling."""
    return """
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
    display: flex;
    flex-direction: column;
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

.sidebar-card .card-body {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
}

.file-input-section {
    flex-shrink: 0;
}

.experiment-section {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
}

.experiment-tree {
    flex: 1;
    min-height: 0;
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

.main-content {
    height: 100%;
    overflow-y: auto;
}

/* Resizable panel styles */
.resizable-container {
    display: flex;
    height: 100%;
    position: relative;
}

.sidebar-resizable {
    min-width: 200px;
    max-width: 60%;
    width: 25%;
    position: relative;
    flex-shrink: 0;
}

.resize-handle {
    width: 6px;
    background: #ced4da;
    cursor: col-resize;
    position: absolute;
    top: 0;
    right: -3px;
    bottom: 0;
    z-index: 100;
    transition: all 0.2s;
    border-radius: 0 3px 3px 0;
}

.resize-handle:hover,
.resize-handle.resizing {
    background: #0d6efd;
    width: 8px;
    right: -4px;
}

.resize-handle:hover::after,
.resize-handle.resizing::after {
    content: '';
    position: absolute;
    left: -3px;
    right: -3px;
    top: 0;
    bottom: 0;
    background: rgba(13, 110, 253, 0.15);
    border-radius: 3px;
}

.main-content-resizable {
    flex: 1;
    min-width: 300px;
    padding-left: 10px;
}

/* Prevent text selection during resize */
.resizing * {
    user-select: none !important;
    pointer-events: none !important;
}

/* Resize cursor for the entire document during resize */
body.resizing {
    cursor: col-resize !important;
}
</style>
"""


def get_resize_javascript():
    """Return the JavaScript for resizable panels."""
    return """
<script>
// Resizable panel functionality with better Dash compatibility
function initializeResize() {
    const resizeHandle = document.querySelector('.resize-handle');
    const sidebar = document.querySelector('.sidebar-resizable');
    const container = document.querySelector('.resizable-container');

    if (!resizeHandle || !sidebar || !container) {
        // Retry after a short delay if elements aren't found
        setTimeout(initializeResize, 100);
        return;
    }

    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    // Remove any existing event listeners
    resizeHandle.onmousedown = null;

    resizeHandle.onmousedown = function(e) {
        isResizing = true;
        startX = e.clientX;
        startWidth = parseInt(document.defaultView.getComputedStyle(sidebar).width, 10);

        document.body.classList.add('resizing');
        resizeHandle.classList.add('resizing');

        e.preventDefault();
        e.stopPropagation();
    };

    document.onmousemove = function(e) {
        if (!isResizing) return;

        const width = startWidth + e.clientX - startX;
        const containerWidth = container.offsetWidth;
        const minWidth = 200;
        const maxWidth = containerWidth * 0.6;

        if (width >= minWidth && width <= maxWidth) {
            const percentage = (width / containerWidth) * 100;
            sidebar.style.width = percentage + '%';
        }

        e.preventDefault();
    };

    document.onmouseup = function() {
        if (isResizing) {
            isResizing = false;
            document.body.classList.remove('resizing');
            resizeHandle.classList.remove('resizing');
        }
    };

    // Handle window resize
    window.onresize = function() {
        const containerWidth = container.offsetWidth;
        const currentWidth = sidebar.offsetWidth;
        const percentage = (currentWidth / containerWidth) * 100;

        if (percentage > 60) {
            sidebar.style.width = '60%';
        } else if (percentage < 15) {
            sidebar.style.width = '200px';
        }
    };

    console.log('Resize functionality initialized');
}

// Initialize immediately and also after Dash renders
document.addEventListener('DOMContentLoaded', initializeResize);
setTimeout(initializeResize, 500);
setTimeout(initializeResize, 1000);
</script>
"""


def get_html_template(title="Chronicle Viewer", app_name=None):
    """
    Generate the HTML template with custom CSS and JS.

    Args:
        title: Page title for the browser tab
        app_name: Application name (unused but kept for compatibility)
    """
    custom_css = get_custom_css()
    resize_js = get_resize_javascript()

    return f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{title}</title>
        {{%favicon%}}
        {{%css%}}
        {custom_css}
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
        {resize_js}
    </body>
</html>"""


# ============================================================================
# Figure Conversion Utilities
# ============================================================================

def convert_figure_to_plotly(fig):
    """
    Convert various figure types to Plotly figures for display in Dash.

    Supports:
    - Plotly figures (returned as-is)
    - Matplotlib figures (converted to Plotly)
    - Other types (error message)

    Args:
        fig: Figure object of various types

    Returns:
        go.Figure: Plotly figure ready for display
    """
    if isinstance(fig, go.Figure):
        # Already a Plotly figure
        return fig

    # Check if it's a matplotlib figure
    try:
        import matplotlib.figure
        if isinstance(fig, matplotlib.figure.Figure):
            # Convert matplotlib figure to Plotly
            try:
                # Try using plotly.tools.mpl_to_plotly (if available)
                if hasattr(plotly.tools, 'mpl_to_plotly'):
                    plotly_fig = plotly.tools.mpl_to_plotly(fig)
                    return plotly_fig
                else:
                    # Fallback: save matplotlib figure as image and embed
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                    img_buffer.seek(0)
                    img_b64 = base64.b64encode(img_buffer.read()).decode()

                    # Create Plotly figure with embedded image
                    plotly_fig = go.Figure()
                    plotly_fig.add_layout_image(
                        {
                            "source": f"data:image/png;base64,{img_b64}",
                            "xref": "paper", "yref": "paper",
                            "x": 0, "y": 1, "sizex": 1, "sizey": 1,
                            "xanchor": "left", "yanchor": "top",
                            "layer": "below"
                        }
                    )
                    plotly_fig.update_layout(
                        xaxis={"visible": False},
                        yaxis={"visible": False},
                        margin={"l": 0, "r": 0, "t": 0, "b": 0},
                        showlegend=False
                    )

                    # Close matplotlib figure to free memory
                    import matplotlib.pyplot as plt
                    plt.close(fig)

                    return plotly_fig
            except Exception as e:
                # Close matplotlib figure and return error
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except:
                    pass
                return go.Figure().add_annotation(
                    text=f"Error converting matplotlib figure: {str(e)[:100]}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
    except ImportError:
        pass

    # Unsupported figure type
    return go.Figure().add_annotation(
        text=f"Unsupported figure type: {type(fig).__name__}. Supported: Plotly Figure, matplotlib Figure",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )


# ============================================================================
# Tree View Components
# ============================================================================

def create_tree_view_items(experiments):
    """
    Create tree view items from the flat experiment list.
    Organizes experiments in a hierarchical structure based on entry paths.
    Also identifies which parent nodes have their own experiment records.
    """
    if not experiments:
        return []

    # Group experiments by their hierarchy
    tree_nodes = {}
    # Track parent experiments that have their own records
    parent_experiments = {}

    # First pass: identify all parent experiments
    for exp in experiments:
        # Check if this experiment is a parent experiment (could have children)
        # by looking for other experiments that start with this path
        is_potential_parent = any(
            other_exp['entry_path'].startswith(exp['entry_path'] + '/')
            for other_exp in experiments if other_exp != exp
        )

        if is_potential_parent:
            path_parts = exp['path_parts']
            full_name = path_parts[-1].replace('.run', '')

            # Extract clean name (remove number prefix)
            if '-' in full_name and full_name.split('-')[0].isdigit():
                clean_name = '-'.join(full_name.split('-')[1:])
            else:
                clean_name = full_name

            parent_key = exp['entry_path']
            parent_experiments[parent_key] = {
                'record_id': exp['record_id'],
                'display_name': f"{clean_name} ({exp['record_id']})",
                'timestamp': exp['timestamp'],
                'clean_name': clean_name,
                'full_path': '/'.join(path_parts)
            }

    # Second pass: build the tree structure
    for exp in experiments:
        path_parts = exp['path_parts']
        record_id = exp['record_id']
        timestamp = exp['timestamp']

        if not path_parts:
            continue

        # Create display name from the experiment path
        full_name = path_parts[-1].replace('.run', '')

        # Extract clean name (remove number prefix)
        if '-' in full_name and full_name.split('-')[0].isdigit():
            clean_name = '-'.join(full_name.split('-')[1:])
        else:
            clean_name = full_name

        # Build the tree path
        current_level = tree_nodes
        for i, part in enumerate(path_parts[:-1]):  # All parts except the last
            part_clean = part.replace('.run', '')
            if '-' in part_clean and part_clean.split('-')[0].isdigit():
                parent_name = '-'.join(part_clean.split('-')[1:])
            else:
                parent_name = part_clean

            if parent_name not in current_level:
                # Check if this parent has its own experiment record
                parent_path = '/root/' + '/'.join(path_parts[:i+1])
                parent_exp_data = parent_experiments.get(parent_path)

                current_level[parent_name] = {
                    'children': {},
                    'experiments': [],
                    'is_parent': True,
                    'parent_experiment': parent_exp_data  # Store parent experiment data if exists
                }
            else:
                # Update existing parent node with experiment data if we found some
                parent_path = '/root/' + '/'.join(path_parts[:i+1])
                parent_exp_data = parent_experiments.get(parent_path)
                if parent_exp_data and not current_level[parent_name]['parent_experiment']:
                    current_level[parent_name]['parent_experiment'] = parent_exp_data

            current_level = current_level[parent_name]['children']

        # Add the experiment to the appropriate level
        if clean_name not in current_level:
            current_level[clean_name] = {
                'children': {},
                'experiments': [],
                'is_parent': False,
                'parent_experiment': None
            }

        current_level[clean_name]['experiments'].append({
            'record_id': record_id,
            'display_name': f"{clean_name} ({record_id})",
            'timestamp': timestamp,
            'full_path': '/'.join(path_parts)
        })

    # Third pass: Update is_parent flag based on whether there are multiple experiments
    def update_is_parent(nodes):
        for name, node in nodes.items():
            # Set is_parent to True if:
            # 1. It has children (nested structure), OR
            # 2. It has multiple experiments under this type
            if node['children'] or len(node['experiments']) > 1:
                node['is_parent'] = True

            # Recursively update children
            if node['children']:
                update_is_parent(node['children'])

    update_is_parent(tree_nodes)

    return tree_nodes


def render_tree_nodes(tree_nodes, level=0):
    """
    Recursively render tree nodes as HTML elements.
    """
    items = []

    for name, node in tree_nodes.items():
        indent_style = {"marginLeft": f"{level * 20}px"}

        if node['is_parent'] and node['children']:
            # Parent node with children
            # Check if this parent has its own experiment data
            parent_exp = node.get('parent_experiment')

            # Create the summary content with optional SELECT button
            if parent_exp:
                # Parent has its own experiment - add SELECT button
                summary_content = html.Div([
                    html.Strong(name, className="me-2"),
                    dbc.Button(
                        "SELECT",
                        id={"type": "experiment-btn", "index": parent_exp['record_id']},
                        color="primary",
                        size="sm",
                        className="ms-auto",
                        style={"fontSize": "0.7rem", "padding": "2px 8px"}
                    )
                ], className="d-flex align-items-center w-100")
            else:
                # Parent is just organizational - no SELECT button
                summary_content = html.Strong(name)

            items.append(
                html.Details([
                    html.Summary(
                        summary_content,
                        style=indent_style,
                        className="mb-1"
                    ),
                    html.Div([
                        *render_tree_nodes(node['children'], level + 1),
                        # Add direct experiments of this parent
                        *[dbc.Button(
                            exp['display_name'],
                            id={"type": "experiment-btn", "index": exp['record_id']},
                            color="outline-primary",
                            size="sm",
                            className="mb-1 w-100 text-start",
                            style={"marginLeft": f"{(level + 1) * 15}px", "fontSize": "0.85rem"}
                        ) for exp in node['experiments']]
                    ])
                ], open=True, className="mb-2")
            )
        else:
            # Leaf experiments
            for exp in node['experiments']:
                items.append(
                    dbc.Button(
                        exp['display_name'],
                        id={"type": "experiment-btn", "index": exp['record_id']},
                        color="outline-primary",
                        size="sm",
                        className="mb-1 w-100 text-start",
                        style={**indent_style, "fontSize": "0.85rem"}
                    )
                )

    return items


# ============================================================================
# Experiment Attributes Panel
# ============================================================================

def format_argument_value(arg):
    """Helper function to format argument values for display."""
    if isinstance(arg, (int, float, bool)):
        return str(arg)
    elif isinstance(arg, str):
        return f'"{arg}"' if len(arg) < 50 else f'"{arg[:47]}..."'
    elif isinstance(arg, (list, tuple)):
        return f"{type(arg).__name__}[{len(arg)}]"
    elif isinstance(arg, dict):
        return f"dict[{len(arg)} keys]"
    elif hasattr(arg, '__name__'):
        return f"{type(arg).__name__}: {arg.__name__}"
    else:
        return str(type(arg).__name__)


def get_run_function_params(experiment):
    """Extract parameter names from the experiment's run function."""
    try:
        # Look for run method
        if hasattr(experiment, 'run') and callable(experiment.run):
            sig = inspect.signature(experiment.run)
            params = list(sig.parameters.keys())
            # Remove 'self' if present
            if params and params[0] == 'self':
                params = params[1:]
            return params
    except Exception:
        pass
    return []


def create_experiment_attributes_panel(experiment=None, record_id=None, error_msg=None):
    """Create the experiment attributes panel content."""
    if error_msg:
        return dbc.Alert(
            [
                html.I(className="bi bi-exclamation-triangle me-2"),
                f"Error loading attributes: {error_msg}"
            ],
            color="warning"
        )

    if not experiment:
        return dbc.Alert(
            [
                html.I(className="bi bi-arrow-left me-2"),
                "Select an experiment to view its attributes"
            ],
            color="info",
            className="text-center"
        )

    # Extract experiment attributes
    attributes = []

    # Basic information
    attributes.append(
        html.Div([
            html.H6("Basic Information", className="text-primary mb-2 border-bottom pb-1"),
            html.P([html.Strong("Type: "), html.Span(type(experiment).__name__, className="font-monospace")], className="mb-1 small"),
            html.P([html.Strong("Record ID: "), html.Span(record_id, className="font-monospace")], className="mb-1 small"),
        ], className="mb-3")
    )

    # Get parameter names from run function
    run_param_names = get_run_function_params(experiment)

    # Display run_args
    if hasattr(experiment, 'run_args') and experiment.run_args is not None:
        try:
            run_args = experiment.run_args
            run_args_items = []

            if isinstance(run_args, tuple):
                # Unpack tuple elements with parameter names if available
                for i, arg in enumerate(run_args):
                    display_value = format_argument_value(arg)

                    # Use parameter name if available, otherwise use index
                    if i < len(run_param_names):
                        param_label = f"{run_param_names[i]}:"
                    else:
                        param_label = f"Arg {i}:"

                    run_args_items.append(
                        html.P([
                            html.Strong(param_label + " "),
                            html.Span(display_value, className="text-muted small font-monospace")
                        ], className="mb-1 small", style={"wordBreak": "break-word"})
                    )
            else:
                # Handle non-tuple run_args
                display_value = str(run_args)[:100] + "..." if len(str(run_args)) > 100 else str(run_args)
                run_args_items.append(
                    html.P([
                        html.Strong("Value: "),
                        html.Span(display_value, className="text-muted small font-monospace")
                    ], className="mb-1 small", style={"wordBreak": "break-word"})
                )

            if run_args_items:
                attributes.append(
                    html.Div([
                        html.H6("Run Arguments", className="text-primary mb-2 border-bottom pb-1"),
                        html.Small(f"Type: {type(run_args).__name__}" + (f", Length: {len(run_args)}" if hasattr(run_args, '__len__') else ""),
                                  className="text-muted d-block mb-2"),
                        html.Div(run_args_items)
                    ], className="mb-3")
                )
        except Exception as e:
            # If there's an error processing run_args, show a brief error message
            attributes.append(
                html.Div([
                    html.H6("Run Arguments", className="text-primary mb-2 border-bottom pb-1"),
                    html.P(f"Error processing run_args: {str(e)[:50]}...", className="text-muted small")
                ], className="mb-3")
            )

    # Display run_kwargs
    if hasattr(experiment, 'run_kwargs') and experiment.run_kwargs is not None:
        try:
            run_kwargs = experiment.run_kwargs
            run_kwargs_items = []

            if isinstance(run_kwargs, dict):
                # Display each key-value pair
                for key, value in run_kwargs.items():
                    display_value = format_argument_value(value)

                    run_kwargs_items.append(
                        html.P([
                            html.Strong(f"{key}: "),
                            html.Span(display_value, className="text-muted small font-monospace")
                        ], className="mb-1 small", style={"wordBreak": "break-word"})
                    )
            else:
                # Handle non-dict run_kwargs (unusual case)
                display_value = str(run_kwargs)[:100] + "..." if len(str(run_kwargs)) > 100 else str(run_kwargs)
                run_kwargs_items.append(
                    html.P([
                        html.Strong("Value: "),
                        html.Span(display_value, className="text-muted small font-monospace")
                    ], className="mb-1 small", style={"wordBreak": "break-word"})
                )

            if run_kwargs_items:
                attributes.append(
                    html.Div([
                        html.H6("Run Keyword Arguments", className="text-primary mb-2 border-bottom pb-1"),
                        html.Small(f"Type: {type(run_kwargs).__name__}" + (f", Length: {len(run_kwargs)}" if hasattr(run_kwargs, '__len__') else ""),
                                  className="text-muted d-block mb-2"),
                        html.Div(run_kwargs_items)
                    ], className="mb-3")
                )
        except Exception as e:
            # If there's an error processing run_kwargs, show a brief error message
            attributes.append(
                html.Div([
                    html.H6("Run Keyword Arguments", className="text-primary mb-2 border-bottom pb-1"),
                    html.P(f"Error processing run_kwargs: {str(e)[:50]}...", className="text-muted small")
                ], className="mb-3")
            )

    # Experiment-specific attributes
    exp_attrs = []
    if hasattr(experiment, '__dict__'):
        for attr_name in sorted(dir(experiment)):
            if not attr_name.startswith('_') and not callable(getattr(experiment, attr_name)):
                try:
                    attr_value = getattr(experiment, attr_name)
                    if attr_value is not None and str(attr_value) != "":
                        # Format the attribute value
                        if isinstance(attr_value, (int, float)):
                            display_value = str(attr_value)
                        elif isinstance(attr_value, str):
                            display_value = attr_value[:100] + "..." if len(attr_value) > 100 else attr_value
                        elif hasattr(attr_value, '__len__') and not isinstance(attr_value, str):
                            display_value = f"{type(attr_value).__name__} (length: {len(attr_value)})"
                        else:
                            display_value = str(type(attr_value).__name__)

                        exp_attrs.append(
                            html.P([
                                html.Strong(f"{attr_name}: "),
                                html.Span(display_value, className="text-muted small")
                            ], className="mb-1 small", style={"wordBreak": "break-word"})
                        )
                except:
                    continue

    if exp_attrs:
        attributes.append(
            html.Div([
                html.H6("Experiment Attributes", className="text-primary mb-2 border-bottom pb-1"),
                html.Div(exp_attrs[:20])  # Limit to first 20 attributes to avoid overwhelming UI
            ], className="mb-3")
        )
    else:
        attributes.append(
            html.Div([
                html.H6("Experiment Attributes", className="text-primary mb-2 border-bottom pb-1"),
                html.P("No public attributes found", className="text-muted small")
            ], className="mb-3")
        )

    # Methods/Functions info
    if hasattr(experiment, 'get_browser_functions'):
        try:
            plot_functions = experiment.get_browser_functions()
            if plot_functions:
                func_list = [html.Li(name, className="small") for name, _ in plot_functions]
                attributes.append(
                    html.Div([
                        html.H6("Available Plot Functions", className="text-primary mb-2 border-bottom pb-1"),
                        html.Ul(func_list, className="small mb-0")
                    ], className="mb-3")
                )
        except:
            pass

    return html.Div(attributes, style={"fontSize": "0.9rem"})


# ============================================================================
# Shared Layout Components
# ============================================================================

def create_header_layout(title, subtitle):
    """Create the header section of the dashboard."""
    return dbc.Row([
        dbc.Col([
            html.H1(title, className="mb-2 text-primary"),
            html.P(subtitle, className="text-muted mb-3"),
            html.Hr(className="mb-3"),
        ])
    ], className="flex-shrink-0")


def create_plot_display_section():
    """Create the plot display area with loading states."""
    return dcc.Loading(
        id="loading-plot",
        type="graph",
        children=dcc.Graph(
            id="plot-display",
            style={"height": "100%", "minHeight": "400px"},
            config={"displayModeBar": True, "displaylogo": False}
        ),
        color="#0d6efd"
    )
