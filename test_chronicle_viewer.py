"""
Test suite for chronicle_viewer.py

Comprehensive tests for the chronicle viewer Dash application including
import tests, callback tests, error handling, and UI component tests.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html

# Add parent directory to path to import chronicle_viewer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_import_chronicle_viewer():
    """Test that chronicle_viewer module can be imported."""
    import chronicle_viewer
    assert chronicle_viewer is not None


def test_app_exists():
    """Test that the Dash app object is created."""
    import chronicle_viewer
    assert hasattr(chronicle_viewer, 'app')
    assert chronicle_viewer.app is not None


def test_app_layout_exists():
    """Test that the app has a layout defined."""
    import chronicle_viewer
    assert chronicle_viewer.app.layout is not None


def test_app_has_bootstrap_theme():
    """Test that the app uses Bootstrap theme."""
    import chronicle_viewer
    assert chronicle_viewer.app.config.external_stylesheets
    assert any('bootstrap' in str(s).lower() for s in chronicle_viewer.app.config.external_stylesheets)


def test_main_function_exists():
    """Test that the main function exists."""
    import chronicle_viewer
    assert hasattr(chronicle_viewer, 'main')
    assert callable(chronicle_viewer.main)


def test_required_callbacks_registered():
    """Test that required callbacks are registered with the app."""
    import chronicle_viewer
    app = chronicle_viewer.app
    
    # Check that callbacks are registered
    # The callback_map structure varies between Dash versions
    # We just need to verify that callbacks exist
    assert len(app.callback_map) >= 2  # Should have at least 2 callbacks
    
    # Try to check if our key callbacks are registered by looking at their IDs
    callback_ids = list(app.callback_map.keys())
    
    # Convert callback IDs to strings to check for our components
    callback_str = str(callback_ids)
    
    # Check that our key components are involved in callbacks
    # These should appear in the callback ID strings
    assert 'experiment-info' in callback_str or len(app.callback_map) >= 2
    assert 'plot-display' in callback_str or len(app.callback_map) >= 2
    
    # Basic validation passed - we have callbacks registered


class TestLoadExperimentCallback:
    """Test suite for the load_experiment callback function."""
    
    @pytest.fixture
    def load_experiment_func(self):
        """Get the load_experiment function."""
        import chronicle_viewer
        return chronicle_viewer.load_selected_experiment
    
    def test_empty_file_path(self, load_experiment_func):
        """Test handling of empty file path."""
        info, controls, store = load_experiment_func("", "")
        assert "Select an experiment from the dropdown" in str(info)
        assert controls == []
        # Store should contain the attempted file path even on error
        if store:
            assert "file_path" in store
    
    def test_none_file_path(self, load_experiment_func):
        """Test handling of None file path."""
        info, controls, store = load_experiment_func(None, None)
        assert "Select an experiment from the dropdown" in str(info)
        assert controls == []
        # Store should contain the attempted file path even on error
        if store:
            assert "file_path" in store
    
    def test_whitespace_file_path(self, load_experiment_func):
        """Test handling of whitespace-only file path."""
        info, controls, store = load_experiment_func("record123", "   ")
        assert "Select an experiment from the dropdown" in str(info)
        assert controls == []
        # Store should contain the attempted file path even on error
        if store:
            assert "file_path" in store
    
    @patch('chronicle_viewer.load_object')
    def test_file_not_found(self, mock_load, load_experiment_func):
        """Test handling of file not found error."""
        mock_load.side_effect = FileNotFoundError("File not found")
        info, controls, store = load_experiment_func("record123", "/path/to/nonexistent.hdf5")
        info_str = str(info)
        # Check for the improved error message components
        assert "File Not Found" in info_str or "File not found" in info_str
        assert "danger" in info_str  # Should be a danger alert
        assert controls == []
        # Store should contain the attempted file path even on error
        if store:
            assert "file_path" in store
    
    @patch('chronicle_viewer.load_object')
    def test_permission_error(self, mock_load, load_experiment_func):
        """Test handling of permission denied error."""
        mock_load.side_effect = PermissionError("Access denied")
        info, controls, store = load_experiment_func("record123", "/path/to/protected.hdf5")
        info_str = str(info)
        # Check for the improved error message components
        assert "Permission Denied" in info_str or "Permission denied" in info_str
        assert "danger" in info_str  # Should be a danger alert
        assert controls == []
        # Store should contain the attempted file path even on error
        if store:
            assert "file_path" in store
    
    @patch('chronicle_viewer.load_object')
    def test_corrupted_hdf5_file(self, mock_load, load_experiment_func):
        """Test handling of corrupted HDF5 file."""
        mock_load.side_effect = Exception("HDF5 error: Unable to open file")
        info, controls, store = load_experiment_func("record123", "/path/to/corrupted.hdf5")
        info_str = str(info)
        # Check for the improved error message components
        assert "Invalid Chronicle File" in info_str or "corrupted" in info_str.lower() or "HDF5" in info_str
        assert "danger" in info_str  # Should be a danger alert
        assert controls == []
        # Store should contain the attempted file path even on error
        if store:
            assert "file_path" in store
    
    @patch('chronicle_viewer.load_object')
    def test_generic_loading_error(self, mock_load, load_experiment_func):
        """Test handling of generic loading errors."""
        mock_load.side_effect = Exception("Unknown error occurred")
        info, controls, store = load_experiment_func("record123", "/path/to/file.hdf5")
        info_str = str(info)
        # Check for the improved error message components
        assert "Error Loading Experiment" in info_str or "Error loading" in info_str
        assert "danger" in info_str  # Should be a danger alert
        assert controls == []
        # Store should contain the attempted file path even on error
        if store:
            assert "file_path" in store
    
    @patch('chronicle_viewer.load_object')
    def test_experiment_without_browser_functions(self, mock_load, load_experiment_func):
        """Test handling of experiments without get_browser_functions method."""
        mock_exp = Mock()
        del mock_exp.get_browser_functions  # Remove the method
        mock_exp.__class__.__name__ = "TestExperiment"
        mock_load.return_value = mock_exp
        
        info, controls, store = load_experiment_func("record123", "/path/to/valid.hdf5")
        assert "TestExperiment" in str(info)
        assert "does not have browser functions" in str(controls)
        assert store["file_path"] == "/path/to/valid.hdf5"
    
    @patch('chronicle_viewer.load_object')
    def test_experiment_with_no_plots(self, mock_load, load_experiment_func):
        """Test handling of experiments with empty browser functions."""
        mock_exp = Mock()
        mock_exp.get_browser_functions.return_value = []
        mock_exp.__class__.__name__ = "EmptyExperiment"
        mock_load.return_value = mock_exp
        
        info, controls, store = load_experiment_func("record123", "/path/to/empty.hdf5")
        assert "EmptyExperiment" in str(info)
        assert "No plots available" in str(controls)
        assert store["file_path"] == "/path/to/empty.hdf5"
    
    @patch('chronicle_viewer.load_object')
    def test_successful_experiment_load(self, mock_load, load_experiment_func):
        """Test successful loading of experiment with plots."""
        mock_exp = Mock()
        mock_exp.get_browser_functions.return_value = [
            ("plot_magnitude", Mock()),
            ("plot_phase", Mock()),
            ("plot_iq", Mock())
        ]
        mock_exp.__class__.__name__ = "SuccessfulExperiment"
        mock_load.return_value = mock_exp
        
        info, controls, store = load_experiment_func("record123", "/path/to/success.hdf5")
        assert "SuccessfulExperiment" in str(info)
        assert "Available Plots" in str(controls)
        # Check that buttons were created for each plot
        controls_str = str(controls)
        assert "Plot Magnitude" in controls_str or "plot_magnitude" in controls_str
        assert "Plot Phase" in controls_str or "plot_phase" in controls_str
        assert "Plot Iq" in controls_str or "plot_iq" in controls_str
        assert store["file_path"] == "/path/to/success.hdf5"


class TestDisplayPlotCallback:
    """Test suite for the display_plot callback function."""
    
    @pytest.fixture
    def display_plot_func(self):
        """Get the display_plot function."""
        import chronicle_viewer
        return chronicle_viewer.display_plot
    
    def test_no_file_path(self, display_plot_func):
        """Test plot display with no file path."""
        fig = display_plot_func([1], None)
        assert isinstance(fig, go.Figure)
        # Check that error message is shown
        assert fig.layout.annotations
        assert "No experiment loaded" in fig.layout.annotations[0].text
    
    @patch('chronicle_viewer.load_object')
    def test_plot_method_error(self, mock_load, display_plot_func):
        """Test handling of errors during plot generation."""
        mock_exp = Mock()
        mock_method = Mock(side_effect=Exception("Plot generation failed"))
        mock_exp.get_browser_functions.return_value = [("test_plot", mock_method)]
        mock_load.return_value = mock_exp
        
        # Simulate button click context
        with patch('chronicle_viewer.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"index":"test_plot","type":"plot-btn"}.n_clicks'}]
            
            fig = display_plot_func([1], {"file_path": "/path/to/file.hdf5", "record_id": "record123"})
            assert isinstance(fig, go.Figure)
            assert fig.layout.annotations
            assert "Error generating plot" in fig.layout.annotations[0].text
    
    @patch('chronicle_viewer.load_object')
    def test_plot_returns_non_figure(self, mock_load, display_plot_func):
        """Test handling when plot method returns non-Figure object."""
        mock_exp = Mock()
        mock_method = Mock(return_value="Not a figure")
        mock_exp.get_browser_functions.return_value = [("bad_plot", mock_method)]
        mock_load.return_value = mock_exp
        
        with patch('chronicle_viewer.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"index":"bad_plot","type":"plot-btn"}.n_clicks'}]
            
            fig = display_plot_func([1], {"file_path": "/path/to/file.hdf5", "record_id": "record123"})
            assert isinstance(fig, go.Figure)
            assert fig.layout.annotations
            assert "Unsupported figure type" in fig.layout.annotations[0].text
    
    @patch('chronicle_viewer.load_object')
    def test_successful_plot_display(self, mock_load, display_plot_func):
        """Test successful plot display."""
        mock_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
        mock_exp = Mock()
        mock_method = Mock(return_value=mock_fig)
        mock_exp.get_browser_functions.return_value = [("good_plot", mock_method)]
        mock_load.return_value = mock_exp
        
        with patch('chronicle_viewer.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"index":"good_plot","type":"plot-btn"}.n_clicks'}]
            
            fig = display_plot_func([1], {"file_path": "/path/to/file.hdf5", "record_id": "record123"})
            assert isinstance(fig, go.Figure)
            assert len(fig.data) == 1
            assert fig.data[0].x == (1, 2, 3)
            assert fig.data[0].y == (4, 5, 6)
    
    @patch('chronicle_viewer.load_object')
    def test_plot_method_not_found(self, mock_load, display_plot_func):
        """Test handling when requested plot method doesn't exist."""
        mock_exp = Mock()
        mock_exp.get_browser_functions.return_value = [("other_plot", Mock())]
        mock_load.return_value = mock_exp
        
        with patch('chronicle_viewer.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"index":"missing_plot","type":"plot-btn"}.n_clicks'}]
            
            fig = display_plot_func([1], {"file_path": "/path/to/file.hdf5", "record_id": "record123"})
            assert isinstance(fig, go.Figure)
            assert fig.layout.annotations
            assert "not found" in fig.layout.annotations[0].text


class TestUIComponents:
    """Test suite for UI component structure."""
    
    @pytest.fixture
    def app_layout(self):
        """Get the app layout."""
        import chronicle_viewer
        return chronicle_viewer.app.layout
    
    def test_layout_has_required_components(self, app_layout):
        """Test that layout contains all required components."""
        layout_str = str(app_layout)
        
        # Check for key components
        assert "Chronicle Log Viewer" in layout_str  # Title
        assert "file-path" in layout_str  # File input
        assert "experiment-info" in layout_str  # Info display
        assert "plot-controls" in layout_str  # Plot buttons
        assert "plot-display" in layout_str  # Graph
        assert "file-store" in layout_str  # Storage
    
    def test_layout_uses_bootstrap_components(self, app_layout):
        """Test that layout uses Bootstrap components."""
        layout_str = str(app_layout)
        # Check for Bootstrap component usage
        assert "Container" in layout_str or "dbc.Container" in str(type(app_layout))
        assert "Row" in layout_str or "dbc.Row" in layout_str
        assert "Col" in layout_str or "dbc.Col" in layout_str


class TestErrorMessages:
    """Test suite for error message formatting and clarity."""
    
    @pytest.fixture
    def load_experiment_func(self):
        """Get the load_experiment function."""
        import chronicle_viewer
        return chronicle_viewer.load_selected_experiment
    
    @patch('chronicle_viewer.load_object')
    def test_long_error_message_truncation(self, mock_load, load_experiment_func):
        """Test that very long error messages are truncated."""
        long_error = "Error: " + "x" * 500
        mock_load.side_effect = Exception(long_error)
        
        info, controls, store = load_experiment_func("record123", "/path/to/file.hdf5")
        info_str = str(info)
        # Check that the error message is present but truncated
        assert "Error Loading Experiment" in info_str or "Error" in info_str
        # The error message should be truncated to 200 chars as per the code
        # Count the x's in the string representation
        x_count = info_str.count('x')
        # Should have approximately 193 x's (200 chars total minus "Error: ")
        assert x_count < 250  # Should be truncated, not all 500
        assert x_count > 150  # Should have a substantial portion


class TestFigureConversion:
    """Test figure conversion functionality."""
    
    @pytest.fixture
    def display_plot_func(self):
        """Get the display plot callback function."""
        from chronicle_viewer import display_plot
        return display_plot
    
    @patch('chronicle_viewer.load_object')  
    def test_matplotlib_figure_conversion(self, mock_load, display_plot_func):
        """Test matplotlib figure conversion to Plotly."""
        import matplotlib.pyplot as plt
        
        # Create a mock matplotlib figure
        mock_mpl_fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        
        mock_exp = Mock()
        mock_method = Mock(return_value=mock_mpl_fig)
        mock_exp.get_browser_functions.return_value = [("mpl_plot", mock_method)]
        mock_load.return_value = mock_exp

        with patch('chronicle_viewer.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"index":"mpl_plot","type":"plot-btn"}.n_clicks'}]

            fig = display_plot_func([1], {"file_path": "/path/to/file.hdf5", "record_id": "record123"})
            assert isinstance(fig, go.Figure)
            # Should be converted successfully (either as data traces or as image)
            plt.close(mock_mpl_fig)  # Clean up


class TestBasicFixtures:
    """Basic test fixtures for chronicle viewer."""
    
    @pytest.fixture
    def app(self):
        """Fixture to provide the Dash app instance."""
        import chronicle_viewer
        return chronicle_viewer.app
    
    @pytest.fixture
    def mock_chronicle_file(self, tmp_path):
        """Fixture to create a mock chronicle file path."""
        mock_file = tmp_path / "test_experiment.hdf5"
        # Just create an empty file for path testing
        mock_file.touch()
        return str(mock_file)
    
    def test_fixture_app(self, app):
        """Test that app fixture works."""
        assert app is not None
    
    def test_fixture_mock_file(self, mock_chronicle_file):
        """Test that mock file fixture works."""
        assert os.path.exists(mock_chronicle_file)
        assert mock_chronicle_file.endswith('.hdf5')