"""
EPII Data Serialization Utilities

This module provides serialization utilities for converting between
LeeQ data types (numpy arrays, plotly figures) and EPII protobuf messages.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import plotly  # noqa: F401
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .proto import epii_pb2

logger = logging.getLogger(__name__)


def numpy_array_to_protobuf(
    array: np.ndarray,
    name: str = "data",
    metadata: Optional[Dict[str, str]] = None
) -> epii_pb2.NumpyArray:
    """
    Convert a numpy array to EPII protobuf NumpyArray message.

    Uses numpy.tobytes() for efficient serialization as recommended
    by research on gRPC numpy transmission (4.5ms for 921k×3 arrays).

    Args:
        array: Numpy array to serialize
        name: Name identifier for the array (e.g., "raw_data", "processed")
        metadata: Optional metadata dictionary

    Returns:
        EPII NumpyArray protobuf message
    """
    numpy_msg = epii_pb2.NumpyArray()

    if array is None:
        return numpy_msg

    try:
        # Ensure array is C-contiguous for consistent serialization
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)

        # Serialize array data using tobytes() - most efficient method
        numpy_msg.data = array.tobytes()

        # Store shape information
        numpy_msg.shape.extend(array.shape)

        # Store dtype as string
        numpy_msg.dtype = str(array.dtype)

        # Set name
        numpy_msg.name = name

        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                numpy_msg.metadata[key] = str(value)

        return numpy_msg

    except Exception as e:
        logger.error("Failed to convert numpy array to protobuf: %s", e)
        return numpy_msg


def protobuf_to_numpy_array(numpy_msg: epii_pb2.NumpyArray) -> np.ndarray:
    """
    Convert EPII protobuf NumpyArray message back to numpy array.

    Args:
        numpy_msg: EPII NumpyArray protobuf message

    Returns:
        Reconstructed numpy array
    """
    if not numpy_msg.data:
        return np.array([])

    try:
        # Reconstruct array from buffer
        array = np.frombuffer(numpy_msg.data, dtype=numpy_msg.dtype)

        # Reshape to original dimensions
        if numpy_msg.shape:
            array = array.reshape(list(numpy_msg.shape))

        return array

    except Exception as e:
        logger.error("Failed to convert protobuf to numpy array: %s", e)
        return np.array([])


def browser_function_to_plot_component(func_name: str, figure: Any) -> epii_pb2.PlotComponent:
    """
    Convert browser function result to PlotComponent with JSON and PNG data.
    
    Args:
        func_name: Name of browser function
        figure: Plotly or matplotlib figure object
    
    Returns:
        PlotComponent with description, JSON, and PNG data
    """
    import plotly.io as pio
    from plotly.tools import mpl_to_plotly
    
    component = epii_pb2.PlotComponent()
    
    # Convert matplotlib to plotly if needed
    if hasattr(figure, 'savefig'):  # matplotlib figure
        try:
            plotly_fig = mpl_to_plotly(figure)
        except:
            # Fallback if conversion fails
            plotly_fig = figure
    else:  # assume plotly figure
        plotly_fig = figure
    
    # Extract plot title from figure
    plot_title = ""
    try:
        if hasattr(plotly_fig, 'layout') and hasattr(plotly_fig.layout, 'title'):
            title_obj = plotly_fig.layout.title
            if hasattr(title_obj, 'text') and title_obj.text:
                plot_title = str(title_obj.text)
            elif title_obj and str(title_obj) != 'layout.Title()':
                plot_title = str(title_obj)
        elif hasattr(figure, '_suptitle') and figure._suptitle:
            plot_title = figure._suptitle.get_text()
    except Exception:
        pass
    
    # Create description including function name
    if plot_title:
        component.description = f"{plot_title} [{func_name}]"
    else:
        component.description = f"Plot from {func_name}"
    
    # Generate JSON and PNG with simple error handling
    try:
        component.plotly_json = plotly_fig.to_json()
        component.image_png = pio.to_image(plotly_fig, format='png')
    except Exception:
        component.plotly_json = ""
        component.image_png = b""
    
    return component



def protobuf_to_plotly_figure(plot_msg) -> Dict[str, Any]:  # epii_pb2.PlotData removed in Phase 1
    """
    Convert EPII protobuf PlotData message back to plotly figure dictionary.

    Args:
        plot_msg: EPII PlotData protobuf message

    Returns:
        Dictionary representation of plotly figure
    """
    if not plot_msg.traces:
        return {}

    try:
        fig_dict = {
            'data': [],
            'layout': {
                'title': {'text': plot_msg.title} if plot_msg.title else {}
            }
        }

        # Convert traces
        for trace_msg in plot_msg.traces:
            trace_dict = {
                'type': trace_msg.type or 'scatter',
                'name': trace_msg.name
            }

            if trace_msg.x:
                trace_dict['x'] = list(trace_msg.x)
            if trace_msg.y:
                trace_dict['y'] = list(trace_msg.y)
            if trace_msg.z:
                trace_dict['z'] = list(trace_msg.z)

            fig_dict['data'].append(trace_dict)

        # Restore layout from JSON strings
        for key, value in plot_msg.layout.items():
            try:
                fig_dict['layout'][key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                fig_dict['layout'][key] = value

        return fig_dict

    except Exception as e:
        logger.error("Failed to convert protobuf to plotly figure: %s", e)
        return {}


def handle_complex_dtype(array: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Handle complex and other special numpy dtypes for serialization.

    Args:
        array: Input numpy array

    Returns:
        Tuple of (converted array, dtype string for metadata)
    """
    original_dtype = str(array.dtype)

    # Handle complex numbers by converting to real representation
    if np.issubdtype(array.dtype, np.complexfloating):
        # Store as interleaved real and imaginary parts
        real_imag = np.empty(array.shape + (2,), dtype=np.float64)
        real_imag[..., 0] = array.real
        real_imag[..., 1] = array.imag
        return real_imag.reshape(-1), f"complex:{original_dtype}"

    # Handle object arrays by converting to string
    elif array.dtype == np.object:
        str_array = np.array([str(x) for x in array.flat], dtype='U')
        return str_array, f"object:{original_dtype}"

    # Handle structured arrays
    elif array.dtype.names is not None:
        # Convert to bytes representation
        return array.tobytes(), f"structured:{original_dtype}"

    # Standard numeric types pass through
    else:
        return array, original_dtype


def restore_complex_dtype(array: np.ndarray, dtype_info: str) -> np.ndarray:
    """
    Restore complex and special dtypes from serialized form.

    Args:
        array: Deserialized array
        dtype_info: Dtype information string with metadata

    Returns:
        Array with restored dtype
    """
    if dtype_info.startswith("complex:"):
        # Restore complex array from interleaved real/imaginary
        original_dtype = dtype_info.split(":", 1)[1]
        real_imag = array.reshape(-1, 2)
        complex_array = real_imag[..., 0] + 1j * real_imag[..., 1]
        return complex_array.astype(original_dtype)

    elif dtype_info.startswith("object:"):
        # Return as string array (can't fully restore objects)
        return array

    elif dtype_info.startswith("structured:"):
        # Would need the dtype definition to fully restore
        logger.warning("Cannot fully restore structured array without dtype definition")
        return array

    else:
        return array


# Keep backward compatibility functions
def serialize_numpy_array(array: np.ndarray) -> bytes:
    """
    Legacy function - serialize a numpy array to bytes.

    Args:
        array: Numpy array to serialize

    Returns:
        Serialized bytes representation
    """
    if array is None:
        return b''

    try:
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
        return array.tobytes()
    except Exception as e:
        logger.error("Failed to serialize numpy array: %s", e)
        return b''


def deserialize_numpy_array(data: bytes, shape: tuple, dtype: str = 'float64') -> np.ndarray:
    """
    Legacy function - deserialize bytes back to a numpy array.

    Args:
        data: Serialized bytes data
        shape: Expected array shape
        dtype: Numpy data type string

    Returns:
        Reconstructed numpy array
    """
    if not data:
        return np.array([])

    try:
        array = np.frombuffer(data, dtype=dtype)
        return array.reshape(shape)
    except Exception as e:
        logger.error("Failed to deserialize numpy array: %s", e)
        return np.array([])


def serialize_value(value: Any) -> Dict[str, Any]:
    """
    Serialize a generic Python value to a dictionary.

    Handles various Python types and converts them to a
    serializable format for protobuf transmission.

    Args:
        value: Python value to serialize

    Returns:
        Dictionary with type and value information
    """
    if value is None:
        return {"type": "none", "value": None}
    elif isinstance(value, bool):
        return {"type": "bool", "value": value}
    elif isinstance(value, int):
        return {"type": "int", "value": value}
    elif isinstance(value, float):
        return {"type": "float", "value": value}
    elif isinstance(value, str):
        return {"type": "string", "value": value}
    elif isinstance(value, list):
        return {"type": "list", "value": value}
    elif isinstance(value, dict):
        return {"type": "dict", "value": value}
    elif isinstance(value, np.ndarray):
        return {
            "type": "numpy",
            "value": serialize_numpy_array(value),
            "shape": value.shape,
            "dtype": str(value.dtype)
        }
    else:
        return {"type": "unknown", "value": str(value)}


def deserialize_value(data: Dict[str, Any]) -> Any:
    """
    Deserialize a dictionary back to a Python value.

    Args:
        data: Dictionary with type and value information

    Returns:
        Reconstructed Python value
    """
    value_type = data.get("type", "unknown")
    value = data.get("value")

    if value_type == "none":
        return None
    elif value_type in ["bool", "int", "float", "string", "list", "dict"]:
        return value
    elif value_type == "numpy":
        return deserialize_numpy_array(
            value,
            data.get("shape", ()),
            data.get("dtype", "float64")
        )
    else:
        return value
