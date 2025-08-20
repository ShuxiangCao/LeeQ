"""
Tests for EPII serialization utilities.
"""

import json
import numpy as np
import pytest

from leeq.epii.serialization import (
    numpy_array_to_protobuf,
    protobuf_to_numpy_array,
    plotly_figure_to_protobuf,
    protobuf_to_plotly_figure,
    handle_complex_dtype,
    restore_complex_dtype,
    serialize_numpy_array,
    deserialize_numpy_array,
    serialize_value,
    deserialize_value,
)
from leeq.epii.proto import epii_pb2

# Skip plotly tests if not available
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TestNumpyArraySerialization:
    """Test numpy array serialization to/from protobuf."""
    
    def test_simple_array_serialization(self):
        """Test basic array serialization."""
        # Create test array
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Convert to protobuf
        pb_msg = numpy_array_to_protobuf(arr, name="test_data")
        
        assert pb_msg.name == "test_data"
        assert list(pb_msg.shape) == [5]
        assert pb_msg.dtype == "float64"
        assert len(pb_msg.data) > 0
        
        # Convert back
        restored = protobuf_to_numpy_array(pb_msg)
        
        np.testing.assert_array_equal(restored, arr)
        
    def test_multidimensional_array(self):
        """Test 2D and 3D array serialization."""
        # 2D array
        arr_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        pb_msg = numpy_array_to_protobuf(arr_2d, name="2d_array")
        
        assert list(pb_msg.shape) == [2, 3]
        assert pb_msg.dtype == "int32"
        
        restored_2d = protobuf_to_numpy_array(pb_msg)
        np.testing.assert_array_equal(restored_2d, arr_2d)
        
        # 3D array
        arr_3d = np.random.randn(3, 4, 5).astype(np.float32)
        pb_msg = numpy_array_to_protobuf(arr_3d, name="3d_array")
        
        assert list(pb_msg.shape) == [3, 4, 5]
        assert pb_msg.dtype == "float32"
        
        restored_3d = protobuf_to_numpy_array(pb_msg)
        np.testing.assert_array_almost_equal(restored_3d, arr_3d)
        
    def test_different_dtypes(self):
        """Test serialization of different numpy dtypes."""
        dtypes = [
            np.float32,
            np.float64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.bool_,
        ]
        
        for dtype in dtypes:
            arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
            pb_msg = numpy_array_to_protobuf(arr)
            restored = protobuf_to_numpy_array(pb_msg)
            
            assert restored.dtype == arr.dtype
            np.testing.assert_array_equal(restored, arr)
            
    def test_empty_array(self):
        """Test empty array handling."""
        arr = np.array([])
        pb_msg = numpy_array_to_protobuf(arr)
        restored = protobuf_to_numpy_array(pb_msg)
        
        assert restored.size == 0
        
    def test_none_array(self):
        """Test None array handling."""
        pb_msg = numpy_array_to_protobuf(None)
        assert pb_msg.data == b''
        
    def test_metadata(self):
        """Test metadata attachment to arrays."""
        arr = np.array([1, 2, 3])
        metadata = {
            "unit": "Hz",
            "description": "Frequency sweep",
            "timestamp": "2024-01-01T00:00:00"
        }
        
        pb_msg = numpy_array_to_protobuf(arr, name="frequency", metadata=metadata)
        
        assert pb_msg.metadata["unit"] == "Hz"
        assert pb_msg.metadata["description"] == "Frequency sweep"
        assert pb_msg.metadata["timestamp"] == "2024-01-01T00:00:00"
        
    def test_non_contiguous_array(self):
        """Test handling of non-contiguous arrays."""
        # Create non-contiguous array by slicing
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        non_contiguous = arr[:, ::2]  # Every other column
        
        assert not non_contiguous.flags['C_CONTIGUOUS']
        
        pb_msg = numpy_array_to_protobuf(non_contiguous)
        restored = protobuf_to_numpy_array(pb_msg)
        
        np.testing.assert_array_equal(restored, non_contiguous)


class TestComplexDtypeHandling:
    """Test handling of complex and special dtypes."""
    
    def test_complex_arrays(self):
        """Test complex number array serialization."""
        arr = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
        
        converted, dtype_info = handle_complex_dtype(arr)
        assert dtype_info.startswith("complex:")
        
        # Converted array should be real with doubled size
        assert converted.dtype.kind == 'f'
        assert converted.size == arr.size * 2
        
        # Restore
        restored = restore_complex_dtype(converted, dtype_info)
        np.testing.assert_array_equal(restored, arr)
        
    def test_complex64_arrays(self):
        """Test complex64 array handling."""
        arr = np.array([1+2j, 3+4j], dtype=np.complex64)
        
        converted, dtype_info = handle_complex_dtype(arr)
        restored = restore_complex_dtype(converted, dtype_info)
        
        assert restored.dtype == np.complex64
        np.testing.assert_array_almost_equal(restored, arr)


class TestLegacyFunctions:
    """Test backward compatibility functions."""
    
    def test_serialize_deserialize_numpy(self):
        """Test legacy numpy serialization functions."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        
        # Serialize
        data = serialize_numpy_array(arr)
        assert isinstance(data, bytes)
        assert len(data) == arr.nbytes
        
        # Deserialize
        restored = deserialize_numpy_array(data, arr.shape, str(arr.dtype))
        np.testing.assert_array_equal(restored, arr)
        
    def test_serialize_value(self):
        """Test generic value serialization."""
        # Test various types
        values = [
            None,
            True,
            42,
            3.14,
            "hello",
            [1, 2, 3],
            {"key": "value"},
            np.array([1, 2, 3]),
        ]
        
        for value in values:
            serialized = serialize_value(value)
            assert "type" in serialized
            assert "value" in serialized
            
            # Deserialize
            restored = deserialize_value(serialized)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(restored, value)
            else:
                assert restored == value


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestPlotlySerialization:
    """Test plotly figure serialization."""
    
    def test_simple_scatter_plot(self):
        """Test simple scatter plot serialization."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4],
            y=[10, 11, 12, 13],
            name="Test trace"
        ))
        fig.update_layout(title="Test Plot")
        
        # Convert to protobuf
        pb_msg = plotly_figure_to_protobuf(fig)
        
        assert pb_msg.plot_type == "scatter"
        assert pb_msg.title == "Test Plot"
        assert len(pb_msg.traces) == 1
        
        trace = pb_msg.traces[0]
        assert trace.name == "Test trace"
        assert list(trace.x) == [1, 2, 3, 4]
        assert list(trace.y) == [10, 11, 12, 13]
        
        # Convert back
        fig_dict = protobuf_to_plotly_figure(pb_msg)
        assert fig_dict['layout']['title']['text'] == "Test Plot"
        assert len(fig_dict['data']) == 1
        assert fig_dict['data'][0]['x'] == [1, 2, 3, 4]
        
    def test_multiple_traces(self):
        """Test plot with multiple traces."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name="Trace 1"))
        fig.add_trace(go.Scatter(x=[2, 3], y=[4, 5], name="Trace 2"))
        
        pb_msg = plotly_figure_to_protobuf(fig)
        
        assert len(pb_msg.traces) == 2
        assert pb_msg.traces[0].name == "Trace 1"
        assert pb_msg.traces[1].name == "Trace 2"
        
    def test_heatmap_plot(self):
        """Test heatmap plot serialization."""
        z_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        fig = go.Figure(data=go.Heatmap(z=z_data))
        
        pb_msg = plotly_figure_to_protobuf(fig)
        
        assert pb_msg.plot_type == "heatmap"
        assert len(pb_msg.traces[0].z) == 9  # Flattened 3x3
        
    def test_layout_serialization(self):
        """Test layout properties serialization."""
        fig = go.Figure()
        fig.update_layout(
            xaxis={"title": "X Axis"},
            yaxis={"title": "Y Axis"},
            width=800,
            height=600,
            showlegend=False
        )
        
        pb_msg = plotly_figure_to_protobuf(fig)
        
        # Check layout is serialized as JSON strings
        assert "xaxis" in pb_msg.layout
        assert "yaxis" in pb_msg.layout
        
        xaxis = json.loads(pb_msg.layout["xaxis"])
        # Plotly wraps title in a dict with 'text' key
        assert xaxis["title"]["text"] == "X Axis"
        
    def test_none_figure(self):
        """Test None figure handling."""
        pb_msg = plotly_figure_to_protobuf(None)
        assert len(pb_msg.traces) == 0
        
    def test_dict_figure(self):
        """Test dictionary figure format."""
        fig_dict = {
            'data': [{
                'x': [1, 2, 3],
                'y': [4, 5, 6],
                'type': 'scatter',
                'name': 'Dict trace'
            }],
            'layout': {'title': {'text': 'Dict Plot'}}
        }
        
        pb_msg = plotly_figure_to_protobuf(fig_dict)
        
        assert pb_msg.title == "Dict Plot"
        assert pb_msg.traces[0].name == "Dict trace"


def test_performance_large_array():
    """Test serialization performance with large arrays."""
    # Create large array similar to research benchmark (921k×3)
    large_array = np.random.randn(921000, 3).astype(np.float32)
    
    # Time the serialization
    import time
    start = time.time()
    pb_msg = numpy_array_to_protobuf(large_array)
    serialization_time = time.time() - start
    
    # Should be fast (research showed 4.5ms for similar size)
    assert serialization_time < 0.1  # 100ms is generous for CI
    
    # Verify data integrity
    restored = protobuf_to_numpy_array(pb_msg)
    assert restored.shape == large_array.shape
    assert restored.dtype == large_array.dtype
    
    # Check a few random values
    indices = np.random.randint(0, large_array.shape[0], 10)
    for i in indices:
        np.testing.assert_array_almost_equal(restored[i], large_array[i])