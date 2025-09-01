"""
Helper functions for EPII clients to work with the new protocol structure.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_data(response):
    """
    Extract all data from an ExperimentResponse.

    Args:
        response: EPII ExperimentResponse

    Returns:
        dict: Data name -> value mapping
    """
    data = {}

    for item in response.data:
        if item.HasField('number'):
            data[item.name] = item.number
        elif item.HasField('text'):
            data[item.name] = item.text
        elif item.HasField('boolean'):
            data[item.name] = item.boolean
        elif item.HasField('array'):
            # Reconstruct numpy array
            arr = item.array
            value = np.frombuffer(arr.data, dtype=arr.dtype)
            data[item.name] = value.reshape(arr.shape)

    return data


def get_data_with_descriptions(response):
    """
    Extract data with descriptions.

    Returns:
        dict: Data name -> (value, description) tuples
    """
    data = {}

    for item in response.data:
        if item.HasField('number'):
            value = item.number
        elif item.HasField('text'):
            value = item.text
        elif item.HasField('boolean'):
            value = item.boolean
        elif item.HasField('array'):
            arr = item.array
            value = np.frombuffer(arr.data, dtype=arr.dtype).reshape(arr.shape)
        else:
            value = None

        data[item.name] = (value, item.description)

    return data


def get_calibration_results(response):
    """
    Extract calibration results from data based on description.

    Searches through the data items for numeric values whose descriptions
    contain the word 'calibration', typically fit parameters from experiments.

    Args:
        response: EPII ExperimentResponse containing data items

    Returns:
        dict: Calibration parameter name -> value mapping
    """
    results = {}
    for item in response.data:
        if item.HasField('number') and 'calibration' in item.description.lower():
            results[item.name] = item.number
    return results


def get_arrays(response):
    """
    Extract all numpy arrays from data.

    Retrieves all data items that contain array values, typically measurement
    data, traces, or multi-dimensional results from experiments.

    Args:
        response: EPII ExperimentResponse containing data items

    Returns:
        dict: Array name -> numpy.ndarray mapping
    """
    arrays = {}
    for item in response.data:
        if item.HasField('array'):
            arr = item.array
            value = np.frombuffer(arr.data, dtype=arr.dtype)
            arrays[item.name] = value.reshape(arr.shape)
    return arrays


def get_docs(response):
    """
    Get all documentation from the response.

    Extracts both the run documentation (how to use the experiment) and
    data documentation (what the data means) from the response.

    Args:
        response: EPII ExperimentResponse with documentation

    Returns:
        dict: Dictionary with 'run' and 'data' documentation strings
    """
    return {
        'run': response.docs.run,
        'data': response.docs.data
    }


def get_metadata(response):
    """
    Get EPII metadata as a dictionary.

    Extracts metadata key-value pairs from the response, typically containing
    information like experiment purpose, category, analysis methods, and other
    descriptive attributes from the EPII_INFO decorator.

    Args:
        response: EPII ExperimentResponse with metadata

    Returns:
        dict: Metadata key -> value mapping (all values as strings)
    """
    return dict(response.metadata)


# Legacy compatibility functions (deprecated, will be removed in future)
def get_extended_data(response):
    """
    Legacy function for backward compatibility.
    Extract all data using the new protocol structure.

    Args:
        response: EPII ExperimentResponse

    Returns:
        dict: All data as a dictionary
    """
    logger.warning("get_extended_data is deprecated. Use get_data() instead.")
    return get_data(response)


def list_extended_attributes(response):
    """
    Legacy function for backward compatibility.
    List all available data item names.

    Args:
        response: EPII ExperimentResponse

    Returns:
        list: Names of all data items
    """
    logger.warning("list_extended_attributes is deprecated. Use get_data().keys() instead.")
    return [item.name for item in response.data]


def get_extended_attribute(response, attribute_name, default=None):
    """
    Legacy function for backward compatibility.
    Get a specific data item by name.

    Args:
        response: EPII ExperimentResponse
        attribute_name: Name of the data item to retrieve
        default: Default value if not found

    Returns:
        The data value or default
    """
    logger.warning("get_extended_attribute is deprecated. Use get_data()[attribute_name] instead.")
    data = get_data(response)
    return data.get(attribute_name, default)
