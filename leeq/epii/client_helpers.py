"""
Helper functions for EPII clients to work with extended_data.
"""

import pickle
import logging

logger = logging.getLogger(__name__)


def get_extended_data(response):
    """
    Deserialize extended_data from an ExperimentResponse.
    
    Args:
        response: EPII ExperimentResponse containing extended_data
        
    Returns:
        dict: Deserialized data with attribute names as keys
    """
    data = {}
    
    if not hasattr(response, 'extended_data'):
        return data
    
    for key, value in response.extended_data.items():
        try:
            # Deserialize from pickle
            data[key] = pickle.loads(value)
        except Exception as e:
            # If unpickling fails, keep as bytes
            logger.debug(f"Could not unpickle {key}: {e}")
            data[key] = value
    
    return data


def list_extended_attributes(response):
    """
    List all available attributes in extended_data.
    
    Args:
        response: EPII ExperimentResponse
        
    Returns:
        list: Names of all extended attributes
    """
    if hasattr(response, 'extended_data'):
        return list(response.extended_data.keys())
    return []


def get_extended_attribute(response, attribute_name, default=None):
    """
    Get a specific attribute from extended_data.
    
    Args:
        response: EPII ExperimentResponse
        attribute_name: Name of the attribute to retrieve
        default: Default value if attribute not found
        
    Returns:
        The deserialized attribute value or default
    """
    if not hasattr(response, 'extended_data'):
        return default
    
    if attribute_name not in response.extended_data:
        return default
    
    try:
        return pickle.loads(response.extended_data[attribute_name])
    except Exception as e:
        logger.debug(f"Could not unpickle {attribute_name}: {e}")
        return response.extended_data[attribute_name]