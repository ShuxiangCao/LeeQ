import copy

from leeq.core.base import LeeQObject
from leeq.utils import elementwise_update_dict


class SharedParameterObject(LeeQObject):
    """
    Base class for all objects that share parameters. The parameters are passed by share, so that modification
    of the parameters will be reflected in all the objects that share the parameters.
    """

    def __init__(self, name: str, parameters: dict):
        """
        Initialize the object with the name and parameters.

        Parameters:
            name (str): The name of the object.
            parameters (dict): The parameters of the object.
        """
        super().__init__(name)
        self._parameters = parameters

    # @log_event too costly to log every parameter update
    def update_parameters(self, **kwargs):
        """
        Update the parameters of the object.

        Parameters:
            kwargs (dict): The parameters to be updated.
        """
        elementwise_update_dict(self._parameters, kwargs)

    def set_parameters(self, parameters: dict = None, **kwargs):
        """
        Compatibility wrapper for updating parameters from notebooks.
        """
        updates = {}
        if parameters:
            updates.update(parameters)
        updates.update(kwargs)
        self.update_parameters(**updates)

    def __getitem__(self, item):
        """
        Return a parameter by key for dict-like notebook examples.
        """
        return self._parameters[item]

    def get_parameters(self):
        """
        Get the parameters of the object. The returned object is safe to modify.

        Returns:
            dict: The parameters of the object.
        """
        return copy.deepcopy(self._parameters)
