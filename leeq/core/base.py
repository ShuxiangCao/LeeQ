from labchronicle import LoggableObject


class LeeQObject(LoggableObject):
    """
    Base class for most of the LeeQ objects. It maps the name of the object to the human readable id,
    for the persistance with labchronicle.
    """

    def __init__(self, name):
        self._name = name
        super(LoggableObject, self).__init__()

    @property
    def hrid(self):
        """
        Get the human readable id of the object.

        Returns:
            str: The human readable id of the object.
        """
        return self._name
