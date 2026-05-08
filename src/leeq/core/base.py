import uuid

from leeq.chronicle import LoggableObject


class LeeQObject(LoggableObject):
    """
    Base class for most of the LeeQ objects. It maps the name of the object to the human readable id,
    for the persistance with leeq.chronicle.
    """

    def __init__(self, name):
        self._name = name
        self._uuid = uuid.uuid4()
        super(LeeQObject, self).__init__()

    @property
    def hrid(self):
        """
        Get the human readable id of the object.

        Returns:
            str: The human readable id of the object.
        """
        return self._name

    @property
    def uuid(self):
        """
        Get the uuid of the object.

        Returns:
            str: The uuid of the object.
        """
        return self._uuid
