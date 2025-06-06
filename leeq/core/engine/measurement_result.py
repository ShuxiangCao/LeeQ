import numpy as np

from leeq.core.base import LeeQObject


class MeasurementResult(LeeQObject):
    """
    The MeasurementResult class is used to store the measurement result of a single step, single lpb.
    """

    def __init__(self, step_no: int, data: np.ndarray, mprim_uuid: str):
        super().__init__(
            name=f"MeasurementResult: {mprim_uuid}, step_no: {step_no}")

        self._step_no = step_no
        self._data = [data]
        self._mprim_uuid = mprim_uuid

    @property
    def step_no(self):
        """
        Get the step number.
        """
        return self._step_no

    @property
    def data(self):
        """
        Get the data.
        """
        return np.asarray(self._data)

    def append_data(self, new_data):
        """
        Append new data to the data.
        """
        self._data.append(new_data)

    @property
    def mprim_uuid(self):
        """
        Get the mprim uuid.
        """
        return self._mprim_uuid

    @property
    def shape(self):
        return self.data.shape
