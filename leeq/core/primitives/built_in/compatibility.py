from leeq.core.primitives.base import SharedParameterObject


class PulseArgsUpdatable(SharedParameterObject):
    """
    A mixin class for the pulse arguments updatable objects. This class is used to provide backwards compatibility
    to the old pulse arguments update methods.
    """

    def update_pulse_args(self, **kwargs):
        """
        An alias of update_parameters for backwards compatibility.

        Parameters:
            kwargs (dict): The pulse arguments to be updated.
        """
        return self.update_parameters(**kwargs)

    def update_freq(self, freq):
        """
        Update the frequency of the pulse.

        Parameters:
            freq (float): The frequency of the pulse.
        """
        return self.update_parameters(freq=freq)

    def set_pulse_shapes(self, func, **kwargs):
        """
        Set the pulse shapes of the pulse.
        """
        return self.update_parameters(shape=func.__qualname__, kwargs=kwargs)

    def set_iq_skew(self, iq_skew):
        """
        Set the iq skew of the pulse.
        """
        return self.update_parameters(iq_skew=iq_skew)

    def primary_shape(self):
        """
        Get the shape of the pulse.
        """
        args = self.get_parameters()
        return args["shape"]

    def primary_kwargs(self):
        """
        Get the kwargs of the pulse.
        """
        return self.get_parameters()

    def primary_iq_skew(self):
        """
        Get the iq skew of the pulse.
        """
        return self.get_parameters()["iq_skew"]

    def primary_channel(self):
        """
        Get the channel of the pulse.
        """
        return self.get_parameters()["channel"]

    def get_pulse_args(self):
        """
        Get the pulse arguments of the pulse.
        """
        return self.get_parameters()
