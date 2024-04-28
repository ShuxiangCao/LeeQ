# This file contains the base class for implementing Hamiltonian tomography.
from leeq import Experiment


class HamiltonianTomographyBaseSingleQudit(Experiment):
    """
    Base class for implementing Hamiltonian tomography.

    """

    def run(self, duts, tomography_axis):
        """
        This method should be implemented by the child class.
        """
        raise NotImplementedError


