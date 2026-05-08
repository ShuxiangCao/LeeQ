# This file contains the base class for implementing Hamiltonian tomography.
from leeq import Experiment


class HamiltonianTomographyBaseSingleQudit(Experiment):
    """
    Base class for implementing Hamiltonian tomography.

    """

    def run(self, duts, tomography_axis):
        """
        Execute the Hamiltonian tomography experiment on hardware.

        This method should be implemented by the child class.

        Parameters
        ----------
        duts : list
            List of device under test (qudit/qubit objects).
        tomography_axis : str or list of str
            Axis or axes along which to perform tomography ('X', 'Y', or 'Z').

        Returns
        -------
        None
            Results are stored in instance attributes by child implementations.

        Raises
        ------
        NotImplementedError
            This is an abstract method that must be implemented by child classes.
        """
        raise NotImplementedError
