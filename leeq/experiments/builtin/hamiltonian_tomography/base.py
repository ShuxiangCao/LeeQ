# This file contains the base class for implementing Hamiltonian tomography.
from leeq import Experiment


class HamiltonianTomographyBaseSingleQudit(Experiment):
    """
    Base class for implementing Hamiltonian tomography.

    """
    
    EPII_INFO = {
        "name": "HamiltonianTomographyBaseSingleQudit",
        "description": "Base class for Hamiltonian tomography experiments on single qudits",
        "purpose": "Abstract base class that provides the foundation for implementing Hamiltonian tomography experiments. This class defines the interface for characterizing unknown Hamiltonians by measuring qudit evolution under different tomography axes.",
        "attributes": {
            # Base class typically doesn't have many direct attributes
            # Child classes implement the actual logic
        },
        "notes": [
            "This is an abstract base class - do not instantiate directly",
            "Child classes must implement the run() method",
            "Used as foundation for single-qubit Hamiltonian characterization experiments"
        ]
    }

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
