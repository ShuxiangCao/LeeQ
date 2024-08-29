import numpy as np
from labchronicle import register_browser_function, log_and_record

from leeq import Experiment, Sweeper, basic_run
from leeq.theory.utils import to_dense_probabilities
from leeq.utils.compatibility import prims

from typing import List, Any


class CalibrateFullAssignmentMatrices(Experiment):
    """
    A class that extends Experiment to perform an assignment matrix experiment
    on a set of Device Under Test (DUTs) based on specified measurement primitives.
    The experiment generates the assignment matrix based on the DUTs and their quantum state transitions and it is
    exponentially more complex than the single DUT version.

    Attributes:
        qubit_number (int): The number of qubits in the DUTs.
        result (np.ndarray): The raw result data from the measurement.
        assignment_matrix (np.ndarray): The computed probability matrix for state assignments.
        max_level (int): The highest level of distinguishable state supported.
    """

    @log_and_record
    def run(self, duts: List[Any], mprim_index: int = 0, plot: bool = True) -> None:
        """
        Executes the experiment, generating the assignment matrix based on the DUTs and their quantum state transitions.

        Args:
            duts (List[Any]): List of Device Under Test instances to perform measurements.
            mprim_index (int): Index to select the measurement primitive from each DUT.
            plot (bool): Flag to indicate whether to plot results after running the experiment.
        """
        # Initialize groups for logical Pauli Block (LPB) configuration.
        lpb_groups = [[]]
        self.qubit_number = len(duts)

        # Generate new LPB groups based on the max distinguishable quantum state levels.
        for dut in duts:
            mprim = dut.get_measurement_prim_intlist(mprim_index)
            max_level = np.max(mprim.get_parameters()['distinguishable_states'])
            new_lpb_groups = []
            for x in lpb_groups:
                # Add gates for different transitions.
                new_lpb_groups.append(x + [dut.get_gate('I', transition_name='f01')])
                new_lpb_groups.append(x + [dut.get_gate('X', transition_name='f01')])
                if max_level == 1:
                    continue
                new_lpb_groups.append(
                    x + [dut.get_gate('X', transition_name='f01') + dut.get_gate('X', transition_name='f12')])
                if max_level == 2:
                    continue
                new_lpb_groups.append(x + [
                    dut.get_gate('X', transition_name='f01') + dut.get_gate('X', transition_name='f12') + dut.get_gate(
                        'X', transition_name='f23')])
                if max_level > 3:
                    assert False, f'Level {max_level} not supported yet.'

            lpb_groups = new_lpb_groups

        # Create the LPB structures from the groups and a Sweeper to control the experiment.
        lpbs = [prims.ParallelLPB(x) for x in lpb_groups]
        lpb = prims.SweepLPB(lpbs)
        swp = Sweeper.from_sweep_lpb(lpb)

        # Gather measurement primitives for all DUTs.
        mprims = [dut.get_measurement_prim_intlist(mprim_index) for dut in duts]

        # Execute the basic run and record results.
        basic_run(lpb + prims.ParallelLPB(mprims), swp, '<zs>')

        # Process and transpose results to format them as a dense probability matrix.
        self.result = np.squeeze(np.asarray([x.result() for x in mprims][::-1]), axis=-1)
        self.assignment_matrix = to_dense_probabilities(self.result.transpose([0, 2, 1]), base=max_level + 1)
        self.max_level = max_level

    @register_browser_function(available_after=(run,))
    def plot_result(self) -> None:
        """
        Plots the results using Plotly to visualize the assignment matrix.
        """
        import plotly.express as px

        # Prepare the labels for each axis based on the number of qubits and levels.
        qubit_number = self.qubit_number
        labels = [f'{i}' for i in range(self.max_level + 1)]
        label_group = ['']
        for i in range(qubit_number):
            label_group = [x + y for x in label_group for y in labels]

        # Create and display the plot.
        fig = px.imshow(self.assignment_matrix,
                        labels=dict(x="Prepare", y="Measurement", color="Probability"),
                        x=label_group,
                        y=label_group,
                        color_continuous_scale='teal')

        fig.update_xaxes(side="top")
        fig.show()

    def apply_inverse(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply the inverse of the assignment matrix to the given probabilities.

        Args:
            probs (np.ndarray): The probabilities to apply the inverse assignment matrix to.

        Returns:
            np.ndarray: The probabilities after applying the inverse assignment matrix.
        """
        prob_shape = probs.shape
        new_probs = np.linalg.inv(self.assignment_matrix) @ probs.reshape([prob_shape[0], -1])
        return new_probs.reshape(prob_shape)


class CalibrateSingleDutAssignmentMatrices(Experiment):
    """
    A class that extends Experiment to perform an assignment matrix experiment
    on a set of Device Under Test (DUTs) based on specified measurement primitives.
    The assignment matrix is generated for a single DUT.

    Attributes:
        qubit_number (int): The number of qubits in the DUTs.
        result (np.ndarray): The raw result data from the measurement.
        assignment_matrix (np.ndarray): The computed probability matrix for state assignments.
        max_level (int): The highest level of distinguishable state supported.
    """

    @log_and_record
    def run(self, duts: List[Any], mprim_index: int = 0, plot: bool = True) -> None:
        """
        Executes the experiment, generating the assignment matrix based on the DUTs and their quantum state transitions.

        Args:
            duts (List[Any]): List of Device Under Test instances to perform measurements.
            mprim_index (int): Index to select the measurement primitive from each DUT.
            plot (bool): Flag to indicate whether to plot results after running the experiment.
        """
        # Initialize groups for logical Pauli Block (LPB) configuration.
        self.qubit_number = len(duts)

        # Generate new LPB groups based on the max distinguishable quantum state levels.
        mprim = duts[0].get_measurement_prim_intlist(mprim_index)
        max_level = np.max(mprim.get_parameters()['distinguishable_states'])
        lpb_groups = []
        # Add gates for different transitions.
        lpb_groups.append(prims.ParallelLPB([dut.get_gate('I', transition_name='f01') for dut in duts]))
        lpb_groups.append(prims.ParallelLPB([dut.get_gate('X', transition_name='f01') for dut in duts]))
        if max_level > 1:
            lpb_groups.append(
                prims.ParallelLPB(
                    [dut.get_gate('X', transition_name='f01') + dut.get_gate('X', transition_name='f12') for dut in
                     duts]))
        if max_level > 2:
            lpb_groups.append(prims.ParallelLPB([
                dut.get_gate('X', transition_name='f01') + dut.get_gate('X', transition_name='f12') + dut.get_gate(
                    'X', transition_name='f23') for dut in duts]))
        if max_level > 3:
            assert False, f'Level {max_level} not supported yet.'

        # Create the LPB structures from the groups and a Sweeper to control the experiment.
        lpb = prims.SweepLPB(lpb_groups)
        swp = Sweeper.from_sweep_lpb(lpb)

        # Gather measurement primitives for all DUTs.
        mprims = [dut.get_measurement_prim_intlist(mprim_index) for dut in duts]

        # Execute the basic run and record results.
        basic_run(lpb + prims.ParallelLPB(mprims), swp, 'prob')

        # Process and transpose results to format them as a dense probability matrix.
        self.result = [np.squeeze(x.result()) for x in mprims]
        self.assignment_matrix = [np.squeeze(r).T for r in self.result]
        self.max_level = max_level

    @register_browser_function(available_after=(run,))
    def plot_all(self):
        """
        Plots the results using Plotly to visualize the assignment matrix.
        """
        for i in range(len(self.assignment_matrix)):
            self.plot_result(i)

    def plot_result(self, i) -> None:
        """
        Plots the results using Plotly to visualize the assignment matrix.
        """
        import plotly.express as px

        # Prepare the labels for each axis based on the number of qubits and levels.
        qubit_number = self.qubit_number
        labels = [f'{i}' for i in range(self.max_level + 1)]
        label_group = ['']
        for i in range(qubit_number):
            label_group = [x + y for x in label_group for y in labels]

        # Create and display the plot.
        fig = px.imshow(self.assignment_matrix[i],
                        labels=dict(x="Prepare", y="Measurement", color="Probability"),
                        x=label_group,
                        y=label_group,
                        color_continuous_scale='teal')

        fig.update_xaxes(side="top")
        fig.show()

    def apply_inverse(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply the inverse of the assignment matrix to the given probabilities.

        Args:
            probs (np.ndarray): The probabilities to apply the inverse assignment matrix to.

        Returns:
            np.ndarray: The probabilities after applying the inverse assignment matrix.
        """
        new_probs = []
        for i, prob in enumerate(probs):
            prob_shape = prob.shape
            new_prob = np.linalg.inv(self.assignment_matrix[i]) @ prob.reshape([prob_shape[0], -1])
            new_probs.append(new_prob.reshape(prob_shape))

        return new_probs
