from typing import List, Any, Union

import numpy as np
from leeq.chronicle import log_and_record

from k_agents.inspection.decorator import text_inspection
from leeq.utils.compatibility import prims
from .base import GeneralisedTomographyBase, GeneralisedSingleDutStateTomography, GeneralisedStateTomography, \
    GeneralisedProcessTomography
from leeq.theory.tomography.utils import evaluate_fidelity_density_matrix_with_state_vector, \
    evaluate_fidelity_ptm_with_unitary


class QubitTomographyBase(GeneralisedTomographyBase):
    def initialize_gate_lpbs(self, duts):

        gate_names = ['I', 'Xp', 'Yp', 'X']
        full_gate_names = gate_names

        number_of_qubits = len(duts)
        for i in range(number_of_qubits - 1):
            new_gate_names = []
            for gate_name in gate_names:
                for full_gate_name in full_gate_names:
                    new_gate_names.append(full_gate_name + ':' + gate_name)
            full_gate_names = new_gate_names

        def _get_multi_qubit_lpb_from_name(name):
            gate_names = name.split(':')
            lpb = prims.ParallelLPB(
                [duts[i].get_gate(gate_name, transition_name='f01') for i, gate_name in enumerate(gate_names)])
            return lpb

        self._gate_lpbs = dict(zip(full_gate_names, [_get_multi_qubit_lpb_from_name(name) for name in full_gate_names]))


class SingleQubitStateTomography(GeneralisedSingleDutStateTomography):

    def initialize_gate_lpbs(self, dut):
        self._gate_lpbs = {}

        self._gate_lpbs['I'] = dut.get_gate('I')
        self._gate_lpbs['Xp'] = dut.get_gate('Xp', transition_name='f01')
        self._gate_lpbs['Yp'] = dut.get_gate('Yp', transition_name='f01')
        self._gate_lpbs['X'] = dut.get_gate('X', transition_name='f01')

    @log_and_record
    def run(self, dut, mprim_index=0, initial_lpb=None, extra_measurement_duts=None):
        from leeq.theory.tomography import SingleQubitModel
        model = SingleQubitModel()
        super().run(dut, model, mprim_index, initial_lpb, extra_measurement_duts)


class MultiQubitsStateTomography(GeneralisedStateTomography, QubitTomographyBase):
    _experiment_result_analysis_instructions = """
    The result of the experiment is the density matrix of the quantum state. Check if the density matrix has been reported.
    by data analysis. Report the fidelity if applicable. The fidelity is not the criterion for the success of the
    experiment, do not conclude unsuccessful if the fidelity is low. Conclude unsuccessful if the density matrix is
    not physically meaningful.
    """

    @log_and_record
    def run(self, duts: List[Any], mprim_index: Union[int, str] = 0, initial_lpb: 'LogicalPrimitiveBlock' = None,
            extra_measurement_duts: List[Any] = None,
            measurement_mitigation: Union[None, 'CalibrateFullAssignmentMatrices', bool] = None,
            state_vec_ideal: np.ndarray = None):
        """
        Experiment for multi-qubit state tomography.

        Parameters
        ----------
        duts : List[Any]
            List of DUTs to be characterized.
        mprim_index : Union[int, str]
            Measurement primitive index. For qubit, default is 0.
        initial_lpb : LogicalPrimitiveBlock
            Initial logical primitive block to prepare the quantum state for characterization.
        extra_measurement_duts : List[Any]
            List of DUTs to be used for extra measurements.
        measurement_mitigation : Union[None, 'CalibrateFullAssignmentMatrices', bool]
            Measurement mitigation. If True, use the default calibration. If None, no calibration. If an instance of
            CalibrateFullAssignmentMatrices, use the instance for calibration.
        state_vec_ideal : np.ndarray
            Ideal state vector for comparison. If provided and the state vector is reported, the fidelity will be
            calculated and reported.

        Example
        --------
        >>> # Run two-qubit state tomography of a bell pair.
        >>> from leeq.experiments.builtin import MultiQubitsStateTomography
        >>> import numpy as np
        >>> c1 = dut_q1.get_c1('f01') # assuming the initialized qubits are dut_q1 and dut q2.
        >>> lpb = c1.hadamard() + c2_q1q2.get_cnot() # c2 is the two qubit collection which is initialized in advance.
        >>> state_vec_ideal = np.array([1,0,0,1])/np.sqrt(2)
        >>> tomo = MultiQubitsStateTomography(duts=[dut_q1,dut_q2],initial_lpb=lpb,measurement_mitigation=True,state_vec_ideal=state_vec_ideal)
        """
        from leeq.theory.tomography import MultiQubitModel
        self.state_vector_ideal = state_vec_ideal
        model = MultiQubitModel(len(duts))
        super().run(duts, model, mprim_index, initial_lpb, extra_measurement_duts,
                    measurement_mitigation=measurement_mitigation)

    @text_inspection
    def fitting(self) -> Union[str, None]:
        if self.state_vector_ideal is not None:
            fidelity = evaluate_fidelity_density_matrix_with_state_vector(self.dm, self.state_vector_ideal)
            fidelity = "Calculated fidelity {:.2f}%".format(fidelity * 100)
        else:
            fidelity = "No ideal state vector provided for fidelity calculation."

        return f"""State tomography result:
        <density matrix>
        {self.dm}
        </density matrix>
        <fidelity>
        {fidelity}
        </fidelity>
        """


class MultiQubitsProcessTomography(GeneralisedProcessTomography, QubitTomographyBase):
    _experiment_result_analysis_instructions = """
    The result of the experiment is the Pauli transfer matrix of the quantum process. Check if the Pauli transfer matrix
    has been reported by data analysis, and the result is physically meaningful. Report the fidelity if applicable.
    The fidelity is not the criterion for the success of the experiment, but it is a useful metric for the quality of 
    the quantum gate implementation.
    """

    @log_and_record
    def run(self, duts, lpb, mprim_index=0, extra_measurement_duts=None, measurement_mitigation=None, u_ideal=None):
        """
        Experiment for multi-qubit process tomography.

        Parameters
        ----------
        duts : List[Any]
            List of DUTs (qubits) to be characterized.
        lpb : LogicalPrimitiveBlock
            Logical primitive block representing the quantum process.
        mprim_index : int
            Measurement primitive index. For qubit, default is 0.
        extra_measurement_duts : List[Any]
            List of DUTs to be used for extra measurements.
        measurement_mitigation : Union[None, 'CalibrateFullAssignmentMatrices', bool]
            Measurement mitigation. If True, use the default calibration. If None, no calibration. If an instance of
            CalibrateFullAssignmentMatrices, use the instance for calibration.
        u_ideal : np.ndarray
            Ideal unitary matrix for comparison. If provided and the unitary matrix is reported, the fidelity will be
            calculated and reported

        Example:
        --------
        >>> # Run two-qubit process tomography of a CNOT gate.
        >>> # Assume the initialized qubits are dut_q1 and dut_q2.
        >>> import numpy as np
        >>> lpb = c2.get_cnot()
        >>> u_ideal = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        >>> tomo = MultiQubitsProcessTomography(duts=[dut_q1,dut_q2],lpb=lpb,measurement_mitigation=True,u_ideal=u_ideal)

        """
        self.u_ideal = u_ideal
        from leeq.theory.tomography import MultiQubitModel
        model = MultiQubitModel(len(duts))
        super().run(duts=duts, model=model, lpb=lpb, mprim_index=mprim_index,
                    extra_measurement_duts=extra_measurement_duts, measurement_mitigation=measurement_mitigation)

    @text_inspection
    def fitting(self) -> Union[str, None]:

        if self.u_ideal is not None:
            fidelity = evaluate_fidelity_ptm_with_unitary(self.ptm, self.u_ideal, basis=self.model._basis)
            fidelity = "Calculated fidelity {:.2f}%".format(fidelity * 100)
        else:
            fidelity = "No ideal unitary provided for fidelity calculation."

        return f"""Process tomography result:
        <Pauli transfer matrix>
        {self.ptm.real}
        </Pauli transfer matrix>
        <fidelity>
        {fidelity}
        </fidelity>
        """
