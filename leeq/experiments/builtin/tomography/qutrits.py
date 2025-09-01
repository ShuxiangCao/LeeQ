from leeq.chronicle import log_and_record
from leeq.theory.tomography import MultiQutritModel
from leeq.utils.compatibility import prims

from .base import (
    GeneralisedProcessTomography,
    GeneralisedStateTomography,
    GeneralisedTomographyBase,
)


class QutritTomographyBase(GeneralisedTomographyBase):

    def initialize_gate_lpbs(self, duts):
        _gate_lpbs_func = {}

        _gate_lpbs_func['I'] = lambda dut: dut.get_gate('I')
        _gate_lpbs_func['Xp_01'] = lambda dut: dut.get_gate('Xp', transition_name='f01')
        _gate_lpbs_func['Yp_01'] = lambda dut: dut.get_gate('Yp', transition_name='f01')
        _gate_lpbs_func['Xp_12'] = lambda dut: dut.get_gate('Xp', transition_name='f12')
        _gate_lpbs_func['Yp_12'] = lambda dut: dut.get_gate('Yp', transition_name='f12')
        _gate_lpbs_func['H'] = lambda dut: dut.get_gate('qutrit_hadamard')
        _gate_lpbs_func['Z1'] = lambda dut: dut.get_gate('qutrit_clifford_green', angle=(1, 0))
        _gate_lpbs_func['Z2'] = lambda dut: dut.get_gate('qutrit_clifford_green', angle=(0, 1))

        gate_names = list(_gate_lpbs_func.keys())
        full_gate_names = gate_names

        number_of_qubits = len(duts)
        for _i in range(number_of_qubits - 1):
            new_gate_names = []
            for gate_name in gate_names:
                for full_gate_name in full_gate_names:
                    new_gate_names.append(full_gate_name + ':' + gate_name)
            full_gate_names = new_gate_names

        def _get_multi_qubit_lpb_from_name(name):
            gate_names = name.split(':')
            lpb = prims.ParallelLPB(
                [_gate_lpbs_func[gate_name](duts[i]) for i, gate_name in enumerate(gate_names)])
            return lpb

        self._gate_lpbs = dict(zip(full_gate_names, [_get_multi_qubit_lpb_from_name(name) for name in full_gate_names], strict=False))


class MultiQutritsStateTomography(GeneralisedStateTomography, QutritTomographyBase):
    EPII_INFO = {
        "name": "MultiQutritsStateTomography",
        "description": "Multi-qutrit quantum state tomography experiment",
        "purpose": "Performs complete state tomography on multiple qutrits (3-level quantum systems) to reconstruct the joint density matrix. Uses generalized Gell-Mann measurements for full characterization.",
        "attributes": {
            "model": {
                "type": "MultiQutritModel",
                "description": "Multi-qutrit tomography model"
            },
            "state_tomography_model": {
                "type": "Any",
                "description": "State tomography model for multi-qutrit system"
            },
            "measurement_mitigation": {
                "type": "CalibrateFullAssignmentMatrices or None",
                "description": "Measurement error mitigation calibration"
            },
            "result": {
                "type": "np.ndarray",
                "description": "Raw measurement results",
                "shape": "(n_qutrits, 8^n_qutrits, n_outcomes)"
            },
            "prob": {
                "type": "np.ndarray[float]",
                "description": "Probability distributions after processing",
                "shape": "(8^n_qutrits, 3^n_qutrits)"
            },
            "vec": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed state vector",
                "shape": "(3^n_qutrits,)"
            },
            "dm": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed density matrix",
                "shape": "(3^n_qutrits, 3^n_qutrits)"
            },
            "_gate_lpbs": {
                "type": "dict",
                "description": "Dictionary of qutrit gates for measurements (I, Xp_01, Yp_01, Xp_12, Yp_12, H, Z1, Z2)"
            },
            "base": {
                "type": "int",
                "description": "Base for qutrit system (always 3)"
            }
        },
        "notes": [
            "Uses 8 measurement bases per qutrit for complete tomography",
            "mprim_index default is 1 for qutrits",
            "Supports f01 and f12 transitions for qutrit operations",
            "Includes qutrit-specific gates like Hadamard and Clifford operations",
            "Base is automatically set to 3 for qutrit systems"
        ]
    }

    @log_and_record
    def run(self, duts, mprim_index=1, initial_lpb=None, extra_measurement_duts=None):
        """
        Execute multi-qutrit state tomography.
        
        Parameters
        ----------
        duts : List[Any]
            List of qutrit devices under test.
        mprim_index : int, optional
            Measurement primitive index. Default: 1
        initial_lpb : LogicalPrimitiveBlock, optional
            Initial logical primitive block to prepare the state. Default: None
        extra_measurement_duts : List[Any], optional
            Additional DUTs for extra measurements. Default: None
            
        Returns
        -------
        None
            Results are stored in instance attributes (result, prob, vec, dm).
        """
        model = MultiQutritModel(len(duts))
        super().run(duts, model, mprim_index, initial_lpb, extra_measurement_duts, base=3)


class MultiQutritsProcessTomography(GeneralisedProcessTomography, QutritTomographyBase):
    EPII_INFO = {
        "name": "MultiQutritsProcessTomography",
        "description": "Multi-qutrit quantum process tomography experiment",
        "purpose": "Performs complete process tomography on multi-qutrit quantum operations to reconstruct the process matrix. Characterizes arbitrary quantum channels acting on 3-level quantum systems.",
        "attributes": {
            "model": {
                "type": "MultiQutritModel",
                "description": "Multi-qutrit tomography model"
            },
            "process_tomography_model": {
                "type": "Any",
                "description": "Process tomography model for multi-qutrit system"
            },
            "measurement_mitigation": {
                "type": "CalibrateFullAssignmentMatrices or None",
                "description": "Measurement error mitigation calibration"
            },
            "result": {
                "type": "np.ndarray",
                "description": "Raw measurement results",
                "shape": "(n_qutrits, 9^n_qutrits, 8^n_qutrits, n_outcomes)"
            },
            "prob": {
                "type": "np.ndarray[float]",
                "description": "Probability distributions after processing",
                "shape": "(9^n_qutrits, 8^n_qutrits, 3^n_qutrits)"
            },
            "ptm": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed process transfer matrix",
                "shape": "(9^n_qutrits, 9^n_qutrits)"
            },
            "_gate_lpbs": {
                "type": "dict",
                "description": "Dictionary of qutrit gates for state preparation and measurement"
            },
            "base": {
                "type": "int",
                "description": "Base for qutrit system (always 3)"
            }
        },
        "notes": [
            "Uses 9 preparation states and 8 measurement bases per qutrit",
            "The lpb parameter contains the process to be characterized",
            "mprim_index default is 1 for qutrits",
            "Base is automatically set to 3 for qutrit systems",
            "initial_lpb parameter is incorrectly named - should be extra_measurement_duts"
        ]
    }

    @log_and_record
    def run(self, duts, lpb, mprim_index=1, initial_lpb=None, extra_measurement_duts=None):
        """
        Execute multi-qutrit process tomography.
        
        Parameters
        ----------
        duts : List[Any]
            List of qutrit devices under test.
        lpb : LogicalPrimitiveBlock
            The logical primitive block (process) to characterize.
        mprim_index : int, optional
            Measurement primitive index. Default: 1
        initial_lpb : None, optional
            Not used - kept for compatibility. Default: None
        extra_measurement_duts : List[Any], optional
            Additional DUTs for extra measurements. Default: None
            
        Returns
        -------
        None
            Results are stored in instance attributes (result, prob, ptm).
        """
        model = MultiQutritModel(len(duts))
        super().run(duts=duts, model=model, lpb=lpb, mprim_index=mprim_index,
                    extra_measurement_duts=extra_measurement_duts, base=3)
