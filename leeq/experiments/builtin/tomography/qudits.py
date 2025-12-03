from leeq.chronicle import log_and_record
from leeq.theory.tomography import MultiQuditModel
from leeq.utils.compatibility import prims

from .base import (
    GeneralisedProcessTomography,
    GeneralisedStateTomography,
    GeneralisedTomographyBase,
)


class QuditTomographyBase(GeneralisedTomographyBase):

    def initialize_gate_lpbs(self, duts):
        _gate_lpbs_func = {}

        _gate_lpbs_func['I'] = lambda dut: dut.get_gate('I')
        _gate_lpbs_func['Xp_01'] = lambda dut: dut.get_gate('Xp', transition_name='f01')
        _gate_lpbs_func['Yp_01'] = lambda dut: dut.get_gate('Yp', transition_name='f01')
        _gate_lpbs_func['Xp_12'] = lambda dut: dut.get_gate('Xp', transition_name='f12')
        _gate_lpbs_func['Yp_12'] = lambda dut: dut.get_gate('Yp', transition_name='f12')
        _gate_lpbs_func['Xp_23'] = lambda dut: dut.get_gate('Xp', transition_name='f23')
        _gate_lpbs_func['Yp_23'] = lambda dut: dut.get_gate('Yp', transition_name='f23')

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


class MultiQuditsStateTomography(GeneralisedStateTomography, QuditTomographyBase):
    EPII_INFO = {
        "name": "MultiQuditsStateTomography",
        "description": "Multi-qudit quantum state tomography experiment",
        "purpose": "Performs complete state tomography on multiple qudits (4-level quantum systems) to reconstruct the joint density matrix. Extends tomography to higher-dimensional quantum systems.",
        "attributes": {
            "model": {
                "type": "MultiQuditModel",
                "description": "Multi-qudit tomography model"
            },
            "state_tomography_model": {
                "type": "Any",
                "description": "State tomography model for multi-qudit system"
            },
            "measurement_mitigation": {
                "type": "CalibrateFullAssignmentMatrices or None",
                "description": "Measurement error mitigation calibration"
            },
            "result": {
                "type": "np.ndarray",
                "description": "Raw measurement results",
                "shape": "(n_qudits, n_measurements, n_outcomes)"
            },
            "prob": {
                "type": "np.ndarray[float]",
                "description": "Probability distributions after processing",
                "shape": "(n_measurements, 4^n_qudits)"
            },
            "vec": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed state vector",
                "shape": "(4^n_qudits,)"
            },
            "dm": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed density matrix",
                "shape": "(4^n_qudits, 4^n_qudits)"
            },
            "_gate_lpbs": {
                "type": "dict",
                "description": "Dictionary of qudit gates for measurements (I, Xp_01, Yp_01, Xp_12, Yp_12, Xp_23, Yp_23)"
            },
            "base": {
                "type": "int",
                "description": "Base for qudit system (always 4)"
            }
        },
        "notes": [
            "Supports transitions f01, f12, and f23 for 4-level systems",
            "mprim_index default is 2 for qudits",
            "Base is automatically set to 4 for qudit systems",
            "Measurement complexity scales with dimension as O(d^2)",
            "Results must be analyzed before accessing vec and dm attributes"
        ]
    }

    @log_and_record
    def run(self, duts, mprim_index=2, initial_lpb=None, extra_measurement_duts=None):
        """
        Execute multi-qudit state tomography.

        Parameters
        ----------
        duts : List[Any]
            List of qudit devices under test.
        mprim_index : int, optional
            Measurement primitive index. Default: 2
        initial_lpb : LogicalPrimitiveBlock, optional
            Initial logical primitive block to prepare the state. Default: None
        extra_measurement_duts : List[Any], optional
            Additional DUTs for extra measurements. Default: None

        Returns
        -------
        None
            Results are stored in instance attributes (result, prob, vec, dm).
        """
        model = MultiQuditModel(len(duts))
        super().run(duts, model, mprim_index, initial_lpb, extra_measurement_duts, base=4)


class MultiQuditsProcessTomography(GeneralisedProcessTomography, QuditTomographyBase):
    EPII_INFO = {
        "name": "MultiQuditsProcessTomography",
        "description": "Multi-qudit quantum process tomography experiment",
        "purpose": "Performs complete process tomography on multi-qudit quantum operations to reconstruct the process matrix. Characterizes arbitrary quantum channels acting on 4-level quantum systems.",
        "attributes": {
            "model": {
                "type": "MultiQuditModel",
                "description": "Multi-qudit tomography model"
            },
            "process_tomography_model": {
                "type": "Any",
                "description": "Process tomography model for multi-qudit system"
            },
            "measurement_mitigation": {
                "type": "CalibrateFullAssignmentMatrices or None",
                "description": "Measurement error mitigation calibration"
            },
            "result": {
                "type": "np.ndarray",
                "description": "Raw measurement results",
                "shape": "(n_qudits, n_preparations, n_measurements, n_outcomes)"
            },
            "prob": {
                "type": "np.ndarray[float]",
                "description": "Probability distributions after processing",
                "shape": "(n_preparations, n_measurements, 4^n_qudits)"
            },
            "ptm": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed process transfer matrix",
                "shape": "(16^n_qudits, 16^n_qudits)"
            },
            "_gate_lpbs": {
                "type": "dict",
                "description": "Dictionary of qudit gates for state preparation and measurement"
            },
            "base": {
                "type": "int",
                "description": "Base for qudit system (always 4)"
            }
        },
        "notes": [
            "Supports transitions f01, f12, and f23 for 4-level systems",
            "The lpb parameter contains the process to be characterized",
            "mprim_index default is 2 for qudits",
            "Base is automatically set to 4 for qudit systems",
            "initial_lpb parameter is incorrectly named - should be extra_measurement_duts",
            "Process matrix dimension is 16^n_qudits × 16^n_qudits"
        ]
    }

    @log_and_record
    def run(self, duts, lpb, mprim_index=2, initial_lpb=None, extra_measurement_duts=None):
        """
        Execute multi-qudit process tomography.

        Parameters
        ----------
        duts : List[Any]
            List of qudit devices under test.
        lpb : LogicalPrimitiveBlock
            The logical primitive block (process) to characterize.
        mprim_index : int, optional
            Measurement primitive index. Default: 2
        initial_lpb : None, optional
            Not used - kept for compatibility. Default: None
        extra_measurement_duts : List[Any], optional
            Additional DUTs for extra measurements. Default: None

        Returns
        -------
        None
            Results are stored in instance attributes (result, prob, ptm).
        """
        model = MultiQuditModel(len(duts))
        super().run(duts=duts, model=model, lpb=lpb, mprim_index=mprim_index,
                    extra_measurement_duts=extra_measurement_duts, base=4)
