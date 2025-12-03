import numpy as np
from matplotlib import pyplot as plt

from leeq import Experiment, Sweeper, basic_run
from leeq.chronicle import log_and_record, register_browser_function
from leeq.theory.utils import to_dense_probabilities
from leeq.utils.compatibility import prims


class GeneralisedTomographyBase(object):

    def initialize_gate_lpbs(self, dut):
        pass

    def get_lpb(self, name):
        if isinstance(name, tuple) or isinstance(name, list):
            lpbs = [self.get_lpb(x) for x in name]
            lpb = prims.SeriesLPB(lpbs)
            return lpb

        return self._gate_lpbs[name]


class GeneralisedSingleDutStateTomography(Experiment, GeneralisedTomographyBase):
    EPII_INFO = {
        "name": "GeneralisedSingleDutStateTomography",
        "description": "Base class for single-DUT quantum state tomography experiments",
        "purpose": "Performs state tomography on a single quantum device under test (DUT) using linear inversion. This base class provides the framework for reconstructing quantum states from measurement data.",
        "attributes": {
            "model": {
                "type": "Any",
                "description": "Tomography model defining measurement and reconstruction methods"
            },
            "state_tomography_model": {
                "type": "Any",
                "description": "State tomography model constructed from the base model"
            },
            "result": {
                "type": "np.ndarray",
                "description": "Raw measurement results",
                "shape": "(n_measurements,)"
            },
            "vec": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed state vector",
                "shape": "(d,)"
            },
            "dm": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed density matrix",
                "shape": "(d, d)"
            },
            "_gate_lpbs": {
                "type": "dict",
                "description": "Dictionary mapping gate names to logical primitive blocks"
            }
        },
        "notes": [
            "This is a base class that requires a specific tomography model",
            "The model defines the measurement sequences and reconstruction method",
            "Results are reconstructed using linear inversion",
            "Subclasses should implement initialize_gate_lpbs for specific gate sets"
        ]
    }

    def run(self, dut, model, mprim_index=1, initial_lpb=None, extra_measurement_duts=None):
        """
        Execute state tomography on a single DUT.

        Parameters
        ----------
        dut : Any
            The device under test (quantum system).
        model : Any
            Tomography model defining measurement and reconstruction methods.
        mprim_index : int, optional
            Measurement primitive index. Default: 1
        initial_lpb : LogicalPrimitiveBlock, optional
            Initial logical primitive block to prepare the state. Default: None
        extra_measurement_duts : List[Any], optional
            Additional DUTs for extra measurements. Default: None

        Returns
        -------
        None
            Results are stored in instance attributes (result, vec, dm).
        """
        self.model = model
        self.initialize_gate_lpbs(dut=dut)

        self.state_tomography_model = self.model.construct_state_tomography()

        measurement_sequences = self.state_tomography_model.get_measurement_sequence()

        measurement_lpb = prims.SweepLPB([self.get_lpb(x) for x in measurement_sequences])
        swp_measurement = Sweeper.from_sweep_lpb(measurement_lpb)

        mprim = dut.get_measurement_prim_intlist(mprim_index)

        lpb = measurement_lpb + mprim

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        basic_run(lpb, swp_measurement, 'prob')
        self.result = np.squeeze(mprim.result())

        self.vec, self.dm = self.state_tomography_model.linear_inverse_state_tomography(self.result.T)

    @register_browser_function(available_after=(run,))
    def plot(self):
        self.model.plot_density_matrix(self.dm)
        plt.show()


class GeneralisedSingleDutProcessTomography(Experiment, GeneralisedTomographyBase):
    EPII_INFO = {
        "name": "GeneralisedSingleDutProcessTomography",
        "description": "Base class for single-DUT quantum process tomography experiments",
        "purpose": "Performs process tomography on a single quantum device under test (DUT) to characterize quantum operations. Reconstructs the process matrix using linear inversion from measurement data.",
        "attributes": {
            "model": {
                "type": "Any",
                "description": "Tomography model defining measurement and reconstruction methods"
            },
            "process_tomography_model": {
                "type": "Any",
                "description": "Process tomography model constructed from the base model"
            },
            "result": {
                "type": "np.ndarray",
                "description": "Raw measurement results",
                "shape": "(n_prep, n_meas, n_outcomes)"
            },
            "ptm": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed process transfer matrix",
                "shape": "(d^2, d^2)"
            },
            "_gate_lpbs": {
                "type": "dict",
                "description": "Dictionary mapping gate names to logical primitive blocks"
            }
        },
        "notes": [
            "This is a base class that requires a specific tomography model",
            "The model defines preparation and measurement sequences",
            "The lpb parameter contains the process to be characterized",
            "Results are reconstructed using linear inversion process tomography"
        ]
    }

    @log_and_record
    def run(self, dut, model, lpb=None, mprim_index=1, extra_measurement_duts=None):
        """
        Execute process tomography on a single DUT.

        Parameters
        ----------
        dut : Any
            The device under test (quantum system).
        model : Any
            Tomography model defining measurement and reconstruction methods.
        lpb : LogicalPrimitiveBlock, optional
            The logical primitive block (process) to characterize. Default: None
        mprim_index : int, optional
            Measurement primitive index. Default: 1
        extra_measurement_duts : List[Any], optional
            Additional DUTs for extra measurements. Default: None

        Returns
        -------
        None
            Results are stored in instance attributes (result, ptm).
        """
        self.model = model
        self.initialize_gate_lpbs(dut=dut)

        self.process_tomography_model = self.model.construct_process_tomography()

        measurement_sequences = self.process_tomography_model.get_measurement_sequence()
        preparation_sequences = self.process_tomography_model.get_preparation_sequence()

        measurement_lpb = prims.SweepLPB([self.get_lpb(x) for x in measurement_sequences])
        swp_measurement = Sweeper.from_sweep_lpb(measurement_lpb)

        preparation_lpb = prims.SweepLPB([self.get_lpb(x) for x in preparation_sequences])
        swp_preparation = Sweeper.from_sweep_lpb(preparation_lpb)

        mprim = dut.get_measurement_prim_intlist(mprim_index)

        if lpb is not None:
            lpb = preparation_lpb + lpb + measurement_lpb + mprim
        else:
            lpb = preparation_lpb + measurement_lpb + mprim

        basic_run(lpb, swp_measurement + swp_preparation, 'prob')
        self.result = mprim.result()

        self.ptm = self.process_tomography_model.linear_inverse_process_tomography(self.result.transpose([0, 2, 1]))

    @register_browser_function(available_after=(run,))
    def plot(self):
        self.model.plot_process_matrix(self.ptm)
        plt.show()


class GeneralisedStateTomography(Experiment, GeneralisedTomographyBase):
    EPII_INFO = {
        "name": "GeneralisedStateTomography",
        "description": "Base class for multi-DUT quantum state tomography experiments",
        "purpose": "Performs state tomography on multiple quantum devices under test (DUTs) simultaneously. Supports measurement error mitigation and different qudit bases (qubit, qutrit, qudit).",
        "attributes": {
            "model": {
                "type": "Any",
                "description": "Tomography model defining measurement and reconstruction methods"
            },
            "base": {
                "type": "int",
                "description": "Qudit base (2 for qubit, 3 for qutrit, 4 for qudit)"
            },
            "state_tomography_model": {
                "type": "Any",
                "description": "State tomography model constructed from the base model"
            },
            "measurement_mitigation": {
                "type": "CalibrateFullAssignmentMatrices or None",
                "description": "Measurement error mitigation calibration object"
            },
            "result": {
                "type": "np.ndarray",
                "description": "Raw measurement results",
                "shape": "(n_duts, n_measurements, n_outcomes)"
            },
            "prob": {
                "type": "np.ndarray[float]",
                "description": "Probability distributions after processing",
                "shape": "(n_measurements, base^n_duts)"
            },
            "vec": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed state vector",
                "shape": "(base^n_duts,)"
            },
            "dm": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed density matrix",
                "shape": "(base^n_duts, base^n_duts)"
            },
            "_gate_lpbs": {
                "type": "dict",
                "description": "Dictionary mapping gate names to logical primitive blocks"
            }
        },
        "notes": [
            "Supports multi-qudit systems with configurable base",
            "Measurement mitigation can be enabled for error correction",
            "Data analysis must be called before accessing vec and dm",
            "Results are reconstructed using linear inversion"
        ]
    }

    def run(self, duts, model, mprim_index=1, initial_lpb=None, extra_measurement_duts=None, base=2,
            measurement_mitigation=None):
        """
        Execute state tomography on multiple DUTs.

        Parameters
        ----------
        duts : List[Any]
            List of devices under test (quantum systems).
        model : Any
            Tomography model defining measurement and reconstruction methods.
        mprim_index : int, optional
            Measurement primitive index. Default: 1
        initial_lpb : LogicalPrimitiveBlock, optional
            Initial logical primitive block to prepare the state. Default: None
        extra_measurement_duts : List[Any], optional
            Additional DUTs for extra measurements. Default: None
        base : int, optional
            Qudit base (2 for qubit, 3 for qutrit, 4 for qudit). Default: 2
        measurement_mitigation : CalibrateFullAssignmentMatrices or bool or None, optional
            Measurement error mitigation. If True, creates default calibration. Default: None

        Returns
        -------
        None
            Results are stored in instance attributes (result, prob, vec, dm).
        """
        self.model = model
        self.base = base
        self.initialize_gate_lpbs(duts=duts)

        self.measurement_mitigation = measurement_mitigation

        if measurement_mitigation is not None:
            from leeq.experiments.builtin import CalibrateFullAssignmentMatrices
            if not isinstance(measurement_mitigation, CalibrateFullAssignmentMatrices):
                self.measurement_mitigation = CalibrateFullAssignmentMatrices(
                    duts=duts, mprim_index=mprim_index
                )

        self.state_tomography_model = self.model.construct_state_tomography()

        measurement_sequences = self.state_tomography_model.get_measurement_sequence()

        measurement_lpb = prims.SweepLPB([self.get_lpb(x) for x in measurement_sequences])
        swp_measurement = Sweeper.from_sweep_lpb(measurement_lpb)

        mprims = [dut.get_measurement_prim_intlist(mprim_index) for dut in duts]

        lpb = measurement_lpb + prims.ParallelLPB(mprims)

        if initial_lpb is not None:
            lpb = initial_lpb + lpb

        basic_run(lpb, swp_measurement, '<zs>')
        self.result = np.squeeze(np.asarray([mprim.result() for mprim in mprims][::-1]), axis=-1)

    def analyze_data(self):
        self.prob = to_dense_probabilities(self.result.transpose([0, 2, 1]), base=self.base)
        if self.measurement_mitigation is not None:
            self.prob = self.measurement_mitigation.apply_inverse(self.prob)

        self.vec, self.dm = self.state_tomography_model.linear_inverse_state_tomography(self.prob)

    @register_browser_function(available_after=(run,))
    def plot(self):
        self.analyze_data()
        self.model.plot_density_matrix(self.dm, base=self.base)
        plt.show()


class GeneralisedProcessTomography(Experiment, GeneralisedTomographyBase):
    EPII_INFO = {
        "name": "GeneralisedProcessTomography",
        "description": "Base class for multi-DUT quantum process tomography experiments",
        "purpose": "Performs process tomography on multiple quantum devices under test (DUTs) to characterize multi-qudit quantum operations. Supports measurement error mitigation and different qudit bases.",
        "attributes": {
            "model": {
                "type": "Any",
                "description": "Tomography model defining measurement and reconstruction methods"
            },
            "base": {
                "type": "int",
                "description": "Qudit base (2 for qubit, 3 for qutrit, 4 for qudit)"
            },
            "process_tomography_model": {
                "type": "Any",
                "description": "Process tomography model constructed from the base model"
            },
            "measurement_mitigation": {
                "type": "CalibrateFullAssignmentMatrices or None",
                "description": "Measurement error mitigation calibration object"
            },
            "result": {
                "type": "np.ndarray",
                "description": "Raw measurement results",
                "shape": "(n_duts, n_prep, n_meas, n_outcomes)"
            },
            "prob": {
                "type": "np.ndarray[float]",
                "description": "Probability distributions after processing",
                "shape": "(n_prep, n_meas, base^n_duts)"
            },
            "ptm": {
                "type": "np.ndarray[complex]",
                "description": "Reconstructed process transfer matrix",
                "shape": "((base^n_duts)^2, (base^n_duts)^2)"
            },
            "_gate_lpbs": {
                "type": "dict",
                "description": "Dictionary mapping gate names to logical primitive blocks"
            }
        },
        "notes": [
            "Supports multi-qudit systems with configurable base",
            "The lpb parameter contains the process to be characterized",
            "Measurement mitigation can be enabled for error correction",
            "Data analysis must be called before accessing ptm",
            "Results are reconstructed using linear inversion process tomography"
        ]
    }

    @log_and_record
    def run(self, duts, model, lpb=None, mprim_index=1, extra_measurement_duts=None, base=2,
            measurement_mitigation=None):
        """
        Execute process tomography on multiple DUTs.

        Parameters
        ----------
        duts : List[Any]
            List of devices under test (quantum systems).
        model : Any
            Tomography model defining measurement and reconstruction methods.
        lpb : LogicalPrimitiveBlock, optional
            The logical primitive block (process) to characterize. Default: None
        mprim_index : int, optional
            Measurement primitive index. Default: 1
        extra_measurement_duts : List[Any], optional
            Additional DUTs for extra measurements. Default: None
        base : int, optional
            Qudit base (2 for qubit, 3 for qutrit, 4 for qudit). Default: 2
        measurement_mitigation : CalibrateFullAssignmentMatrices or bool or None, optional
            Measurement error mitigation. If True, creates default calibration. Default: None

        Returns
        -------
        None
            Results are stored in instance attributes (result, prob, ptm).
        """
        self.model = model
        self.initialize_gate_lpbs(duts=duts)
        self.base = base
        self.measurement_mitigation = measurement_mitigation

        if measurement_mitigation is not None:
            from leeq.experiments.builtin import CalibrateFullAssignmentMatrices
            if not isinstance(measurement_mitigation, CalibrateFullAssignmentMatrices):
                self.measurement_mitigation = CalibrateFullAssignmentMatrices(
                    duts=duts, mprim_index=mprim_index
                )

        self.process_tomography_model = self.model.construct_process_tomography()

        measurement_sequences = self.process_tomography_model.get_measurement_sequence()
        preparation_sequences = self.process_tomography_model.get_preparation_sequence()

        measurement_lpb = prims.SweepLPB([self.get_lpb(x) for x in measurement_sequences])
        swp_measurement = Sweeper.from_sweep_lpb(measurement_lpb)

        preparation_lpb = prims.SweepLPB([self.get_lpb(x) for x in preparation_sequences])
        swp_preparation = Sweeper.from_sweep_lpb(preparation_lpb)

        mprims = [dut.get_measurement_prim_intlist(mprim_index) for dut in duts]

        if lpb is not None:
            lpb = preparation_lpb + lpb + measurement_lpb + prims.ParallelLPB(mprims)
        else:
            lpb = preparation_lpb + measurement_lpb + prims.ParallelLPB(mprims)

        basic_run(lpb, swp_measurement + swp_preparation, '<zs>')
        self.result = np.squeeze(np.asarray([mprim.result() for mprim in mprims][::-1]), axis=-1)

    def analyze_data(self):
        self.prob = to_dense_probabilities(self.result.transpose([0, 3, 1, 2]), base=self.base)
        if self.measurement_mitigation is not None:
            self.prob = self.measurement_mitigation.apply_inverse(self.prob)
        self.ptm = self.process_tomography_model.linear_inverse_process_tomography(self.prob)

    @register_browser_function(available_after=(run,))
    def plot(self):
        self.analyze_data()
        self.model.plot_process_matrix(self.ptm, base=self.base)
        plt.show()
