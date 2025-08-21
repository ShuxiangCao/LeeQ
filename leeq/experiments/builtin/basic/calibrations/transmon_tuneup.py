from leeq import Experiment
from leeq.chronicle import log_and_record
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockParallel
from leeq.utils import setup_logging

logger = setup_logging(__name__)

__all__ = ['MultilevelTransmonTuneup']


class MultilevelTransmonTuneup(Experiment):
    """
    A class that extends Experiment to perform a multilevel transmon tuneup experiment.
    The experiment tunes up a multilevel transmon qubit to optimize its performance.

    """

    @log_and_record
    def run(self, duts,
            rabi_amplitude_calibration,
            ramsey_frequency_calibration,
            ramsey_tomography_frequency_calibration,
            pingpong_calibration,
            drag_calibration,
            highest_level=2,
            two_photon_transition=False,
            measurement_calibration=True
            ) -> None:
        """
        Executes the experiment, tuning
        up a multilevel transmon qubit to optimize its performance.
        """

        self.duts = duts
        self.rabi_amplitude_calibration = rabi_amplitude_calibration
        self.ramsey_frequency_calibration = ramsey_frequency_calibration
        self.ramsey_tomography_frequency_calibration = ramsey_tomography_frequency_calibration
        self.pingpong_calibration = pingpong_calibration
        self.drag_calibration = drag_calibration
        self.measurement_calibration = measurement_calibration

        self.highest_level = highest_level

        single_photon_transitions = [f'f{i}{i + 1}' for i in range(highest_level - 1)]
        two_photon_transitions = [f'f{i}{i + 2}' for i in range(highest_level - 2)]

        available_transitions = single_photon_transitions
        if two_photon_transition:
            available_transitions += two_photon_transitions

        self._current_iteration_experiments = []
        self._experiments = []
        self.intermediate_step_configuraiton = []

        for transition in available_transitions:
            self.calibrate_single_transition(transition)

    def _get_initial_lpbs(self, transition_name: str) -> LogicalPrimitiveBlockParallel:
        """
        Get the initial logical primitive blocks for a given transition. Populates the transmon
        to the lower state of the transition.

        Parameters:
            transition_name (str): The name of the transition.

        Returns:
            lpb: LogicalPrimitiveBlock: The initial logical primitive block.
        """

        initial_state = transition_name[1]

        if initial_state == '0':
            return LogicalPrimitiveBlockParallel(
                [dut.get_gate('I', transition_name='f01') for dut in self.duts])
        elif initial_state == '1':
            return LogicalPrimitiveBlockParallel(
                [dut.get_gate('X', transition_name='f01') for dut in self.duts])
        elif initial_state == '2':
            return LogicalPrimitiveBlockParallel(
                [dut.get_gate('X', transition_name='f01') + dut.get_gate('X', transition_name='f12') for dut in
                 self.duts])
        else:
            msg = f'Initial state {initial_state} not supported.'
            logger.error(msg)
            raise ValueError(msg)

    def calibrate_measurement(self, mprim_index) -> None:
        """
        Calibrates the measurement primitive for a single transition.

        Parameters:
            mprim_index (int): The index of the measurement primitive.
        """
        from leeq.experiments.builtin import MeasurementCalibrationMultilevelGMM

        lpb_scan = (
            LogicalPrimitiveBlockParallel([dut.get_c1('f01')['I'] for dut in self.duts]),
            LogicalPrimitiveBlockParallel([dut.get_c1('f01')['X'] for dut in self.duts]),
            LogicalPrimitiveBlockParallel([
                dut.get_c1('f01')['X'] + dut.get_c1('f12')['X']
                for dut in self.duts]),
            LogicalPrimitiveBlockParallel([
                dut.get_c1('f01')['X'] + dut.get_c1('f12')['X'] + dut.get_c1('f23')['X']
                for dut in self.duts]),
        )

        highest_level = int(mprim_index) + 1

        calibs = []
        for dut in self.duts:
            calib = MeasurementCalibrationMultilevelGMM(dut, mprim_index=0, sweep_lpb_list=lpb_scan[:highest_level + 1])
            calibs.append(calib)

        self._current_iteration_experiments.append(calibs)

    def calibrate_rabi_amplitude(self, transition_name: str) -> None:
        """
        Calibrates the Rabi amplitude for a single transition.

        Parameters:
            transition_name (str): The name of the transition to calibrate.
        """

        mprim_index = transition_name[1]

        from leeq.experiments.builtin import MultiQubitRabi
        rabi = MultiQubitRabi(
            duts=self.duts,
            amps=None,
            start=0.01,
            stop=0.15,
            step=0.005,
            collection_names=transition_name,
            mprim_indexes=mprim_index,
            pulse_discretization=True,
            update=True,
            initial_lpb=self._get_initial_lpbs(transition_name)
        )

        self._current_iteration_experiments.append(rabi)

    def calibrate_ramsey(self, transition_name: str) -> None:
        """
        Calibrates the Ramsey frequency for a single transition.

        Parameters:
            transition_name (str): The name of the transition to calibrate.
        """

        initial_lpb = self._get_initial_lpbs(transition_name)
        mprim_index = transition_name[1]

        collection_names = [transition_name] * len(self.duts)
        mprim_indexes = [mprim_index] * len(self.duts)

        from leeq.experiments.builtin import MultiQubitRamseyMultilevel
        ramsey_10 = MultiQubitRamseyMultilevel(duts=self.duts, set_offset=10, stop=0.3, step=0.005,
                                               mprim_indexes=mprim_indexes,
                                               collection_names=collection_names, initial_lpb=initial_lpb)
        ramsey_1 = MultiQubitRamseyMultilevel(duts=self.duts, set_offset=1, stop=3, step=0.05,
                                              mprim_indexes=mprim_indexes,
                                              collection_names=collection_names, initial_lpb=initial_lpb)
        ramsey_d1 = MultiQubitRamseyMultilevel(duts=self.duts, set_offset=0.1, stop=30, step=0.5,
                                               mprim_indexes=mprim_indexes,
                                               collection_names=collection_names, initial_lpb=initial_lpb)

        self._current_iteration_experiments.append([ramsey_10, ramsey_1, ramsey_d1])

    def calibrate_ramsey_tomography(self, transition_name: str) -> None:
        """
        Calibrates the Ramsey tomography frequency for a single transition.

        Parameters:
            transition_name (str): The name of the transition to calibrate.
        """
        raise NotImplementedError()
        pass

    def calibrate_pingpong(self, transition_name: str) -> None:
        """
        Calibrates the pingpong frequency for a single transition.

        Parameters:
            transition_name (str): The name of the transition to calibrate.
        """
        initial_lpb = self._get_initial_lpbs(transition_name)

        from leeq.experiments.builtin import AmpTuneUpMultiQubitMultilevel
        pingpong = AmpTuneUpMultiQubitMultilevel(
            duts=self.duts, mprim_indexes=[transition_name[1]] * len(self.duts),
            collection_names=[transition_name] * len(self.duts), initial_lpb=initial_lpb
        )

        self._current_iteration_experiments.append(pingpong)

    def calibrate_drag(self, transition_name: str) -> None:
        """
        Calibrates the drag parameter for a single transition.

        Parameters:
            transition_name (str): The name of the transition to calibrate.
        """
        raise NotImplementedError()
        pass

    def calibrate_single_transition(self, transition_name: str) -> None:
        """
        Calibrates a single transition

        Args:
            transition_name (str): The name of the transition to calibrate.

        """
        duts = self.duts

        # The procedure follows the following pattern. It checks if transition name in the arguments. If it is, it
        # calibrates the transition.

        mprim_index = int(transition_name[1])

        def measurement_gmm(): return self.calibrate_measurement(
            mprim_index) if self.measurement_calibration else None

        measurement_gmm()

        if transition_name in self.ramsey_tomography_frequency_calibration:
            self.calibrate_ramsey_tomography(transition_name)
            measurement_gmm()

        if transition_name in self.ramsey_frequency_calibration:
            self.calibrate_ramsey(transition_name)
            measurement_gmm()

        if transition_name in self.rabi_amplitude_calibration:
            self.calibrate_rabi_amplitude(transition_name)
            measurement_gmm()

        if transition_name in self.pingpong_calibration:
            self.calibrate_pingpong(transition_name)
            measurement_gmm()

        if transition_name in self.drag_calibration:
            self.calibrate_drag(transition_name)
            measurement_gmm()

        self._experiments.append((transition_name, self._current_iteration_experiments))

        calibration_logs = [
            dut.get_calibrations() for dut in duts
        ]

        self.intermediate_step_configuraiton.append((transition_name, calibration_logs))
