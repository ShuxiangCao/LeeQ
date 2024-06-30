import math
from labchronicle import log_and_record, register_browser_function
import uncertainties as unc
import uncertainties.unumpy as unp

from leeq import Experiment, Sweeper, basic_run
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.utils.compatibility import *

import matplotlib.pyplot as plt
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial, LogicalPrimitiveBlock

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Optional, Any, Tuple, Union

from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.utils.compatibility import prims

from leeq.utils import setup_logging
from leeq.utils.ai.vlms import visual_analyze_prompt

logger = setup_logging(__name__)

__all__ = [
    'PingPongSingleQubitMultilevel',
    'PingPongMultiQubitMultilevel',
    'AmpTuneUpSingleQubitMultilevel',
    'AmpTuneUpMultiQubitMultilevel'
]


class PingPongSingleQubitMultilevel(Experiment):
    """
    Class representing a Ping Pong experiment with a single qubit in a multilevel setup.

    Attributes:
        pulse_count (int): Number of pulses in the experiment.
        amplitude (float): Current amplitude of the repeated block.
    """

    @log_and_record(overwrite_func_name='PingPongSingleQubitMultilevel.run')
    def run_simulated(self,
                      dut: TransmonElement,
                      collection_name: str,
                      initial_lpb: LogicalPrimitiveBlock,
                      repeated_block: LogicalPrimitiveBlock,
                      final_gate: str,
                      initial_gate: str,
                      pulse_count: int,
                      mprim_index: Optional[int] = 0, ) -> None:
        """
        Runs the ping pong single qubit experiment with high-level simulator.

        Parameters:
            dut: Device under test.
            collection_name: Name of the lpb collection.
            mprim_index: Index of the primary element.
            initial_lpb: Initial lower pulse block.
            initial_gate: Initial gate identifier.
            repeated_block: Block of repeated pulses.
            final_gate: Final gate identifier.
            pulse_count: Number of pulses.
        """

        c1 = dut.get_c1(collection_name)
        # Getting amplitude argument from the repeated block
        cur_amp = repeated_block.get_pulse_args('amp')

        # Getting the logical primitive for the initial and final gates
        initial_gate_lpb = c1[initial_gate]
        final_gate_lpb = c1[final_gate]

        # Getting the measurement primitive
        mprim = dut.get_measurement_prim_intlist(mprim_index)

        if initial_lpb is not None:
            logger.warning("initial_lpb is ignored in the simulated mode.")

        simulator_setup: HighLevelSimulationSetup = setup().get_default_setup()
        virtual_transmon = simulator_setup.get_virtual_qubit(dut)

        c1 = dut.get_c1(collection_name)

        # hard code a virtual dut here
        rabi_rate_per_amp = simulator_setup.get_omega_per_amp(
            c1.channel)  # MHz
        omega = rabi_rate_per_amp * cur_amp

        # Detuning
        delta = virtual_transmon.qubit_frequency - c1['X'].freq

        area_per_pulse = c1['X'].calculate_envelope_area()

        t_effective = area_per_pulse * (np.arange(0, pulse_count, 1) + 0.5)

        # Rabi oscillation formula
        self.result = ((omega ** 2) / (delta ** 2 + omega ** 2) *
                       np.sin(0.5 * np.sqrt(delta ** 2 + omega ** 2) * t_effective) ** 2)

        # If sampling noise is enabled, simulate the noise
        if setup().status().get_param('Sampling_Noise'):
            # Get the number of shot used in the simulation
            shot_number = setup().status().get_param('Shot_Number')

            # generate binomial distribution of the result to simulate the
            # sampling noise
            self.result = np.random.binomial(
                shot_number, self.data) / shot_number

        self.pulse_count = pulse_count
        self.amplitude = cur_amp

        self.fit()

    @log_and_record
    def run(self,
            dut: TransmonElement,
            collection_name: str,
            initial_lpb: LogicalPrimitiveBlock,
            repeated_block: LogicalPrimitiveBlock,
            final_gate: str,
            initial_gate: str,
            pulse_count: int,
            mprim_index: Optional[int] = 0,
            ) -> None:
        """
        Runs the ping pong single qubit experiment.

        Parameters:
            dut: Device under test.
            collection_name: Name of the lpb collection.
            mprim_index: Index of the primary element.
            initial_lpb: Initial lower pulse block.
            initial_gate: Initial gate identifier.
            repeated_block: Block of repeated pulses.
            final_gate: Final gate identifier.
            pulse_count: Number of pulses.
        """

        c1 = dut.get_c1(collection_name)
        # Getting amplitude argument from the repeated block
        cur_amp = repeated_block.get_pulse_args('amp')

        # Getting the logical primitive for the initial and final gates
        initial_gate_lpb = c1[initial_gate]
        final_gate_lpb = c1[final_gate]

        # Getting the measurement primitive
        mprim = dut.get_measurement_prim_intlist(mprim_index)

        sequence_lpb = []

        # Pulse count is a list of integers, each one denotes a number of
        # repetitions
        for n in pulse_count:
            sequence = initial_gate_lpb + LogicalPrimitiveBlockSerial(
                [repeated_block] * n) + final_gate_lpb

            if initial_lpb is not None:
                sequence = initial_lpb + sequence

            sequence_lpb.append(sequence)

        lpb = prims.SweepLPB(sequence_lpb)

        swp = Sweeper.from_sweep_lpb(lpb)

        lpb = lpb + mprim

        basic_run(lpb, swp, '<z>')

        self.result = np.squeeze(mprim.result(), axis=-1)
        self.pulse_count = pulse_count
        self.amplitude = cur_amp

        self.fit()

    @staticmethod
    def lin(xvec: np.ndarray, a: float, b: float) -> np.ndarray:
        """Linear function for fitting.

        Args:
            xvec: The x values.
            a: Slope of the line.
            b: Y-intercept of the line.

        Returns:
            Resulting y values.
        """
        return a * xvec + b

    def fit(self):
        """
        Fits the results using a linear function.
        """
        x = self.pulse_count + 0.5
        p, cov = np.polyfit(x, self.result, 1, cov=True)
        self.fit_result = unc.correlated_values(np.squeeze(p), np.squeeze(cov))

    @register_browser_function(
        available_after=(run,))
    def plot(self) -> None:
        """
        Plots the results of the ping pong experiment.
        """

        x = self.pulse_count + 0.5  # Adjusting pulse count for plotting

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.5))

        axes.scatter(x, self.result, alpha=0.5)
        axes.plot(x, self.fit_result[0].nominal_value * x + self.fit_result[1].nominal_value, 'r-')
        axes.set_ylim(-1.1, 1.1)
        axes.set_xlabel(u"Repetition")
        axes.set_ylabel(u"<z>")

        return fig


class AmpTuneUpSingleQubitMultilevel(Experiment):
    """
    This class represents an amplitude tuning experiment for a single qubit multilevel system.
    """

    _experiment_result_analysis_instructions = """
    The experiment is considered successful if the amplitude converges through the iterations.
    """

    @log_and_record
    def run(self,
            dut: TransmonElement,
            iteration: int = 9,
            points: int = 10,
            mprim_index: int = 0,
            collection_name: str = 'f01',
            repeated_gate: str = 'X',
            initial_lpb: Optional[LogicalPrimitiveBlock] = None,
            flip_other: bool = False) -> None:
        """
        Run the experiment  for amplitude finetuning of single qubit pulses repeatedly using pingpong scheme.

        Parameters:
            dut (object): The device under test.
            name (str): The name of the experiment.
            mprim_index (int): The index of the mprim.
            initial_lpb (float): The initial length per block.
            repeated_gate (str): The repeated gate.
            iteration (int): The number of iterations.
            points (int): The number of points to run for each pingpong fit.
            flip_other (bool): Whether to flip the other side.
        """
        factor = 1 if repeated_gate in (
            'X', 'Y') else 2  # Make sure each time we have a full pi rotation

        final_gate = ''
        if repeated_gate in ('X', 'Xp'):
            final_gate = 'Xp'
        elif repeated_gate in ('Y', 'Yp'):
            final_gate = 'Yp'
        elif repeated_gate in ('-X', 'Xm'):
            final_gate = 'Xm'
        elif repeated_gate in ('-Y', 'Ym'):
            final_gate = 'Ym'

        c1 = dut.get_c1(collection_name)

        cur_amp = c1[repeated_gate].primary_kwargs()['amp']

        repeated_block = c1[repeated_gate]

        self.tune_up_results = []
        self.fit_params = []
        self.pulse_counts = []
        self.amps = []

        flip = [False] if not flip_other else [False, True]

        for t in flip:
            reps = 8 * factor
            for i in range(iteration):
                interval = math.ceil(reps / points)
                interval += (interval %
                             2)  # round up interval to an even number

                interval = max(interval, 2 * factor)

                pulse_count = np.arange(0, reps, interval)

                trial = PingPongSingleQubitMultilevel(
                    dut=dut,
                    collection_name=collection_name,
                    mprim_index=mprim_index,
                    initial_lpb=initial_lpb,
                    initial_gate='I',
                    repeated_block=repeated_block,
                    final_gate=final_gate,
                    pulse_count=pulse_count)

                k, b = trial.fit_result
                self.error = k
                self.tune_up_results.append(trial.result)
                self.fit_params.append(trial.fit_result)
                self.pulse_counts.append(trial.pulse_count)

                correction = unp.arcsin(self.error) / np.pi * factor
                correction_factor = 1 + correction

                transition_photon_number = int(
                    collection_name[2]) - int(collection_name[1])

                if transition_photon_number > 1:
                    raise NotImplementedError(
                        "Not implemented for transition photon number > 1")
                else:
                    cur_amp = cur_amp * correction_factor

                print(f"Estimated best amplitude {cur_amp}")
                c1[repeated_gate].update_parameters(amp=cur_amp.nominal_value)
                self.amps.append(cur_amp)
                cur_amp = cur_amp.nominal_value
                reps *= 2

        self.iteration = iteration * len(flip)
        self.best_amp = self.amps[-1]
        c1.update_parameters(amp=self.best_amp.nominal_value)

    @register_browser_function(available_after=(run,))
    @visual_analyze_prompt("""
    Please confirm if the amplitude converges through the iterations. If convergence is not achieved, the experiment
    is likely to be unsuccessful. Please check the amplitude plot to confirm the convergence.
    """)
    def plot_amp(self) -> plt.Figure:
        """
        Plot the amplitude over the iterations.

        Return:
            plt.Figure: The figure object.
        """
        fig = plt.figure(figsize=(4, 2.5))
        ax = fig.add_subplot(111)
        ax.errorbar(range(self.iteration), [x.nominal_value for x in self.amps], yerr=[x.std_dev for x in self.amps])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Amplitude [a.u.]')
        ax.set_title('Updated amplitude over iterations')

        return fig


class PingPongMultiQubitMultilevel(Experiment):
    """
    Class representing a Ping Pong experiment with a multi qubit in a multilevel setup.

    Attributes:
        pulse_count (int): Number of pulses in the experiment.
        amplitude (float): Current amplitude of the repeated block.
    """

    @log_and_record
    def run(self,
            duts: List[TransmonElement],
            collection_names: Union[str, List[str]],
            initial_lpb: LogicalPrimitiveBlock,
            repeated_block: LogicalPrimitiveBlock,
            final_gate: str,
            initial_gate: str,
            pulse_count: int,
            mprim_indexes: Optional[Union[int, List[int]]] = 0,
            ) -> None:
        """
        Runs the ping pong multi qubit experiment.

        Parameters:
            duts: List of devices under test.
            collection_names: List of names of the lpb collection.
            mprim_indexes: Index of the primary element. If a list is provided, the length of the list should be the same
                as the number of qubits.
            initial_lpb: Initial lower pulse block.
            initial_gate: Initial gate identifier.
            repeated_block: Block of repeated pulses.
            final_gate: Final gate identifier.
            pulse_count: Number of pulses.
        """
        if not isinstance(mprim_indexes, list):
            mprim_indexes = [mprim_indexes] * len(duts)

        if not isinstance(collection_names, list):
            collection_names = [collection_names] * len(duts)

        c1s = [dut.get_c1(collection_name)
               for dut, collection_name in zip(duts, collection_names)]

        # Getting the logical primitive for the initial and final gates
        initial_gate_lpb = prims.ParallelLPB([c1[initial_gate] for c1 in c1s])
        final_gate_lpb = prims.ParallelLPB([c1[final_gate] for c1 in c1s])

        # Getting the measurement primitive
        mprims = [
            dut.get_measurement_prim_intlist(mprim_index) for dut,
            mprim_index in zip(
                duts,
                mprim_indexes)]

        sequence_lpb = []

        # Pulse count is a list of integers, each one denotes a number of
        # repetitions
        for n in pulse_count:
            sequence = initial_gate_lpb + LogicalPrimitiveBlockSerial(
                [repeated_block] * n) + final_gate_lpb

            if initial_lpb is not None:
                sequence = initial_lpb + sequence

            sequence_lpb.append(sequence)

        lpb = prims.SweepLPB(sequence_lpb)

        swp = Sweeper.from_sweep_lpb(lpb)

        lpb = lpb + prims.ParallelLPB(mprims)

        basic_run(lpb, swp, '<z>')

        self.results = [mprim.result() for mprim in mprims]
        self.pulse_count = pulse_count

        self.fit()

    @staticmethod
    def lin(xvec: np.ndarray, a: float, b: float) -> np.ndarray:
        """Linear function for fitting.

        Args:
            xvec: The x values.
            a: Slope of the line.
            b: Y-intercept of the line.

        Returns:
            Resulting y values.
        """
        return a * xvec + b

    def fit(self):
        """
        Fits the results using a linear function.
        """
        x = self.pulse_count + 0.5
        self.fit_results = [np.polyfit(x, result, 1)
                            for result in self.results]

    @register_browser_function(
        available_after=(run,))
    def plot_all(self):
        """
        Plots the results of the ping pong experiment.
        """
        for i in range(len(self.results)):
            fig = self.plot(i)
            plt.show()

    def plot(self, i) -> None:
        """
        Plots the results of the ping pong experiment.
        """

        args = self.retrieve_args(self.run)
        duts = args['duts']

        x = self.pulse_count + 0.5  # Adjusting pulse count for plotting

        fig, axes = plt.subplots(nrows=1, ncols=1)

        axes.scatter(x, self.results[i], alpha=0.5)
        axes.plot(x, self.fit_results[i][0] * x + self.fit_results[i][1], 'r-')
        axes.set_ylim(-1.1, 1.1)
        axes.set_xlabel(u"Repetition")
        axes.set_ylabel(u"<z>")
        fig.suptitle(f"Qubit {duts[i].hrid}")
        return fig


class AmpTuneUpMultiQubitMultilevel(Experiment):
    """
    This class represents an amplitude tuning experiment for a multi qubit multilevel system.
    """

    @log_and_record
    def run(self,
            duts: List[TransmonElement],
            iteration: int = 10,
            points: int = 10,
            mprim_indexes: Union[int, List[int]] = 0,
            collection_names: Union[str, List[str]] = 'f01',
            repeated_gate: str = 'X',
            initial_lpb: Optional[LogicalPrimitiveBlock] = None,
            flip_other: bool = False) -> None:
        """
        Run the amplitude tuning experiment.

        Parameters:
            dut (object): The device under test.
            collection_names (Union[str,List[str]]): The name of the experiment. If a list is provided, the length of the
                list should be the same as the number of qubits.
            mprim_indexes (Union[int, List[int]]): The index of the mprim. If a list is provided, the length of the list
                should be the same as the number of qubits.
            initial_lpb (float): The initial length per block.
            repeated_gate (str): The repeated gate.
            iteration (int): The number of iterations.
            points (int): The number of points to run for each pingpong fit.
            flip_other (bool): Whether to flip the other side.
        """

        if not isinstance(mprim_indexes, list):
            mprim_indexes = [mprim_indexes] * len(duts)

        if not isinstance(collection_names, list):
            collection_names = [collection_names] * len(duts)

        # Make sure each time we have a full pi rotation
        factor = 1 if repeated_gate in ('X', 'Y') else 2

        final_gate = ''
        if repeated_gate in ('X', 'Xp'):
            final_gate = 'Xp'
        elif repeated_gate in ('Y', 'Yp'):
            final_gate = 'Yp'
        elif repeated_gate in ('-X', 'Xm'):
            final_gate = 'Xm'
        elif repeated_gate in ('-Y', 'Ym'):
            final_gate = 'Ym'

        c1s = [dut.get_c1(collection_name)
               for dut, collection_name in zip(duts, collection_names)]

        cur_amps = [c1[repeated_gate].primary_kwargs()['amp'] for c1 in c1s]

        repeated_block = prims.ParallelLPB([c1[repeated_gate] for c1 in c1s])

        self.tune_up_results = []
        self.fit_params = []
        self.pulse_counts = []
        self.amps = []

        flip = [False] if not flip_other else [False, True]

        for t in flip:
            reps = 4 * factor
            for i in range(iteration):
                self.amps.append(cur_amps)
                interval = math.ceil(reps / points)
                interval += (interval %
                             2)  # round up interval to an even number

                interval = max(interval, 2 * factor)

                pulse_count = np.arange(0, reps, interval)

                print('pulse_count:', pulse_count)

                trial = PingPongMultiQubitMultilevel(
                    duts=duts,
                    collection_names=collection_names,
                    mprim_indexes=mprim_indexes,
                    initial_lpb=initial_lpb,
                    initial_gate='I',
                    repeated_block=repeated_block,
                    final_gate=final_gate,
                    pulse_count=pulse_count)

                self.tune_up_results.append(trial.results)
                self.fit_params.append(trial.fit_results)
                self.pulse_counts.append(trial.pulse_count)

                for i in range(len(trial.fit_results)):
                    k, b = trial.fit_results[i]
                    c1 = c1s[i]
                    error = k[0]
                    collection_name = collection_names[i]
                    cur_amp = cur_amps[i]

                    correction = np.arcsin(error) / np.pi * factor
                    correction_factor = 1 + correction

                    transition_photon_number = int(
                        collection_name[2]) - int(collection_name[1])

                    if transition_photon_number > 1:
                        raise NotImplementedError(
                            "Not implemented for transition photon number > 1")
                    else:
                        cur_amp *= correction_factor

                    cur_amps[i] = cur_amp

                    print(
                        f"Update {duts[i]} {collection_name} amplitude to {cur_amp}")
                    c1[repeated_gate].update_parameters(amp=cur_amp)
                reps *= 2

        self.iteration = iteration * len(flip)
        self.best_amp = cur_amps

    @register_browser_function(available_after=(run,))
    def plot_all(self):
        """
        Plot the amplitude over the iterations.

        Return:
            None
        """
        for i in range(len(self.result)):
            self.plot(i)

    def plot_amp(self, i) -> None:
        """
        Plot the amplitude over the iterations.

        Return:
            None
        """
        args = self.retrieve_args(self.run)
        duts = args['duts']
        plt.plot(range(self.iteration), [x[i] for x in self.amps])
        plt.xlabel('Iteration')
        plt.ylabel('Amplitude [a.u.]')
        plt.title('Updated amplitude over iterations for ' + duts[i].hrid)

        plt.show()
