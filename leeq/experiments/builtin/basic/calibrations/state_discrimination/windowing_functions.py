import itertools
from typing import TYPE_CHECKING, List

import numpy as np
from matplotlib import pyplot as plt

from leeq import Experiment, ExperimentManager, Sweeper, setup
from leeq.chronicle import log_and_record, register_browser_function
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockParallel, LogicalPrimitiveBlockSweep

if TYPE_CHECKING:
    from leeq.core.elements.built_in.qudit_transmon import TransmonElement
    from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock

colors = [
    '#1f77b4',
    '#d62728',
    '#2ca02c',
    '#ff7f0e',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf']


class MeasurementCollectTraces(Experiment):
    @log_and_record
    def run(self,
            duts: 'TransmonElement',
            sweep_lpb_list: List['LogicalPrimitiveBlock'],
            mprim_index: int,
            number_of_traces: int = 100,
            samples_per_seg=256):
        """
        Run the experiment to collect measurement traces for the qubit.

        Parameters:
        dut (TransmonElement): The qubit instance.
        sweep_lpb_list (List[LogicalPrimitiveBlock]): List of LPBs to be included in the sweep.
        mprim_index (int): Index of the measurement primitive in use.
        """

        original_acquisition_type = setup().status().get_param("Acquisition_Type")
        setup().status().set_param("Acquisition_Type", 'traces')
        original_shot_number = setup().status().get_param("Shot_Number")
        setup().status().set_param("Shot_Number", 10)
        total_iteration = np.ceil(number_of_traces / 10)

        swp_iteration = Sweeper(
            np.arange,
            n_kwargs={'start': 0, 'stop': int(total_iteration), 'step': 1},
            params=[]
        )

        self.duts_hrid = [dut.hrid for dut in duts]
        self.samples_per_seg = samples_per_seg
        self.result = None

        # Retrieve the measurement primitive by index
        mprims = [dut.get_measurement_prim_trace(str(mprim_index)) for dut in duts]

        for mprim in mprims:
            # Reset any existing transform function
            mprim.set_transform_function(None)

        # Initialize LPB sweep with provided list
        sweep_lpb = LogicalPrimitiveBlockSweep(children=sweep_lpb_list)

        # Create logical primitive block with parallel processing
        lpb = sweep_lpb + LogicalPrimitiveBlockParallel(mprims)

        # Prepare the sweeper with LPB sweep
        swp = Sweeper.from_sweep_lpb(sweep_lpb)

        # Execute the experiment
        ExperimentManager().run(lpb, swp + swp_iteration)

        # Restore the configuration
        setup().status().set_param("Acquisition_Type", original_acquisition_type)
        setup().status().set_param("Shot_Number", original_shot_number)

        # Format the result and update the class attribute
        results = [np.squeeze(mprim.result()) for mprim in mprims]

        self.results = []

        for result in results:
            # Determine the current size of the last dimension
            current_length = result.shape[-1]

            # Calculate the padding size needed to make the length a multiple of N
            padding_size = (-current_length) % samples_per_seg

            result = result.reshape([result.shape[0], -1, result.shape[-1]])

            # Define the padding configuration.
            # (0, 0) for all dimensions except for the last where it's (0, padding_size)
            padding_config = [(0, 0)] * (result.ndim - 1) + [(0, padding_size)]

            # Pad the array with zeros on the last dimension
            padded_array = np.pad(result, padding_config, mode='constant')

            self.results.append(padded_array)

    def plot_all_analysis(self, index=0):
        """
        Plots the results using Plotly to visualize the assignment matrix.

        Parameters:
            index (int): Index of the qubit to be plotted.
        """
        self.analyze_data()
        self.plot_averaged_traces(index)
        plt.show()
        self.plot_aggregated_trajectories(index)
        plt.show()
        self.plot_difference(index)
        plt.show()

    @register_browser_function()
    def plot_all(self):
        """
        Plots the results using Plotly to visualize the assignment matrix.
        """
        for i in range(len(self.results)):
            self.plot_all_analysis(i)

    def analyze_data(self, samples_per_seg=None):

        if samples_per_seg is None:
            samples_per_seg = self.samples_per_seg

        self.averaged_traces = []
        self.aggregated_traces = []
        self.aggregated_traces_average = []
        self.ts = []

        for result in self.results:
            # Determine the current size of the last dimension
            current_length = result.shape[-1]

            # Calculate the padding size needed to make the length a multiple of N
            padding_size = (-current_length) % samples_per_seg

            result = result.reshape([result.shape[0], -1, result.shape[-1]])

            # Define the padding configuration.
            # (0, 0) for all dimensions except for the last where it's (0, padding_size)
            padding_config = [(0, 0)] * (result.ndim - 1) + [(0, padding_size)]

            # Pad the array with zeros on the last dimension
            padded_array = np.pad(result, padding_config, mode='constant')

            aggregated_traces = padded_array.reshape(
                [padded_array.shape[0], padded_array.shape[1], -1, samples_per_seg]).sum(axis=-1)

            self.averaged_traces.append(np.mean(result, axis=1))
            self.aggregated_traces.append(aggregated_traces)
            self.aggregated_traces_average.append(np.mean(aggregated_traces, axis=1))
            self.ts.append(500e-6 * samples_per_seg * np.arange(aggregated_traces.shape[-1]))  # Hard code for now

        diffs = []

        for traces in self.aggregated_traces_average:
            diff_dict = {}

            for i, j in itertools.combinations(range(traces.shape[0]), 2):
                diff = np.abs(traces[i, :] - traces[j, :])
                diff_dict[rf"$|{i}\rangle$-$|{j}\rangle$"] = diff
            diffs.append(diff_dict)

        self.diffs = diffs

    def plot_averaged_traces(self, index=0):
        traces = self.aggregated_traces_average[index]
        fig = plt.figure(figsize=(traces.shape[0] * 3.5, 3.5))

        for i in range(traces.shape[0]):
            ax = fig.add_subplot(int(f"1{traces.shape[0]}{i + 1}"))
            ax.set_title(rf"{self.duts_hrid[index]} $|{i}\rangle$")
            ax.plot(self.ts[index], traces[i, :].real, color=colors[0], label='I')
            ax.plot(self.ts[index], traces[i, :].imag, color=colors[1], label='Q')
            ax.set_xlabel(r'Time [$\mu s$]')
            ax.set_ylabel(r'Amplitude [a.u.]')
            ax.legend()
        fig.tight_layout()
        return fig

    def plot_aggregated_trajectories(self, index=0):
        traces = self.aggregated_traces_average[index]
        fig = plt.figure(figsize=(2 * 3.5, 3.5))
        colors = [
            '#1f77b4',
            '#d62728',
            '#2ca02c',
            '#ff7f0e',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf']

        ax = fig.add_subplot(int("121"))
        ax.set_title(rf"{self.duts_hrid[index]} Trajectory IQ")
        for i in range(traces.shape[0]):
            ax.plot(traces[i, :].real, traces[i, :].imag, color=colors[i], label=rf"$|{i}\rangle$")
        ax.set_xlabel(r'I Channel [a.u.]')
        ax.set_ylabel(r'Q Channel [a.u.]')
        ax.legend()

        ax = fig.add_subplot(int("122"))
        ax.set_title(r"Accumulated Trajectory IQ")
        for i in range(traces.shape[0]):
            ax.plot(np.cumsum(traces[i, :].real), np.cumsum(traces[i, :].imag), color=colors[i],
                    label=rf"$|{i}\rangle$")
        ax.set_xlabel(r'I Channel [a.u.]')
        ax.set_ylabel(r'Q Channel [a.u.]')
        ax.legend()

        fig.tight_layout()

        return fig

    def plot_difference(self, index=0):
        self.aggregated_traces_average[index]
        fig = plt.figure(figsize=(2 * 3.5, 3.5))

        ax = fig.add_subplot(int("121"))
        ax.set_title(rf"{self.duts_hrid[index]} Traces Difference")

        for i, (key, val) in enumerate(self.diffs[index].items()):
            ax.plot(self.ts[index], val, color=colors[i], label=key)
        ax.set_xlabel(r'Time [$\mu s$]')
        ax.set_ylabel(r'Difference [a.u.]')
        ax.legend()

        fig.tight_layout()

        return fig
