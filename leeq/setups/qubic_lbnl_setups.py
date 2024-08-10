import copy
from http.client import CannotSendRequest
from typing import List, Union, Dict, Any
from uuid import UUID

import numpy as np

from leeq.compiler.lbnl_qubic.circuit_list_compiler import QubiCCircuitListLPBCompiler
from leeq.compiler.lbnl_qubic.utils import register_leeq_pulse_shapes_to_qubic_pulse_shape_factory
from leeq.core.context import ExperimentContext
from leeq.core.engine.grid_sweep_engine import GridSerialSweepEngine, GridBatchSweepEngine
from leeq.core.engine.measurement_result import MeasurementResult
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.experiments.sweeper import Sweeper
from leeq.setups.setup_base import ExperimentalSetup
from leeq.utils import setup_logging

from urllib.parse import urlparse
from pprint import pprint

logger = setup_logging(__name__)


class QubiCCircuitSetup(ExperimentalSetup):
    """
    The QubiCCircuitSetup class defines a setup for using the qubic system. Any setup that uses the qubic system should
    inherit from this class.
    """

    def __init__(
            self,
            name: str,
            runner: Any,
            channel_configs: Any = None,
            fpga_config: Any = None,
            leeq_channel_to_qubic_channel: Dict[str, str] = None,
            qubic_core_number: int = 8,
            batch_process=True,
    ):
        """
        Initialize the QubiCCircuitSetup class.

        Parameters:
            name (str): The name of the setup.
            fpga_config (dict): The configuration of the FPGA for qubic systetm. Refer to QubiC documents.
            channel_configs (dict): The configuration of the channels for qubic system. Refer to QubiC documents.
            runner (Any): The runner to use. Refer to QubiC documents.
            leeq_channel_to_qubic_channel (Dict[str, str]): The mapping from leeq channel to qubic channel.
            qubic_core_number (int): The number of qubic cores to use.
            batch_process (bool): Whether to use batch processing engine.
        """
        self._fpga_config = fpga_config
        self._channel_configs = channel_configs
        self._qubic_core_number = qubic_core_number
        self._leeq_channel_to_qubic_channel = leeq_channel_to_qubic_channel
        self._qubic_channel_to_leeq_channel = None

        self._build_qubic_config()

        assert self._leeq_channel_to_qubic_channel is not None, "Please specify leeq channel to qubic channel config " \
                                                                "if you specified custom qubic channel config."

        self._runner = runner

        self._compiler = QubiCCircuitListLPBCompiler(
            leeq_channel_to_qubic_channel=self._leeq_channel_to_qubic_channel
        )

        if batch_process:
            self._engine = GridBatchSweepEngine(
                compiler=self._compiler, setup=self, name=name + ".engine"
            )
        else:
            self._engine = GridSerialSweepEngine(
                compiler=self._compiler, setup=self, name=name + ".engine"
            )

        self._current_context = None
        self._measurement_results: List[MeasurementResult] = []

        self._result = None

        super().__init__(name)

        self._status.add_param("QubiC_Debug_Print_Circuits", False)
        self._status.add_param(
            "QubiC_Debug_Print_Compiled_Instructions", False)
        self._status.add_param("Engine_Batch_Size", 1)
        self._status.add_param("QubiC_Trace_Acquisition_Decimator", 0)
        self._status.add_param("QubiC_Trace_Acquisition_Extra_Measure_Length", 0)

        # Add channels
        for i in range(qubic_core_number):
            qubit_channel = 2 * i
            readout_channel = 2 * i + 1
            self._status.add_channel(qubit_channel)
            self._status.add_channel(readout_channel)

    def _build_qubic_config(self):
        """
        Build the channel config and FPGA config for QubiC
        """
        tc, qc, FPGAConfig, load_channel_configs, ChannelConfig = self._load_qubic_package()

        if self._fpga_config is None:
            self._fpga_config = FPGAConfig()  # Use the default config
        if self._channel_configs is None:
            self._leeq_channel_to_qubic_channel, self._channel_configs = self._build_default_channel_config(
                core_number=self._qubic_core_number)

        self._qubic_channel_to_leeq_channel = dict(
            [(val, key) for key, val in self._leeq_channel_to_qubic_channel.items()])

        if isinstance(self._channel_configs, dict):
            self._channel_configs = load_channel_configs(self._channel_configs)

    @staticmethod
    def _build_default_channel_config(core_number: int = 8):
        """
        Build the default channel configs for qubic. Here we set the core
        """

        _template_dict = {
            ".qdrv": {
                "core_ind": 7,
                "elem_ind": 0,
                "elem_type": "rf",
                "elem_params": {
                    "samples_per_clk": 16,
                    # "interp_ratio": 1
                    "interp_ratio": 4
                },
                "env_mem_name": "qdrvenv{core_ind}",
                "freq_mem_name": "qdrvfreq{core_ind}",
                "acc_mem_name": "accbuf{core_ind}"
            },

            ".rdrv": {
                "core_ind": 7,
                "elem_ind": 1,
                "elem_type": "rf",
                "elem_params": {
                    "samples_per_clk": 16,
                    "interp_ratio": 16
                },
                "env_mem_name": "rdrvenv{core_ind}",
                "freq_mem_name": "rdrvfreq{core_ind}",
                "acc_mem_name": "accbuf{core_ind}"
            },

            ".rdlo": {
                "core_ind": 7,
                "elem_ind": 2,
                "elem_type": "rf",
                "elem_params": {
                    "samples_per_clk": 4,
                    "interp_ratio": 4
                },
                "env_mem_name": "rdloenv{core_ind}",
                "freq_mem_name": "rdlofreq{core_ind}",
                "acc_mem_name": "accbuf{core_ind}"
            }
        }

        channel_config_dict = {}
        # 2ns per clock cycle for ZCU216 by default
        channel_config_dict["fpga_clk_freq"] = 500e6

        leeq_channel_to_qubic_map = {}

        for i in range(core_number):
            # For the old convention we always set the odd number to resonators
            # and even number for qubits
            leeq_channel_qubit = 2 * i
            leeq_channel_readout = 2 * i + 1
            leeq_channel_to_qubic_map[leeq_channel_qubit] = f'Q{i}'
            leeq_channel_to_qubic_map[leeq_channel_readout] = f'Q{i}'

            # Now setup the channels
            for key, val in _template_dict.items():
                qubic_channel = f'Q{i}' + key
                qubic_channel_setup = copy.deepcopy(val)
                qubic_channel_setup['core_ind'] = i
                channel_config_dict[qubic_channel] = qubic_channel_setup

        return leeq_channel_to_qubic_map, channel_config_dict

    @staticmethod
    def _load_qubic_package():
        """
        Validate the qubic installation by importing the packages.
        """
        try:
            # QubiC toolchain for compiling circuits
            import qubic.toolchain as tc

            # QubiC configuration management libraries
            import qubitconfig.qchip as qc
            from distproc.hwconfig import FPGAConfig, load_channel_configs, ChannelConfig

        except ImportError:
            raise ImportError(
                "Importing QubiC toolchain failed. Please install the QubiC toolchain first."
                " Refer to https://gitlab.com/LBL-QubiC")

        return tc, qc, FPGAConfig, load_channel_configs, ChannelConfig

    def run(self, lpb: LogicalPrimitiveBlock, sweep: Sweeper):
        """
        Run the experiment.

        The experiment run iterates all the parameters described by the sweeper. Each iteration can be break into
         four steps:

        1. Compile the measurement lpb to instructions that going to be passed to the compiler.
        2. Upload the instruction to the compiler, including changing frequencies of the generators etc. Get everything
            ready for the experiment.
        3. Fire the experiment and wait for it to finish.
        4. Collect the data from the compiler and commit it to the measurement primitives.

        So the setup should implement the following methods:
        2. `update_setup_parameters`: Update the setup parameters of the compiler. Usually this function calculates
            the frequencies of the generators etc and pass it to specific experiment setup class to further upload.
        3. `fire_experiment`: Fire the experiment and wait for it to finish.
        4. `collect_data`: Collect the data from the compiler and commit it to the measurement primitives.

        Note that the collected data will be committed to the measurement primitives by the engine, so the setup
            should not commit the data to the measurement primitives.

        The compiler should mainly implement the compile_lpb, like compiling the lpb into a format acceptable by the
        particular setup. This abstraction is to allow the compiler to be used in different setups. For example, the
        compiler can be used in a local setup, or a remote setup.

        Parameters:
            lpb (LogicalPrimitiveBlock): The logical primitive block to run.
            sweep (Sweeper): The sweeper to use.
        """
        return self._engine.run(lpb, sweep)

    def update_setup_parameters(self, context: ExperimentContext):
        """
        Update the setup parameters of the compiler. It accepts the compiled instructions from the compiler, and update
        the local cache first. then use push_instrument_settings to push the settings to the instruments.

        Parameters:
            context (Any): The context between setup and compiler. Generated by the compiler.
        """
        pass

    @staticmethod
    def _get_measurement_info_from_compiled_instructions_for_trace_acquisition(compiled_instructions: dict):
        """
        Get the starting time of the measurement from the compiled instructions.

        Parameters:
            compiled_instructions (dict): The compiled instructions.

        Returns:
            float: The starting time of the measurement.
        """

        measure_start_time = -1
        measurement_length = -1
        channel_demodulation_config = {}
        for proc_group, instructions in compiled_instructions.program.items():
            for instruction in instructions:
                if instruction['op'] == 'pulse' and 'rdlo' in instruction["dest"]:

                    if measure_start_time != -1:
                        assert measure_start_time == instruction["start_time"], ("All the measurement should start "
                                                                                 "at the same time for trace acquisition.")
                    measurement_length = max(instruction["env"]["paradict"]['twidth'], measurement_length)
                    measure_start_time = instruction["start_time"]

                    channel = instruction["dest"].split('.')[0]
                    config = {
                        'freq': instruction["freq"],
                        'start_time': instruction["start_time"],
                        'env': instruction["env"]
                    }
                    channel_demodulation_config[channel] = config

        measure_start_time = measure_start_time / 500e6  # Convert to seconds
        return measure_start_time, measurement_length, channel_demodulation_config

    def _software_demodulation(self, data: np.ndarray, channel_demodulation_config: dict):
        """
        Demodulate the data using the software demodulation.

        Parameters:
            data (np.ndarray): The data to demodulate.
            channel_demodulation_config (dict): The demodulation configuration.

        Returns:
            np.ndarray: The demodulated data.
        """

        demodulated_data = {}
        for channel, config in channel_demodulation_config.items():
            freq = config['freq']
            start_time = config['start_time'] / 500e6  # Convert to seconds

            # Generate the time axis
            decimator = self._status.get_parameters("QubiC_Trace_Acquisition_Decimator")
            samples_per_second = 2e9 / (decimator + 1)
            time_axis = np.arange(data.shape[1]) / samples_per_second + start_time
            demodulation_lo = np.exp(-2j * np.pi * freq * time_axis)

            # There is a bug in the QubiC system that the first 4 samples may not be valid and always 32767 (max value).
            # We need to ignore 4 samples.
            demodulation_lo[:4] = 0

            # Demodulate the data
            demodulated_data[channel] = data * demodulation_lo

        return demodulated_data

    def _run_qubic_circuits(
            self,
            circuits: list,
            batch_size: int,
            zero: bool,
            load_commands: bool,
            load_freqs: bool,
            load_envs: bool):
        """
        Run the qubic circuits.

        Parameters:
            circuits : The asm program to run.
            batch_size (int): The batch size to run.
            zero (bool): Whether to zero out the command buffer before loading the circuit.
            load_commands (bool): Whether to load the commands.
            load_freqs (bool): Whether to load the frequencies.
            load_envs (bool): Whether to load the envelopes.

        Returns:
            dict: The result of the run.
        """

        tc, qc, FPGAConfig, load_channel_configs, ChannelConfig = self._load_qubic_package()

        if self._status.get_parameters("QubiC_Debug_Print_Circuits"):
            pprint(circuits)

        # Register leeq pulse shapes to QubiC
        register_leeq_pulse_shapes_to_qubic_pulse_shape_factory()

        compiled_instructions = tc.run_compile_stage(
            circuits, fpga_config=self._fpga_config, qchip=None, compiler_flags={'resolve_gates': False}
        )

        if self._status.get_parameters(
                "QubiC_Debug_Print_Compiled_Instructions"):
            pprint(compiled_instructions.program)

        asm_prog = tc.run_assemble_stage(
            compiled_instructions, self._channel_configs)

        acquisition_type = self._status.get_parameters("Acquisition_Type")
        n_total_shots = self._status.get_parameters("Shot_Number")

        assert acquisition_type in [
            "IQ",
            "IQ_average",
            "traces",
        ], "Acquisition type should be either IQ or traces. Got " + str(
            acquisition_type
        )

        if acquisition_type == "IQ" or acquisition_type == "IQ_average":

            while True:
                try:
                    self._result = self._runner.run_circuit_batch(executables=[asm_prog],
                                                                  n_total_shots=n_total_shots,
                                                                  # timeout_per_shot,
                                                                  reload_cmd=load_commands,
                                                                  reload_freq=load_freqs,
                                                                  reload_env=load_envs,
                                                                  zero_between_reload=zero)
                    break
                except CannotSendRequest as e:
                    logger.warning(f"Http request cannot be sent. Retrying...{e}")
                    continue

        elif acquisition_type == "traces":

            measure_start_time, measurement_length, channel_demodulation_config = (
                self._get_measurement_info_from_compiled_instructions_for_trace_acquisition(
                    compiled_instructions))

            decimator = self._status.get_parameters("QubiC_Trace_Acquisition_Decimator")
            samples_per_second = 2e9 / (decimator + 1)
            trig_delay = measure_start_time

            # Add extra measurement length
            extra_measure_length = self._status.get_parameters("QubiC_Trace_Acquisition_Extra_Measure_Length") * 1e-6

            measurement_samples = int(np.ceil((measurement_length + extra_measure_length) * samples_per_second))

            if trig_delay > 131.072e-6:
                msg = f"Trig delay {trig_delay} is too large. It should be less than 131.072e-6."
                logger.error(msg)
                raise ValueError(msg)

            # load_and_run_acq is to load the program given by raw_asm_prog and acquire raw
            # adc traces.
            assert batch_size == 1, "Batch size should be 1 for traces acquisition."
            data = self._runner.load_and_run_acq(
                raw_asm_prog=asm_prog,  # The compiled program
                n_total_shots=n_total_shots,  # number of shots to run. Program is restarted from
                # the beginning for each new shot
                nsamples=measurement_samples,  # number of samples to read from the acq buffer
                acq_chans={'0': '0'},
                # current channel mapping is:
                # '0': ADC_237_2 (main readout ADC)
                # '1': ADC_237_0 (other ADC connected in gateware)
                trig_delay=trig_delay,  # delay between trigger and start of acquisition
                # time to delay acquisition, relative to circuit start.
                # NOTE: this value, in units of clock cycles, is a 16-bit value. So, it
                # maxes out at CLK_PERIOD*(2**16) = 131.072e-6
                decimator=int(self._status.get_parameters("QubiC_Trace_Acquisition_Decimator"))
                # decimation interval when sampling. e.g. 0 means full sample rate, 1
                # means capture every other sample, 2 means capture every third sample, etc
            )

            # Demodulate the data
            demodulated_result = self._software_demodulation(data['0'], channel_demodulation_config)

            qubic_channel_channel_to_core = dict([(val, key) for key, val in self._core_to_channel_map.items()])

            self._result = {qubic_channel_channel_to_core[channel]: demodulated_result[channel] for channel in
                            demodulated_result.keys()}

    def fire_experiment(self, context=None):
        """
        Fire the experiment and wait for it to finish.
        """
        self._measurement_results.clear()
        self._result = {}

        shot_interval = self._status.get_parameters("Shot_Period") * 1e-6
        delay_between_shots = [
            {"name": "delay", "t": shot_interval}
        ]

        dirtiness = context.instructions["dirtiness"]

        acquisition_type = self._status.get_parameters("Acquisition_Type")
        circuit = context.instructions["circuits"]

        # For traces acquisition, we need to add the delay between shots, becuase the QubiC system will stall
        # until the previous acquisition is returned to python, which is much longer than the shot interval.
        if acquisition_type != "traces":
            circuit = delay_between_shots + circuit

        return self._run_qubic_circuits(
            circuits=circuit,
            batch_size=1,
            zero=bool(np.sum(context.step_no) == 0),
            load_commands=dirtiness['command'],
            load_freqs=dirtiness['frequency'],
            load_envs=dirtiness['envelope']
        )

    def _qubic_result_format_to_leeq_result_format(self, data: np.ndarray):
        """
        Convert the qubic result format to leeq result format.

        Parameters:
            data (np.ndarray): The data to convert.

        Returns:
            np.ndarray: The converted data.
        """
        acquisition_type = self._status.get_parameters("Acquisition_Type")

        if acquisition_type == 'IQ_average':
            data_collect = np.asarray([data.mean(axis=0)])
        elif acquisition_type == 'IQ':
            data_collect = data
        elif acquisition_type == 'traces':
            data_collect = data
        else:
            msg = f'Unsupported acquisition type {acquisition_type}.'
            logger.error(msg)
            raise NotImplementedError(msg)

        return data_collect

    def collect_data(self, context: ExperimentContext):
        """
        Collect the data from the compiler and commit it to the measurement primitives.
        """

        # We accept one readout per channel for now
        qubic_channel_to_lpb_uuid = context.instructions['qubic_channel_to_lpb_uuid']

        for core_ind, data in self._result.items():
            qubic_channel = self._core_to_channel_map[int(core_ind)]

            if qubic_channel not in qubic_channel_to_lpb_uuid:
                # Sometimes qubic returns default data from the board, which we
                # are not measuring
                continue

            mprim_uuid = qubic_channel_to_lpb_uuid[qubic_channel]

            measurement = MeasurementResult(
                step_no=context.step_no,
                # First index is different measurement, second index for data
                data=self._qubic_result_format_to_leeq_result_format(data),
                mprim_uuid=mprim_uuid
            )

            self._measurement_results.append(measurement)

        context.results = self._measurement_results

        return context

    def fire_experiment_batch(self, contexts):
        """
        Fire the experiment and wait for it to finish. This method concatenate the circuits and build a long circuits
        each time, where the results for each individual circuits are from the intermediate measurement results.

        Parameters:
            contexts (List[ExperimentContext]): The contexts to run.
        """

        shot_interval = self._status.get_parameters("Shot_Period") * 1e-6
        delay_between_shots = [
            {"name": "delay", "t": shot_interval},
            {"name": "barrier"}
        ]

        # Work out the combined circuits
        combined_circuits = []

        # For traces acquisition, we need to add the delay between shots, becuase the QubiC system will stall
        # until the previous acquisition is returned to python, which is much longer than the shot interval.
        acquisition_type = self._status.get_parameters("Acquisition_Type")
        if acquisition_type == 'traces':
            assert len(contexts) == 1, """Traces acquisition only supports batch size 1. Use 
                setup().status().set_param('Engine_Batch_Size',1) to set the batch size to 1."""
            combined_circuits = contexts[0].instructions["circuits"]
        else:
            for context in contexts:
                combined_circuits += delay_between_shots + \
                                     context.instructions["circuits"]

        # The dirtiness is true when at least one of the circuits is dirty
        merged_dirtiness = {
            k: any([context.instructions["dirtiness"][k] for context in contexts]) for k in
            contexts[0].instructions["dirtiness"].keys()
        }

        return self._run_qubic_circuits(
            circuits=combined_circuits,
            batch_size=len(contexts),
            zero=bool(np.sum(contexts[0].step_no) == 0),
            load_commands=merged_dirtiness['command'],
            load_freqs=merged_dirtiness['frequency'],
            load_envs=merged_dirtiness['envelope']
        )

    def collect_data_batch(self, contexts):
        """
        Collect the data from the compiler and commit it to the measurement primitives. This method splits the results
        into individual results and commit them to the individual contexts.

        Parameters:
            contexts (List[ExperimentContext]): The contexts to run.

        Returns:
            List[ExperimentContext]: The contexts with the results.
        """

        # Validate that all the context has the same qubic_channel_to_lpb_uuid
        qubic_channel_to_lpb_uuid = contexts[0].instructions['qubic_channel_to_lpb_uuid']

        for context in contexts[1:]:
            assert context.instructions['qubic_channel_to_lpb_uuid'] == qubic_channel_to_lpb_uuid, \
                "All the contexts should have the same channel mapping. You might be using different mprims in a batch" \
                "run setup, which is not supported yet."

        qubic_channel_to_lpb_uuid = contexts[0].instructions['qubic_channel_to_lpb_uuid']

        for context in contexts:
            context.results = []

        acquisition_type = self._status.get_parameters("Acquisition_Type")
        for channel_name, data in self._result.items():

            if '.rdrv' not in channel_name:
                continue

            qubic_channel = channel_name.split('.')[0]
            if qubic_channel not in qubic_channel_to_lpb_uuid:
                # Sometimes qubic returns default data from the board, which we
                # are not measuring
                continue

            mprim_uuid = qubic_channel_to_lpb_uuid[qubic_channel]

            data = data[0, :, :]  # We only take the outcome of the first circuit from the batch

            for i in range(len(contexts)):
                context = contexts[i]

                # data shape = (n_samples, n_measurement_number)

                # Reshape the data to be 2D array, assume 1 measurement
                # each circuit

                if acquisition_type == 'traces':
                    # For traces acquisition, there is always only one measurement
                    data_step = data
                else:
                    data_step = data[:, i].reshape((-1, 1))

                measurement = MeasurementResult(
                    step_no=context.step_no,
                    # First index is different measurement, second index for
                    # data
                    data=self._qubic_result_format_to_leeq_result_format(
                        data_step),
                    mprim_uuid=mprim_uuid
                )

                context.results.append(measurement)

        return contexts


class QubiCSingleBoardRemoteRPCSetup(QubiCCircuitSetup):
    def __init__(
            self,
            name: str,
            rpc_uri: str,
            channel_configs: Any = None,
            fpga_config: Any = None,
            leeq_channel_to_qubic_channel: Dict[str, str] = None,
            qubic_core_number: int = 8
    ):
        try:
            from qubic.rpc_client import CircuitRunnerClient
        except ImportError:
            msg = "Importing QubiC RPC client failed. Please install the QubiC toolchain first."
            logger.error(msg)
            raise ImportError(msg)

        parsed_uri = urlparse(rpc_uri)

        runner = CircuitRunnerClient(
            ip=parsed_uri.hostname,
            port=parsed_uri.port,
        )

        super().__init__(
            name=name,
            channel_configs=channel_configs,
            runner=runner,
            fpga_config=fpga_config,
            leeq_channel_to_qubic_channel=leeq_channel_to_qubic_channel,
            qubic_core_number=qubic_core_number,
        )
