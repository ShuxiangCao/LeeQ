import copy
from typing import List, Union, Dict, Any
from uuid import UUID

import numpy as np

from leeq.compiler.lbnl_qubic.circuit_list_compiler import QubiCCircuitListLPBCompiler
from leeq.core.context import ExperimentContext
from leeq.core.engine.grid_sweep_engine import GridSerialSweepEngine
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
            qubic_core_number: int = 8
    ):
        """
        Initialize the QubiCCircuitSetup class.

        Parameters:
            name (str): The name of the setup.
            fpga_config (dict): The configuration of the FPGA for qubic systetm. Refer to QubiC documents.
            channel_configs (dict): The configuration of the channels for qubic system. Refer to QubiC documents.
            runner (Any): The runner to use. Refer to QubiC documents.
            leeq_channel_to_qubic_channel (Dict[str, str]): The mapping from leeq channel to qubic channel.
        """
        self._fpga_config = fpga_config
        self._channel_configs = channel_configs
        self._qubic_core_number = qubic_core_number
        self._leeq_channel_to_qubic_channel = leeq_channel_to_qubic_channel

        self._build_qubic_config()

        assert self._leeq_channel_to_qubic_channel is not None, "Please specify leeq channel to qubic channel config " \
                                                                "if you specified custom qubic channel config."

        self._runner = runner

        self._compiler = QubiCCircuitListLPBCompiler(
            leeq_channel_to_qubic_channel=self._leeq_channel_to_qubic_channel
        )
        self._engine = GridSerialSweepEngine(
            compiler=self._compiler, setup=self, name=name + ".engine"
        )

        self._current_context = None
        self._measurement_results: List[MeasurementResult] = []

        self._result = None
        self._core_to_channel_map = self._build_core_id_to_qubic_channel(self._channel_configs)

        super().__init__(name)

        self._status.add_param("QubiC_Debug_Print_Circuits", False)
        self._status.add_param("QubiC_Debug_Print_Compiled_Instructions", False)

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

        if isinstance(self._channel_configs, dict):
            self._channel_configs = load_channel_configs(self._channel_configs)

    @staticmethod
    def _build_core_id_to_qubic_channel(configs):
        """
        Find the map from core id to qubic channel, for assigning readout result.
        """

        core_to_channel_map = {}

        for name, config in configs.items():
            if 'rdlo' not in name:
                continue

            qubic_channel_name = name.split('.')[0]
            core_ind = config.core_ind
            core_to_channel_map[core_ind] = qubic_channel_name

        return core_to_channel_map

    @staticmethod
    def _build_default_channel_config(core_number: int = 8):
        """
        Build the default channel configs for qubic. Here we set the core
        """

        _template_dict = {
            ".qdrv": {
                "core_ind": 7,
                "elem_ind": 0,
                "elem_params": {
                    "samples_per_clk": 16,
                    "interp_ratio": 1
                },
                "env_mem_name": "qdrvenv{core_ind}",
                "freq_mem_name": "qdrvfreq{core_ind}",
                "acc_mem_name": "accbuf{core_ind}"
            },

            ".rdrv": {
                "core_ind": 7,
                "elem_ind": 1,
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
        channel_config_dict["fpga_clk_freq"] = 500e6  # 2ns per clock cycle for ZCU216 by default

        leeq_channel_to_qubic_map = {}

        for i in range(core_number):
            # For the old convention we always set the odd number to resonators and even number for qubits
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

    def fire_experiment(self, context=None):
        """
        Fire the experiment and wait for it to finish.
        """
        self._measurement_results.clear()
        self._result = {}
        tc, qc, FPGAConfig, load_channel_configs, ChannelConfig = self._load_qubic_package()

        shot_interval = self._status.get_parameters("Shot_Period") * 1e-6
        delay_between_shots = [
            {"name": "delay", "t": shot_interval}
        ]

        if self._status.get_parameters("QubiC_Debug_Print_Circuits"):
            pprint(context.instructions['circuits'])

        compiled_instructions = tc.run_compile_stage(
            delay_between_shots + context.instructions['circuits'], fpga_config=self._fpga_config, qchip=None
        )

        if self._status.get_parameters("QubiC_Debug_Print_Compiled_Instructions"):
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

        dirtiness = context.instructions["dirtiness"]

        if acquisition_type == "IQ" or acquisition_type == "IQ_average":
            self._runner.load_circuit(
                rawasm=asm_prog,
                # if True, (default), zero out all cmd buffers before loading circuit
                zero=True,
                # load command buffers when the circuit or parameters (amp or phase) has changed.
                load_commands=dirtiness['command'],
                # load frequency buffers when frequency changed.
                load_freqs=dirtiness['frequency'],
                # load envelope buffers when envelope changed.
                load_envs=dirtiness['envelope']
            )
            self._result = self._runner.run_circuit(
                n_total_shots=n_total_shots,
                reads_per_shot=1,
                # Number of values per shot per channel to read back from accbuf.
                # Unless there is mid-circuit measurement involved this is typically 1
                # TODO: add support for mid-circuit measurement
                delay_per_shot=0,  # Not used and not functionality
            )
        elif acquisition_type == "traces":
            # load_and_run_acq is to load the program given by raw_asm_prog and acquire raw
            # (or downconverted) adc traces.
            self._result = self._runner.load_and_run_acq(
                raw_asm_prog=asm_prog,  # The compiled program
                n_total_shots=1,  # number of shots to run. Program is restarted from
                # the beginning for each new shot
                nsamples=8192,  # number of samples to read from the acq buffer
                acq_chans=["0"],  # list of channels to acquire
                # current channel mapping is:
                # '0': ADC_237_2 (main readout ADC)
                # '1': ADC_237_0 (other ADC connected in gateware)
                trig_delay=0,  # delay between trigger and start of acquisition
                # time to delay acquisition, relative to circuit start.
                # NOTE: this value, in units of clock cycles, is a 16-bit value. So, it
                # maxes out at CLK_PERIOD*(2**16) = 131.072e-6
                decimator=0,
                # decimation interval when sampling. e.g. 0 means full sample rate, 1
                # means capture every other sample, 2 means capture every third sample, etc
                from_server=False,
                # set to true if calling over RPC. If True, pack returned acq arrays into
                # byte objects.
            )

    def collect_data(self, context: ExperimentContext):
        """
        Collect the data from the compiler and commit it to the measurement primitives.
        """

        acquisition_type = self._status.get_parameters("Acquisition_Type")

        # We accept one readout per channel for now
        qubic_channel_to_lpb_uuid = context.instructions['qubic_channel_to_lpb_uuid']

        for core_ind, data in self._result.items():
            qubic_channel = self._core_to_channel_map[int(core_ind)]

            if qubic_channel not in qubic_channel_to_lpb_uuid:
                # Sometimes qubic returns default data from the board, which we are not measuring
                continue

            mprim_uuid = qubic_channel_to_lpb_uuid[qubic_channel]

            if acquisition_type == 'IQ_average':
                data_collect = np.asarray([data.mean(axis=0)])
            elif acquisition_type == 'IQ':
                data_collect = data.transpose()
            else:
                msg = f'Unsupported acquisition type {acquisition_type}.'
                logger.error(msg)
                raise NotImplementedError(msg)

            measurement = MeasurementResult(
                step_no=context.step_no,
                data=data_collect,  # First index is different measurement, second index for data
                mprim_uuid=mprim_uuid
            )

            self._measurement_results.append(measurement)

        context.results = self._measurement_results

        return context


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
