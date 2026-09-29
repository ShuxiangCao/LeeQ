"""Single-board QubiC setup for the Executable/batched XML-RPC interface."""

import inspect
from urllib.parse import urlparse

import numpy as np

from leeq.compiler.lbnl_qubic.utils import register_leeq_pulse_shapes_to_qubic_pulse_shape_factory
from leeq.setups.qubic_lbnl_setups import QubiCCircuitSetup


class QubiCSingleBoardExecutableRPCSetup(QubiCCircuitSetup):
    """Use modern QubiC clients without changing the legacy RPC setup.

    Construction and ``compile_circuits`` are local operations. Experiment
    execution submits an executable to the existing board RPC server.
    Channel metadata and core count must match the selected firmware.
    """

    def __init__(self, name, rpc_uri, channel_configs,
                 leeq_channel_to_qubic_channel, qubic_core_number,
                 fpga_config=None, batch_process=True):
        uri = urlparse(rpc_uri)
        if (uri.scheme != "http" or not uri.hostname or uri.port is None
                or uri.username or uri.password or uri.path not in ("", "/")
                or uri.query or uri.fragment):
            raise ValueError("rpc_uri must be an HTTP URL with an explicit host and port")
        if not channel_configs or not leeq_channel_to_qubic_channel:
            raise ValueError("Matching channel metadata and a LeeQ channel mapping are required")
        if not isinstance(qubic_core_number, int) or qubic_core_number < 1:
            raise ValueError("qubic_core_number must be a positive integer")

        from qubic.rpc_client import CircuitRunnerClient

        runner = CircuitRunnerClient(ip=uri.hostname, port=uri.port)
        batch_runner = getattr(type(runner), "run_circuit_batch", None)
        if not callable(batch_runner) or "executables" not in inspect.signature(batch_runner).parameters:
            raise TypeError("This setup requires the Executable-based QubiC run_circuit_batch client")
        super().__init__(name=name, runner=runner, channel_configs=channel_configs,
                         fpga_config=fpga_config,
                         leeq_channel_to_qubic_channel=leeq_channel_to_qubic_channel,
                         qubic_core_number=qubic_core_number, batch_process=batch_process)
        if any(core < 0 or core >= qubic_core_number for core in self._core_to_channel_map):
            raise ValueError("Readout channel metadata exceeds the configured core count")
        for channel, config in self._channel_configs.items():
            if not hasattr(config, "core_ind"):
                continue
            if config.board_name:
                raise ValueError("This setup supports a single board only")
            if not 0 <= config.core_ind < qubic_core_number:
                raise ValueError(f"Channel {channel} exceeds the configured core count")
        readout_cores = [config.core_ind for channel, config in self._channel_configs.items()
                         if channel.endswith(".rdlo")]
        if len(readout_cores) != len(set(readout_cores)):
            raise ValueError("Each core must have a unique readout channel")
        self.rpc_uri = rpc_uri

    def compile_circuits(self, circuits, batch_size=1):
        """Compile locally and return an Executable with readout metadata.

        As in the LeeQ sweep engine, each measured channel must appear once
        per sweep point. Mid-circuit measurements are outside this adapter.
        """
        from distproc.executable import Executable

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        read_counts = {}
        for instruction in circuits:
            dest = instruction.get("dest", "")
            if instruction.get("name") == "pulse" and dest.endswith(".rdlo"):
                read_counts[dest] = read_counts.get(dest, 0) + 1
        if not read_counts or any(count != batch_size for count in read_counts.values()):
            raise ValueError("Each measured channel must have one readout per sweep point")
        for channel in read_counts:
            if channel not in self._channel_configs:
                raise ValueError(f"Missing channel metadata for {channel}")

        tc, *_ = self._load_qubic_package()
        register_leeq_pulse_shapes_to_qubic_pulse_shape_factory()
        compiled = tc.run_compile_stage(circuits, fpga_config=self._fpga_config,
                                        qchip=None, compiler_flags={"resolve_gates": False})
        executable = tc.run_assemble_stage(compiled, self._channel_configs)
        if not isinstance(executable, Executable):
            raise TypeError("This setup requires the QubiC Executable toolchain")
        readouts = {}
        for channel in read_counts:
            if channel not in executable.result_channels:
                raise ValueError(f"Missing executable result channel {channel}")
            result = executable.result_channels[channel]
            if result.board:
                raise ValueError("This setup supports a single board only")
            if result.dtype != "s11":
                raise ValueError(f"Readout {channel} must use the s11 result type")
            result.reads_per_shot = batch_size
            readouts[channel] = result
        executable.result_channels = readouts
        return executable

    def _run_qubic_circuits(self, circuits, batch_size, zero,
                           load_commands, load_freqs, load_envs):
        self._result = {}
        acquisition_type = self._status.get_parameters("Acquisition_Type")
        if acquisition_type not in ("IQ", "IQ_average"):
            raise NotImplementedError("The Executable RPC setup currently supports IQ and IQ_average")
        executable = self.compile_circuits(circuits, batch_size=batch_size)
        shots = self._status.get_parameters("Shot_Number")
        results = self._runner.run_circuit_batch(
            executables=[executable], n_total_shots=shots,
            reads_per_shot=batch_size, reload_cmd=load_commands,
            reload_freq=load_freqs, reload_env=load_envs,
            zero_between_reload=zero)
        if len(results) != 1 or set(results[0]) != set(executable.result_channels):
            raise ValueError("RPC result channels do not match the submitted executable")
        normalized = {}
        for channel, value in results[0].items():
            core = str(self._channel_configs[channel].core_ind)
            if core in normalized:
                raise ValueError(f"Multiple readout channels returned for core {core}")
            data = np.asarray(value)
            if data.shape != (shots, batch_size):
                raise ValueError(f"Unexpected readout shape for {channel}: {data.shape}")
            normalized[core] = data
        self._result = normalized
