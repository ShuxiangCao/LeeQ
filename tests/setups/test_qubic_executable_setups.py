"""Modern QubiC compilation and RPC decoding, with all networking blocked."""

import copy
import json
import socket
from types import SimpleNamespace
from unittest.mock import Mock
from xmlrpc.client import Binary

import numpy as np
import pytest

pytest.importorskip("distproc.executable")
pytest.importorskip("qubic.rpc_client")

from leeq.setups.huracan import create_huracan_setup
from leeq.setups.qubic_executable_setups import QubiCSingleBoardExecutableRPCSetup


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Offline QubiC tests attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def metadata():
    config = {"fpga_clk_freq": 500e6}
    for core in (0, 1):
        for element, index, interpolation, samples in (("qdrv", 0, 4, 16), ("rdrv", 1, 16, 16), ("rdlo", 2, 4, 4)):
            channel = dict(core_ind=core, elem_ind=index, core_name="qubit",
                           env_mem_name=f"qubit_{element}_env{core}",
                           freq_mem_name=f"qubit_{element}_freq{core}",
                           elem_params=dict(interp_ratio=interpolation, samples_per_clk=samples),
                           elem_type="rf_mix" if element == "rdlo" else "rf")
            if element == "rdlo":
                channel["acc_mem_name"] = f"qubit_accbuf{core}"
            config[f"Q{core}.{element}"] = channel
    return config


@pytest.fixture
def setup(tmp_path, metadata):
    path = tmp_path / "channel_config.json"
    path.write_text(json.dumps(metadata))
    instance = create_huracan_setup(path)
    instance.status.set_parameters(Acquisition_Type="IQ", Shot_Number=3)
    return instance


def circuits(cores=(0, 1), points=1):
    program = []
    for _ in range(points):
        program.append({"name": "delay", "t": 1e-6})
        for core in cores:
            for element in ("rdrv", "rdlo"):
                program.append(dict(name="pulse", dest=f"Q{core}.{element}", twidth=32e-9,
                                    freq=100e6, amp=0.0 if element == "rdrv" else 1.0, phase=0.0,
                                    env={"env_func": "leeq_square", "paradict": {"width": 0.032}}))
        program.append({"name": "barrier"})
    return program


def run(setup, points=2):
    setup._run_qubic_circuits(circuits(points=points), batch_size=points,
                             zero=True, load_commands=True, load_freqs=True, load_envs=True)


def test_setup_and_real_compile_are_offline(setup):
    executable = setup.compile_circuits(circuits(cores=(1,), points=2), batch_size=2)
    assert setup.rpc_uri == "http://127.0.0.1:9095"
    assert set(executable.result_channels) == {"Q1.rdlo"}
    assert executable.result_channels["Q1.rdlo"].mem_name == "qubit_accbuf1"
    assert executable.result_channels["Q1.rdlo"].reads_per_shot == 2
    assert executable.program_binaries


def test_real_client_serialization_and_result_mapping(setup):
    arrays = {"Q0.rdlo": np.arange(6).reshape(3, 2) + 2j,
              "Q1.rdlo": -np.arange(6).reshape(3, 2) - 4j}
    packed = {channel: Binary(np.column_stack((value.imag.ravel(), value.real.ravel())).astype("<i4").tobytes())
              for channel, value in arrays.items()}
    # Exercise the real CircuitRunnerClient; replace only its remote proxy.
    proxy = Mock()
    proxy.run_circuit_batch.return_value = [packed]
    setup._runner.proxy = proxy
    run(setup)
    args = proxy.run_circuit_batch.call_args.args
    assert args[1:3] == (3, 2)
    assert set(args[0][0]["result_channels"]) == set(arrays)
    np.testing.assert_array_equal(setup._result["0"], arrays["Q0.rdlo"])
    np.testing.assert_array_equal(setup._result["1"], arrays["Q1.rdlo"])


@pytest.mark.parametrize("acquisition", ["IQ", "IQ_average"])
def test_sweep_result_order_and_averaging(setup, acquisition):
    setup.status.set_parameter("Acquisition_Type", acquisition)
    values = np.arange(6).reshape(3, 2).astype(complex)
    setup._runner = SimpleNamespace(run_circuit_batch=Mock(return_value=[{"Q0.rdlo": values, "Q1.rdlo": values + 10}]))
    run(setup)
    contexts = [SimpleNamespace(instructions={"qubic_channel_to_lpb_uuid": {"Q0": "m0", "Q1": "m1"}},
                                step_no=(i,), results=[]) for i in range(2)]
    setup.collect_data_batch(contexts)
    for i, context in enumerate(contexts):
        expected = values[:, i:i + 1]
        if acquisition == "IQ_average":
            expected = expected.mean(axis=0, keepdims=True)
        np.testing.assert_array_equal(context.results[0].data[0], expected)


@pytest.mark.parametrize("result", [[], [{"Q0.rdlo": np.zeros((3, 2))}],
                                    [{"Q0.rdlo": np.zeros((3, 1)), "Q1.rdlo": np.zeros((3, 2))}]])
def test_bad_results_do_not_leave_stale_data(setup, result):
    setup._result = {"0": np.ones((3, 2))}
    setup._runner = SimpleNamespace(run_circuit_batch=Mock(return_value=result))
    with pytest.raises(ValueError):
        run(setup)
    assert setup._result == {}


def test_no_retry_on_rpc_failure(setup):
    call = Mock(side_effect=ConnectionError("transport lost"))
    setup._runner = SimpleNamespace(run_circuit_batch=call)
    with pytest.raises(ConnectionError):
        run(setup)
    call.assert_called_once()


def test_trace_mode_fails_before_rpc(setup):
    setup.status.set_parameter("Acquisition_Type", "traces")
    with pytest.raises(NotImplementedError, match="IQ"):
        run(setup)


def test_unbalanced_readout_counts_fail_before_rpc(setup):
    with pytest.raises(ValueError, match="one readout per sweep point"):
        setup.compile_circuits(circuits(points=1), batch_size=2)


def test_huracan_rejects_eight_core_metadata(tmp_path, metadata):
    metadata["Q2.rdlo"] = copy.deepcopy(metadata["Q1.rdlo"])
    path = tmp_path / "channel_config.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="exactly Q0/Q1"):
        create_huracan_setup(path)


def test_legacy_batch_client_is_rejected(monkeypatch, metadata):
    class LegacyClient:
        def __init__(self, ip, port):
            pass

        def run_circuit_batch(self, raw_asm_list, n_total_shots):
            pytest.fail("Legacy RPC must never be called")

    monkeypatch.setattr("qubic.rpc_client.CircuitRunnerClient", LegacyClient)
    with pytest.raises(TypeError, match="Executable-based"):
        QubiCSingleBoardExecutableRPCSetup("test", "http://127.0.0.1:9095", metadata,
                                          {0: "Q0", 1: "Q0", 2: "Q1", 3: "Q1"}, 2)


def test_duplicate_readout_core_is_rejected(metadata):
    metadata["Q1.rdlo"]["core_ind"] = 0
    with pytest.raises(ValueError, match="unique readout"):
        QubiCSingleBoardExecutableRPCSetup("test", "http://127.0.0.1:9095", metadata,
                                          {0: "Q0", 1: "Q0", 2: "Q1", 3: "Q1"}, 2)
