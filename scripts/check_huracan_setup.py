#!/usr/bin/env python3
"""Compile a synthetic LeeQ readout locally; network and RPC calls are blocked."""

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import xmlrpc.client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-config", type=Path, required=True,
                        help="Local channel_config.json from the aa01c78f package")
    parser.add_argument("--client-path", type=Path,
                        help="Optional fallback directory containing installed modern QubiC/distproc packages")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.client_path:
        sys.path.append(str(args.client_path.resolve()))

    rpc_attempts = []

    def block_socket(*args, **kwargs):
        raise RuntimeError("Network access is disabled by the offline Huracan check")

    def block_rpc(*args, **kwargs):
        rpc_attempts.append(True)
        raise RuntimeError("RPC calls are disabled by the offline Huracan check")

    socket.socket.connect = block_socket
    socket.socket.connect_ex = block_socket
    socket.create_connection = block_socket
    xmlrpc.client.Transport.request = block_rpc
    xmlrpc.client.SafeTransport.request = block_rpc

    from leeq.core.context import ExperimentContext
    from leeq.core.primitives.built_in.simple_drive import SimpleDispersiveMeasurement
    from leeq.setups.huracan import create_huracan_setup
    import qubic.rpc_client
    import distproc.executable

    setup = create_huracan_setup(args.channel_config)
    # Synthetic values exercise the compiler; they are not Huracan calibration.
    measurements = [SimpleDispersiveMeasurement(
        name=f"offline-Q{core}", parameters=dict(channel=2 * core + 1,
            freq=100.0, width=0.032, amp=0.0, phase=0.0, shape="square",
            distinguishable_states=[0, 1])) for core in (0, 1)]
    context = ExperimentContext("offline-Huracan")
    context.set_step_no((0,))
    setup._compiler.compile_lpb(context, measurements[0] * measurements[1])
    executable = setup.compile_circuits(context.instructions["circuits"])
    if set(executable.result_channels) != {"Q0.rdlo", "Q1.rdlo"}:
        raise RuntimeError("Unexpected compiled readout channels")
    if rpc_attempts:
        raise RuntimeError("The offline check attempted RPC access")
    print(json.dumps({
        "mode": "offline_compile_only",
        "expected_image": "aa01c78f",
        "rpc_uri": setup.rpc_uri,
        "rpc_attempts": len(rpc_attempts),
        "channel_config": str(args.channel_config.resolve()),
        "channel_config_sha256": hashlib.sha256(args.channel_config.read_bytes()).hexdigest(),
        "qubic_client_source": qubic.rpc_client.__file__,
        "distproc_source": distproc.executable.__file__,
        "program_buffers": sorted(executable.program_binaries),
        "readouts": {name: {"memory": result.mem_name, "reads_per_shot": result.reads_per_shot}
                     for name, result in executable.result_channels.items()},
        "pass": True,
    }, indent=2))


if __name__ == "__main__":
    main()
