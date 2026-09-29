#!/usr/bin/env python3
"""Validate archived X6Y3 calibration through LeeQ and qcal, with networking blocked."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import xmlrpc.client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--channel-config', type=Path, required=True)
    parser.add_argument('--client-path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full', action='store_true', help='Check complete default experiment grids')
    args = parser.parse_args()
    # LeeQ's optional LLM dependency otherwise fetches a price map at import.
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    attempts = []

    def blocked(*a, **kw):
        attempts.append(True)
        raise RuntimeError('Network and RPC disabled for X6Y3 offline validation')

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    xmlrpc.client.Transport.request = blocked
    xmlrpc.client.SafeTransport.request = blocked
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.client_path:
        sys.path.append(str(args.client_path.resolve()))

    from leeq.setups.x6y3 import X6Y3Calibration
    from leeq.setups.huracan import create_huracan_setup
    from leeq.experiments.x6y3 import readout_plan, amplitude_plan, ramsey_plan, t1_plan, provenance
    from leeq.experiments.x6y3_validation import verify_plan_parity

    calibration = X6Y3Calibration(args.calibration)
    setup = create_huracan_setup(args.channel_config)
    report = {**provenance(calibration, setup), 'mode': 'offline', 'experiments': []}
    for q in (0, 1):
        plans = [readout_plan(q)]
        if args.full:
            plans += [amplitude_plan(q, n_gates=n) for n in (1, 4)]
            plans += [ramsey_plan(q), t1_plan(q)]
        else:
            plans += [amplitude_plan(q, scales=[0.7, 1.0, 1.3], n_gates=n) for n in (1, 4)]
            plans += [ramsey_plan(q, delays_us=[0, 0.123, 1], detunings_mhz=[-2.5, 2.5]),
                      t1_plan(q, delays_us=[0, 17.25, 350])]
        for plan in plans:
            result = verify_plan_parity(calibration, setup, plan)
            report['experiments'].append(result)
            print(f'Q{q} {plan.name}: {result["points_verified"]} points matched', flush=True)
    report.update(network_attempts=len(attempts), passed=not attempts)
    if attempts:
        raise RuntimeError('Offline validation attempted networking')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: report written to {args.output}')


if __name__ == '__main__':
    main()
