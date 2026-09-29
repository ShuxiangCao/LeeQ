#!/usr/bin/env python3
"""Run one reviewed X6Y3 Ramsey/ping-pong scan; compile only unless --execute."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--channel-config', type=Path, required=True)
    parser.add_argument('--client-path', type=Path)
    parser.add_argument('--readout-reference', type=Path, required=True)
    parser.add_argument('--qubit', type=int, choices=[0, 1], required=True)
    parser.add_argument('--mode', choices=['ramsey', 'pingpong'], required=True)
    parser.add_argument('--stage', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--center-offset', type=float, default=0)
    parser.add_argument('--x90-scale', type=float, default=1)
    parser.add_argument('--gate', choices=['X', 'X90'], default='X90')
    parser.add_argument('--scales', default='0.98,1,1.02')
    parser.add_argument('--counts', default='0,4,8,12')
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.client_path:
        sys.path.append(str(args.client_path.resolve()))
    if args.execute:
        socket.setdefaulttimeout(60)
    else:
        def blocked(*a, **k):
            raise RuntimeError('Offline tune-up check: network disabled')
        socket.socket.connect = blocked
        socket.socket.connect_ex = blocked
        socket.create_connection = blocked
    from leeq.setups.x6y3 import X6Y3Calibration
    from leeq.setups.huracan import create_huracan_setup
    from leeq.experiments.experiments import ExperimentManager
    from leeq.experiments.x6y3 import acquire_plan
    from leeq.experiments.x6y3_tuneup import leeq_ramsey_plan, pingpong_plan, fit_ramsey, fit_pingpong
    from leeq.experiments.x6y3_validation import verify_plan_parity

    calibration = X6Y3Calibration(args.calibration)
    setup = create_huracan_setup(args.channel_config)
    reference_manifest = json.loads((args.readout_reference / 'manifest.json').read_text())
    if (reference_manifest['qubit'] != args.qubit or reference_manifest['status'] != 'complete'
            or reference_manifest['calibration_sha256'] != calibration.sha256):
        raise ValueError('Readout reference must match selected qubit and baseline calibration')
    reference = json.loads((args.readout_reference / 'analysis.json').read_text())
    if args.mode == 'ramsey':
        plan = leeq_ramsey_plan(args.qubit, args.stage, center_offset_mhz=args.center_offset,
                                x90_scale=args.x90_scale, shots=args.shots)
    else:
        plan = pingpong_plan(args.qubit, args.gate, [float(x) for x in args.scales.split(',')],
                             [int(x) for x in args.counts.split(',')], center_offset_mhz=args.center_offset,
                             x90_scale=args.x90_scale, shots=args.shots, blocks=args.blocks, seed=args.seed)
    if not args.execute:
        report = verify_plan_parity(calibration, setup, plan)
        print(f'PASS offline parity: {report["points_verified"]} points')
        return
    ExperimentManager().register_setup(setup)
    result = acquire_plan(calibration, setup, plan, args.output)
    report = (fit_ramsey if args.mode == 'ramsey' else fit_pingpong)(result, reference)
    (args.output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('mean', 'standard_error')}, indent=2))


if __name__ == '__main__':
    main()
