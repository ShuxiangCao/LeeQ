#!/usr/bin/env python3
"""Compile X6Y3 experiments locally; --execute explicitly submits shots to Huracan."""

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
    parser.add_argument('--experiment', choices=['readout', 'amplitude', 'ramsey', 't1'], required=True)
    parser.add_argument('--qubit', type=int, choices=[0, 1], default=0)
    parser.add_argument('--shots', type=int, default=512)
    parser.add_argument('--blocks', type=int, default=2, help='Readout preparation blocks')
    parser.add_argument('--n-gates', type=int, default=4, help='X90 repetitions for amplitude sweep')
    parser.add_argument('--output', type=Path, required=True, help='New local result directory')
    parser.add_argument('--readout-reference', type=Path, help='Readout result directory for IQ projection fits')
    parser.add_argument('--execute', action='store_true', help='Perform physical measurements through existing RPC')
    args = parser.parse_args()
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.client_path:
        sys.path.append(str(args.client_path.resolve()))
    if not args.execute:
        def blocked(*a, **kw):
            raise RuntimeError('Networking disabled: --execute was not specified')
        socket.socket.connect = blocked
        socket.socket.connect_ex = blocked
        socket.create_connection = blocked
    else:
        socket.setdefaulttimeout(60)

    from leeq.experiments.experiments import ExperimentManager
    from leeq.setups.huracan import create_huracan_setup
    from leeq.setups.x6y3 import X6Y3Calibration
    from leeq.experiments.x6y3 import (
        readout_plan, amplitude_plan, ramsey_plan, t1_plan,
        acquire_plan, compile_plan, analyze_readout, analyze_sweep,
    )

    calibration = X6Y3Calibration(args.calibration)
    setup = create_huracan_setup(args.channel_config)
    if args.experiment == 'readout':
        plan = readout_plan(args.qubit, args.shots, args.blocks)
    elif args.experiment == 'amplitude':
        plan = amplitude_plan(args.qubit, n_gates=args.n_gates, shots=args.shots)
    else:
        plan = {'ramsey': ramsey_plan, 't1': t1_plan}[args.experiment](args.qubit, shots=args.shots)
    reference = None
    if args.readout_reference:
        manifest = json.loads((args.readout_reference / 'manifest.json').read_text())
        if (manifest['qubit'] != plan.qubit or manifest['calibration_sha256'] != calibration.sha256
                or manifest['status'] != 'complete' or manifest['experiment'] != 'readout'):
            raise ValueError('Readout reference must match qubit and calibration and be complete')
        reference = json.loads((args.readout_reference / 'analysis.json').read_text())
    if not args.execute:
        records = compile_plan(calibration, setup, plan)
        print(f'Offline: compiled {len(records)} {plan.name} points on Q{plan.qubit}; no shots submitted')
        return
    ExperimentManager().register_setup(setup)
    result = acquire_plan(calibration, setup, plan, args.output)
    if plan.name == 'readout' and args.blocks >= 2:
        analysis = analyze_readout(result)
    elif reference:
        analysis = analyze_sweep(result, reference)
    else:
        analysis = {'note': 'Raw IQ saved; supply matching readout reference for analysis'}
    (args.output / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'output': str(args.output), 'points': len(plan.points),
                      'shots': len(plan.points) * plan.shots, 'analysis': analysis}, indent=2))


if __name__ == '__main__':
    main()
