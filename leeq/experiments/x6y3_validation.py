"""Independent qcal transpilation reference for offline X6Y3 parity checks."""

import copy
import hashlib

import numpy as np

from leeq.experiments.x6y3 import compile_plan


def qcal_reference(calibration, plan, point):
    from qcal.config import Config
    from qcal.circuit import Cycle
    from qcal.gates.single_qubit import X90, X, Idle, Rz, Meas
    from qcal.backend.qubic.transpiler import cycle_pulse

    config = Config(str(calibration.path))
    config._parameters = calibration.snapshot
    q = plan.qubit
    scope = [f'Q{q}']
    program = [{'name': 'delay', 't': calibration.passive_delay_us / 1e6, 'scope': scope},
               {'name': 'barrier', 'scope': scope}]
    for op in point['operations']:
        name = op['gate']
        if name in ('X90', 'X'):
            # Restore the source for each operation, so scale does not accumulate.
            config._parameters = calibration.snapshot
            for pulse in config[f'single_qubit/{q}/GE/{name}/pulse']:
                if pulse['env'] != 'virtualz':
                    pulse['kwargs']['amp'] *= op.get('amplitude_scale', 1.0)
            gate = (X90 if name == 'X90' else X)(q)
        elif name == 'Idle':
            gate = Idle(q, duration=op['time_us'] / 1e6)
        elif name == 'Rz':
            gate = Rz(q, op['phase_rad'])
        else:
            raise ValueError(name)
        # Calling cycle_pulse directly avoids qcal's duration-insensitive Idle cache.
        program.extend(cycle_pulse(config, Cycle({gate})))
        program.append({'name': 'barrier', 'scope': scope})
    program.extend(cycle_pulse(config, Cycle({Meas(q)})))
    return program


def verify_plan_parity(calibration, setup, plan):
    """Require matching scheduled pulses and hardware waveform/frequency buffers."""
    import qubic.toolchain as tc
    from qubic.pulse_factory import PulseShapeFactory

    results = []
    for record in compile_plan(calibration, setup, plan):
        programs = [record['instructions'], qcal_reference(calibration, plan, record['point'])]
        scheduled = [tc.run_compile_stage(copy.deepcopy(program), setup._fpga_config, None,
                                          compiler_flags={'resolve_gates': False}) for program in programs]
        pulses = [[pulse for group in compiled.program.values() for pulse in group
                   if pulse['op'] == 'pulse'] for compiled in scheduled]
        if len(pulses[0]) != len(pulses[1]):
            raise AssertionError('LeeQ/qcal pulse counts differ')
        max_error = 0.0
        for actual, expected in zip(*pulses):
            if actual['dest'] != expected['dest'] or actual['start_time'] != expected['start_time']:
                raise AssertionError(f'Schedule mismatch: {actual} versus {expected}')
            for key in ('freq', 'phase', 'amp'):
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-13, atol=1e-10,
                                           err_msg=f'{actual["dest"]} {key}')
            element = setup.channel_metadata[actual['dest']]['elem_params']
            rate = setup.channel_metadata['fpga_clk_freq'] * element['samples_per_clk'] / element['interp_ratio']
            env = actual['env']
            _, samples = PulseShapeFactory().get_pulse_shape_function(env['env_func'])(
                dt=1 / rate, **env['paradict'])
            np.testing.assert_allclose(samples, expected['env'], rtol=0, atol=1e-7)
            max_error = max(max_error, float(np.max(np.abs(samples - expected['env']))))
        reference = tc.run_assemble_stage(scheduled[1], setup._channel_configs)
        actual = record['executable']
        # Instruction encodings can differ while schedules match. Waveform and
        # frequency memories must be byte-identical for faithful calibration.
        memories = [name for name in reference.program_binaries if 'env' in name or 'freq' in name]
        for name in memories:
            if actual.program_binaries[name] != reference.program_binaries[name]:
                raise AssertionError(f'Hardware memory differs: {name}')
        results.append({'point': record['point'], 'pulse_count': len(pulses[0]),
                        'max_envelope_error': max_error,
                        'matching_memories': {name: hashlib.sha256(actual.program_binaries[name]).hexdigest()
                                              for name in memories}})
    return {'experiment': plan.name, 'qubit': plan.qubit, 'points_verified': len(results),
            'pass': True, 'points': results}
