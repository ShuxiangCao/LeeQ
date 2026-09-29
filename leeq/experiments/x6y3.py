"""Baseline-preserving X6Y3 protocols, offline compilation and opt-in acquisition."""

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from leeq.core.context import ExperimentContext
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSweep
from leeq.experiments.experiments import ExperimentManager
from leeq.experiments.sweeper import Sweeper


@dataclass
class X6Y3Plan:
    name: str
    qubit: int
    points: list
    shots: int

    def __post_init__(self):
        if type(self.qubit) is not int or self.qubit not in (0, 1):
            raise ValueError('Select Q0 or Q1')
        if type(self.shots) is not int or not 1 <= self.shots <= 5000:
            raise ValueError('shots must be an integer between 1 and 5000')
        if not self.points:
            raise ValueError('A plan must contain at least one point')


def _axis(values, name, *, nonnegative=False):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be a nonempty finite one-dimensional axis')
    if nonnegative and np.any(result < 0):
        raise ValueError(f'{name} cannot be negative')
    return result.tolist()


def readout_plan(qubit=0, shots=512, blocks=2):
    """Alternating nominal ground and X90+X90 excited preparations."""
    if type(blocks) is not int or blocks < 1:
        raise ValueError('blocks must be a positive integer')
    return X6Y3Plan('readout', qubit, [
        {'block': block, 'prepared_state': state,
         'operations': [{'gate': 'X90'}] * (2 * state)}
        for block in range(blocks) for state in (0, 1)], shots)


def amplitude_plan(qubit=0, scales=None, n_gates=4, shots=512):
    """qcal GE X90 repetition protocol; never changes calibration parameters."""
    if type(n_gates) is not int or (n_gates != 1 and (n_gates <= 0 or n_gates % 4)):
        raise ValueError('X90 n_gates must be one or a positive multiple of four')
    values = _axis(np.linspace(0.7, 1.3, 31) if scales is None else scales,
                   'amplitude scales', nonnegative=True)
    return X6Y3Plan('amplitude', qubit, [
        {'amplitude_scale': scale, 'n_gates': n_gates,
         'operations': [{'gate': 'X90', 'amplitude_scale': scale}] * n_gates}
        for scale in values], shots)


def ramsey_plan(qubit=0, delays_us=None, detunings_mhz=(-5, -2.5, 2.5, 5), shots=512):
    """qcal Frequency: X90, idle, Rz(2*pi*detuning*time), X90.

    Detuning is a virtual-Z ramp, not a physical drive-frequency change.
    """
    delays = _axis(np.linspace(0, 1, 30) if delays_us is None else delays_us,
                   'Ramsey delays', nonnegative=True)
    detunings = _axis(detunings_mhz, 'Ramsey detunings')
    return X6Y3Plan('ramsey', qubit, [
        {'delay_us': delay, 'detuning_mhz': detuning,
         'operations': [{'gate': 'X90'}, {'gate': 'Idle', 'time_us': delay},
                        {'gate': 'Rz', 'phase_rad': 2 * np.pi * detuning * delay}, {'gate': 'X90'}]}
        for detuning in detunings for delay in delays], shots)


def t1_plan(qubit=0, delays_us=None, shots=2000):
    """qcal's default GE T1 preparation uses two calibrated X90 gates."""
    delays = _axis(np.linspace(0, 350, 50) if delays_us is None else delays_us,
                   'T1 delays', nonnegative=True)
    return X6Y3Plan('t1', qubit, [
        {'delay_us': delay, 'operations': [{'gate': 'X90'}, {'gate': 'X90'},
                                         {'gate': 'Idle', 'time_us': delay}]}
        for delay in delays], shots)


def point_lpb(calibration, plan, point):
    measurement = calibration.measurement(plan.qubit)
    return calibration.sequence(plan.qubit, point['operations']) + measurement, measurement


def compile_plan(calibration, setup, plan):
    """Compile every point locally; never contact RPC or register a setup."""
    calibration.verify_channel_metadata(setup.channel_metadata)
    current = ExperimentManager().get_default_setup()
    if current is not None and current is not setup:
        raise ValueError('Another default setup could apply unrelated channel callbacks')
    compiled = []
    for index, point in enumerate(plan.points):
        lpb, _ = point_lpb(calibration, plan, point)
        context = ExperimentContext(f'{plan.name}.{index}')
        context.set_step_no((index,))
        setup._compiler.compile_lpb(context, lpb)
        instructions = context.instructions['circuits']
        executable = setup.compile_circuits(instructions)
        compiled.append({'point': copy.deepcopy(point), 'instructions': instructions,
                         'executable': executable})
    return compiled


def provenance(calibration, setup):
    import qubic.rpc_client
    import distproc.executable

    versions = {}
    for package in ('LeeQ', 'quantum-calibration', 'lbl-qubic', 'distproc', 'numpy', 'scipy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'uninstalled checkout'
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=False).stdout.strip()
    source_files = ('leeq/setups/x6y3.py', 'leeq/setups/huracan.py',
                    'leeq/setups/qubic_executable_setups.py', 'leeq/experiments/x6y3.py',
                    'leeq/experiments/x6y3_tuneup.py',
                    'leeq/compiler/lbnl_qubic/circuit_list_compiler.py',
                    'leeq/compiler/lbnl_qubic/utils.py')
    source_hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in source_files}
    return {'calibration_path': str(calibration.path), 'calibration_sha256': calibration.sha256,
            'leeq_revision': revision, 'source_sha256': source_hashes, 'versions': versions,
            'qubic_client_source': qubic.rpc_client.__file__,
            'distproc_source': distproc.executable.__file__, 'rpc_uri': setup.rpc_uri,
            'channel_config_path': setup.channel_config_path,
            'channel_config_sha256': setup.channel_config_sha256,
            'channel_metadata': setup.channel_metadata,
            'expected_image': 'aa01c78f', 'source_herald': calibration.source_herald,
            'herald': False, 'active_reset': False, 'calibration_updated': False,
            'passive_delay_us': calibration.passive_delay_us}


def acquire_plan(calibration, setup, plan, output_dir):
    """Run native LeeQ LPBs through its sweep engine; explicitly performs shots.

    One point per submission limits waveform memory and saves each completed
    point before continuing. Errors propagate without retries; completed data
    and an error manifest remain on disk. Call only when hardware use is intended.
    """
    if ExperimentManager().get_default_setup() is not setup:
        raise ValueError('Register the intended Huracan setup as the LeeQ default first')
    # Validate every point before the first hardware call.
    compile_plan(calibration, setup, plan)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'calibration.yaml').write_bytes(calibration._source)
    manifest = {**provenance(calibration, setup), 'experiment': plan.name,
                'qubit': plan.qubit, 'shots_per_point': plan.shots, 'points': plan.points,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'running', 'completed_points': 0}
    manifest_path = output / 'manifest.json'

    def save_manifest():
        temporary = output / 'manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        temporary.replace(manifest_path)

    save_manifest()
    previous = setup.status.get_parameters()
    # Passive wait is already in the native sequence, once per shot.
    setup.status.set_parameters(Acquisition_Type='IQ', Shot_Number=plan.shots,
                                Shot_Period=0.0, Engine_Batch_Size=1, Measurement_Basis=None)
    values = []
    try:
        for index, point in enumerate(plan.points):
            lpb, measurement = point_lpb(calibration, plan, point)
            sweep_lpb = LogicalPrimitiveBlockSweep([lpb])
            setup.run(sweep_lpb, Sweeper.from_sweep_lpb(sweep_lpb))
            iq = np.asarray(measurement.result(raw_data=True)).reshape(-1)
            if iq.shape != (plan.shots,) or not np.all(np.isfinite(iq)):
                raise ValueError('Unexpected IQ shot count or nonfinite data')
            np.savez_compressed(output / f'point-{index:04d}.npz', iq=iq)
            values.append(iq)
            manifest['completed_points'] = index + 1
            save_manifest()
        iq = np.stack(values)
        np.savez_compressed(output / 'iq.npz', iq=iq)
        manifest['status'] = 'complete'
        return {'iq': iq, 'manifest': manifest, 'output_dir': output}
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        setup.status.set_parameters(**previous)
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save_manifest()


def analyze_readout(result):
    """Held-out assignment to nominal preparation labels, not latent fidelity."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import confusion_matrix

    iq = result['iq']
    points = result['manifest']['points']
    labels = np.array([point['prepared_state'] for point in points])
    # With repeated blocks, hold out the later blocks entirely (drift-sensitive).
    blocks = np.array([point['block'] for point in points])
    unique = np.unique(blocks)
    if len(unique) < 2:
        raise ValueError('Use at least two readout blocks for held-out analysis')
    training = blocks < unique[len(unique) // 2]
    train_iq, test_iq = iq[training].ravel(), iq[~training].ravel()
    train_labels = np.repeat(labels[training], iq.shape[1])
    test_labels = np.repeat(labels[~training], iq.shape[1])
    classifier = LinearDiscriminantAnalysis().fit(
        np.column_stack((train_iq.real, train_iq.imag)), train_labels)
    prediction = classifier.predict(np.column_stack((test_iq.real, test_iq.imag)))
    centers = [complex(iq[training & (labels == state)].mean()) for state in (0, 1)]
    report = {'metric': 'held-out nominal preparation assignment',
              'accuracy': float(np.mean(prediction == test_labels)),
              'confusion_counts': confusion_matrix(test_labels, prediction, labels=[0, 1]).tolist(),
              'training_blocks': unique[:len(unique) // 2].tolist(),
              'held_out_blocks': unique[len(unique) // 2:].tolist(),
              'iq_centers': [[z.real, z.imag] for z in centers],
              'lda_coefficients': classifier.coef_.tolist(),
              'lda_intercept': classifier.intercept_.tolist(), 'calibration_updated': False}
    return report


def analyze_sweep(result, readout_report):
    """Fit a baseline-referenced IQ projection; reports never update calibration."""
    from scipy.optimize import curve_fit

    centers = [complex(*pair) for pair in readout_report['iq_centers']]
    axis = centers[1] - centers[0]
    if abs(axis) == 0:
        raise ValueError('Readout centers coincide')
    projection = np.real((result['iq'] - centers[0]) / axis)
    y = projection.mean(axis=1)
    points = result['manifest']['points']
    name = result['manifest']['experiment']
    report = {'observable': 'baseline-referenced IQ projection (not corrected population)',
              'mean': y.tolist(), 'standard_error': (projection.std(axis=1, ddof=1)
                                                    / np.sqrt(projection.shape[1])).tolist(),
              'calibration_updated': False, 'fits': []}
    try:
        if name == 't1':
            x = np.array([p['delay_us'] for p in points])
            fit, cov = curve_fit(lambda t, a, tau, b: a * np.exp(-t / tau) + b, x, y,
                                 p0=[y[0] - y[-1], max(np.ptp(x) / 5, 1e-3), y[-1]],
                                 bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
                                 maxfev=10000)
            report['fits'].append({'T1_us': float(fit[1]), 'T1_std_us': float(np.sqrt(cov[1, 1])),
                                   'parameters': fit.tolist()})
        elif name == 'ramsey':
            for detuning in sorted({p['detuning_mhz'] for p in points}):
                indices = [i for i, p in enumerate(points) if p['detuning_mhz'] == detuning]
                x = np.array([points[i]['delay_us'] for i in indices])
                yy = y[indices]
                fit, cov = curve_fit(
                    lambda t, a, f, phase, b: a * np.cos(2 * np.pi * f * t + phase) + b,
                    x, yy, p0=[np.ptp(yy) / 2, abs(detuning), 0, yy.mean()], maxfev=10000)
                report['fits'].append({'detuning_mhz': detuning, 'oscillation_mhz': abs(float(fit[1])),
                                       'frequency_std_mhz': float(np.sqrt(cov[1, 1])),
                                       'parameters': fit.tolist()})
        elif name == 'amplitude' and all(p['n_gates'] == 4 for p in points):
            x = np.array([p['amplitude_scale'] for p in points])
            near = np.abs(x - 1) <= 0.15
            if near.sum() >= 4:
                fit, cov = np.polyfit(x[near] - 1, y[near], 2, cov=True)
                candidate = 1 - fit[1] / (2 * fit[0])
                if fit[0] > 0 and min(x[near]) <= candidate <= max(x[near]):
                    report['fits'].append({'candidate_amplitude_scale': float(candidate),
                                           'quadratic_coefficients': fit.tolist(), 'covariance': cov.tolist()})
                else:
                    report['fit_note'] = 'No resolved minimum near the existing calibration; no correction proposed'
        else:
            report['fit_note'] = 'Diagnostic curve only'
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        report['fit_error'] = str(error)
    return report
