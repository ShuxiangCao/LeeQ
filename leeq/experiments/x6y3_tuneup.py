"""Three-stage LeeQ Ramsey and independent X/X90 ping-pong refinement."""

from pathlib import Path

import numpy as np

from leeq.experiments.x6y3 import X6Y3Plan


# TuneUpExample.ipynb, cells 14/20: physical drive-frequency offsets, us/MHz.
RAMSEY_STAGES = ((10.0, 0.3, 0.005), (1.0, 3.0, 0.05), (0.1, 30.0, 0.5))


def leeq_ramsey_plan(qubit, stage, *, center_offset_mhz=0.0, x90_scale=1.0, shots=1024):
    """Xp–idle–Xm at center + offset, exactly the example's three scan grids."""
    offset, stop, step = RAMSEY_STAGES[stage]
    points = []
    for delay in np.arange(0, stop, step):
        points.append({'stage': stage, 'delay_us': float(delay), 'set_offset_mhz': offset,
                       'center_offset_mhz': center_offset_mhz,
                       'operations': [
                           {'gate': 'X90', 'amplitude_scale': x90_scale,
                            'frequency_offset_mhz': center_offset_mhz + offset},
                           {'gate': 'Idle', 'time_us': float(delay)},
                           {'gate': 'X90', 'amplitude_scale': x90_scale, 'phase_offset_rad': np.pi,
                            'frequency_offset_mhz': center_offset_mhz + offset}]})
    return X6Y3Plan('leeq_ramsey', qubit, points, shots)


def pingpong_plan(qubit, gate, scales, counts, *, center_offset_mhz=0.0,
                  x90_scale=1.0, shots=1024, blocks=2, seed=20260929):
    """LeeQ ping-pong repetitions followed by Xp, plus an Xm control.

    X uses even counts; X90 uses multiples of four. The opposite final phase
    cancels a common readout offset. Randomized point order/repeated blocks
    reduce drift bias in finding the zero-slope amplitude. The half pulse stays
    fixed while the repeated gate amplitude varies, as in LeeQ's experiment.
    """
    if gate not in ('X', 'X90'):
        raise ValueError('Select independently calibrated X or X90')
    period = 2 if gate == 'X' else 4
    if (len(set(counts)) != len(counts) or len(counts) < 4
            or any(type(n) not in (int, np.int64) or n < 0 or n % period for n in counts)):
        raise ValueError('Ping-pong needs >=4 nonnegative counts aligned to complete rotations')
    if len(scales) < 1 or not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError('Amplitude scales must be finite and positive')
    if len(set(scales)) != len(scales) or type(blocks) is not int or blocks < 1:
        raise ValueError('Use unique scales and at least one block')
    rng = np.random.default_rng(seed)
    points = []
    for block in range(blocks):
        conditions = [(float(scale), int(n)) for scale in scales for n in counts]
        rng.shuffle(conditions)
        for scale, n in conditions:
            finals = [0.0, float(np.pi)]
            rng.shuffle(finals)
            for phase in finals:
                operations = [{'gate': 'Idle', 'time_us': 0.070}]
                operations += [{'gate': gate, 'amplitude_scale': scale,
                                'frequency_offset_mhz': center_offset_mhz}] * n
                operations += [{'gate': 'X90', 'amplitude_scale': x90_scale,
                                'phase_offset_rad': phase, 'frequency_offset_mhz': center_offset_mhz}]
                points.append({'block': block, 'repeated_gate': gate, 'amplitude_scale': scale,
                               'pulse_count': n, 'final_phase_rad': phase,
                               'center_offset_mhz': center_offset_mhz, 'operations': operations})
    return X6Y3Plan('pingpong', qubit, points, shots)


def iq_projection(result, readout_report):
    centers = [complex(*pair) for pair in readout_report['iq_centers']]
    contrast = centers[1] - centers[0]
    if abs(contrast) == 0:
        raise ValueError('Readout reference has zero contrast')
    return np.real((result['iq'] - centers[0]) / contrast)


def fit_ramsey(result, readout_report):
    """Use LeeQ's damped-frequency fit and its signed frequency update rule."""
    from leeq.theory.fits import fit_1d_freq_exp_with_cov

    projection = iq_projection(result, readout_report)
    y = projection.mean(axis=1)
    points = result['manifest']['points']
    times = np.array([p['delay_us'] for p in points])
    fit = fit_1d_freq_exp_with_cov(y, dt=times[1] - times[0])
    parameters = {k: {'value': float(v.n), 'std': float(v.s)} for k, v in fit.items() if k != 'Cov'}
    freq, uncertainty = abs(fit['Frequency'].n), fit['Frequency'].s
    amplitude, decay = fit['Amplitude'].n, fit['Decay'].n
    predicted = amplitude * np.exp(-times / decay) * np.sin(
        2 * np.pi * fit['Frequency'].n * times + fit['Phase'].n) + fit['Offset'].n
    r_squared = 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
    set_offset = points[0]['set_offset_mhz']
    correction = set_offset - freq
    acceptable = (np.isfinite([freq, uncertainty, decay, r_squared]).all()
                  and decay > 0 and r_squared > 0.5
                  and uncertainty < set_offset / 4 and abs(correction) < set_offset / 2)
    return {'model': 'LeeQ fit_1d_freq_exp_with_cov', 'parameters': parameters,
            'mean': y.tolist(), 'standard_error': (projection.std(axis=1, ddof=1) / np.sqrt(projection.shape[1])).tolist(),
            'r_squared': float(r_squared), 'frequency_mhz': float(freq),
            'frequency_std_mhz': float(uncertainty), 'correction_mhz': float(correction),
            'new_center_offset_mhz': float(points[0]['center_offset_mhz'] + correction),
            'accepted': bool(acceptable)}


def fit_pingpong(result, readout_report):
    """Fit phase-paired slopes, then interpolate the amplitude where slope is zero."""
    projection = iq_projection(result, readout_report)
    points = result['manifest']['points']
    scales = sorted({p['amplitude_scale'] for p in points})
    curves = []
    for scale in scales:
        x, y, pairs = [], [], []
        for i, p in enumerate(points):
            if p['amplitude_scale'] != scale or p['final_phase_rad'] != 0:
                continue
            matches = [j for j, other in enumerate(points)
                       if other['amplitude_scale'] == scale and other['block'] == p['block']
                       and other['pulse_count'] == p['pulse_count'] and other['final_phase_rad'] != 0]
            if len(matches) != 1:
                raise ValueError('Missing or duplicate opposite-phase control')
            j = matches[0]
            x.append(p['pulse_count'])
            y.append(float(projection[i].mean() - projection[j].mean()))
            pairs.append({'block': p['block'], 'pulse_count': p['pulse_count'], 'difference': y[-1]})
        fit, covariance = np.polyfit(x, y, 1, cov=True)
        curves.append({'amplitude_scale': scale, 'slope': float(fit[0]),
                       'slope_std': float(np.sqrt(covariance[0, 0])), 'intercept': float(fit[1]),
                       'paired_points': pairs})
    report = {'curves': curves, 'mean': projection.mean(axis=1).tolist(),
              'observable': 'Xp-minus-Xm final-phase paired IQ projection',
              'candidate_scale': None, 'candidate_std': None, 'bracketed': False}
    if len(curves) >= 3:
        errors = np.maximum([c['slope_std'] for c in curves], 1e-12)
        observed = np.array([c['slope'] for c in curves])
        slope_fit, covariance = np.polyfit(scales, observed, 1, w=1 / errors, cov='unscaled')
        reduced_chi2 = float(np.sum(((observed - np.polyval(slope_fit, scales)) / errors) ** 2)
                             / (len(scales) - 2))
        covariance *= max(1, reduced_chi2)
        k, b = slope_fit
        candidate = -b / k
        gradient = np.array([b / k ** 2, -1 / k])
        uncertainty = np.sqrt(max(0, gradient @ covariance @ gradient))
        bracketed = (min(scales) < candidate < max(scales) and k > 0
                     and min(c['slope'] for c in curves) < 0 < max(c['slope'] for c in curves))
        report.update(candidate_scale=float(candidate), candidate_std=float(uncertainty),
                      bracketed=bool(bracketed), slope_vs_amplitude=slope_fit.tolist(),
                      slope_fit_reduced_chi2=reduced_chi2)
    return report


def save_candidate(calibration, qubit_settings, path):
    """Write a separate local YAML, preserving original source and other qubits."""
    import yaml

    config = calibration.snapshot
    for qubit, values in qubit_settings.items():
        ge = config['single_qubit'][int(qubit)]['GE']
        ge['freq'] += values['frequency_offset_mhz'] * 1e6
        for gate in ('X', 'X90'):
            for pulse in ge[gate]['pulse']:
                if pulse['env'] != 'virtualz':
                    pulse['kwargs']['amp'] *= values[gate + '_scale']
    path = Path(path)
    if path.resolve() == calibration.path:
        raise ValueError('Candidate must not overwrite the supplied calibration')
    with path.open('x') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
