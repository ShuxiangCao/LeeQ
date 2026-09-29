import json

import numpy as np
import pytest

from tests.experiments.test_x6y3 import calibration, setup, isolated_offline_manager
from leeq.experiments.x6y3 import compile_plan
from leeq.experiments.x6y3_tuneup import (
    RAMSEY_STAGES, leeq_ramsey_plan, pingpong_plan, fit_ramsey, fit_pingpong, save_candidate,
)
from leeq.experiments.x6y3_validation import verify_plan_parity


def test_three_scan_example_grids_and_negative_final_phase(calibration, setup):
    for stage, (offset, stop, step) in enumerate(RAMSEY_STAGES):
        plan = leeq_ramsey_plan(0, stage, center_offset_mhz=.2)
        assert len(plan.points) == 60
        np.testing.assert_allclose([p['delay_us'] for p in plan.points], np.arange(0, stop, step))
        assert plan.points[0]['operations'][0]['frequency_offset_mhz'] == .2 + offset
        assert plan.points[0]['operations'][-1]['phase_offset_rad'] == np.pi
        plan.points = [plan.points[0], plan.points[17], plan.points[-1]]
        assert verify_plan_parity(calibration, setup, plan)['pass']


@pytest.mark.parametrize('gate,counts', [('X', [0, 2, 4, 6]), ('X90', [0, 4, 8, 12])])
def test_pingpong_native_protocol_and_parity(calibration, setup, gate, counts):
    plan = pingpong_plan(1, gate, [.99, 1.01], counts, blocks=1, center_offset_mhz=-.05)
    assert len(plan.points) == 16
    for point in plan.points:
        assert len(point['operations']) == point['pulse_count'] + 2
        assert point['operations'][-1]['gate'] == 'X90'
    assert verify_plan_parity(calibration, setup, plan)['pass']


def test_pingpong_recovers_zero_slope_amplitude():
    plan = pingpong_plan(0, 'X90', [.985, 1, 1.015], [0, 4, 8, 12], shots=1024, blocks=2)
    rng = np.random.default_rng(9)
    data = []
    target = 1.004
    for point in plan.points:
        phase_sign = 1 if point['final_phase_rad'] == 0 else -1
        mean = .5 + phase_sign * .5 * np.sin(point['pulse_count'] * np.pi / 2
                                            * (point['amplitude_scale'] / target - 1))
        data.append(mean + rng.normal(0, .05, 1024))
    result = {'iq': np.asarray(data, complex), 'manifest': {'points': plan.points}}
    report = fit_pingpong(result, {'iq_centers': [[0, 0], [1, 0]]})
    assert report['bracketed']
    assert report['candidate_scale'] == pytest.approx(target, abs=.0002)


def test_ramsey_frequency_update_sign():
    plan = leeq_ramsey_plan(0, 1, center_offset_mhz=.1)
    times = np.array([p['delay_us'] for p in plan.points])
    rng = np.random.default_rng(1)
    means = .5 - .4 * np.cos(2 * np.pi * .95 * times) * np.exp(-times / 15)
    data = means[:, None] + rng.normal(0, .03, (len(times), 1024))
    report = fit_ramsey({'iq': data.astype(complex), 'manifest': {'points': plan.points}},
                        {'iq_centers': [[0, 0], [1, 0]]})
    assert report['accepted']
    assert report['new_center_offset_mhz'] == pytest.approx(.15, abs=.001)


def test_candidate_is_separate_and_preserves_independent_gates(calibration, tmp_path):
    original = calibration.path.read_bytes()
    output = tmp_path / 'candidate.yaml'
    save_candidate(calibration, {0: {'frequency_offset_mhz': .1, 'X_scale': .98, 'X90_scale': 1.01}}, output)
    from leeq.setups.x6y3 import X6Y3Calibration
    candidate = X6Y3Calibration(output).snapshot
    source = calibration.snapshot
    assert candidate['single_qubit'][1] == source['single_qubit'][1]
    assert candidate['readout'] == source['readout']
    assert candidate['single_qubit'][0]['GE']['freq'] == source['single_qubit'][0]['GE']['freq'] + 1e5
    assert calibration.path.read_bytes() == original
    with pytest.raises(ValueError):
        save_candidate(calibration, {}, calibration.path)
