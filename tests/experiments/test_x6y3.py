"""X6Y3 waveform/timing parity and native-engine acquisition without networking."""

import copy
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

pytest.importorskip('qcal')
pytest.importorskip('distproc.executable')

from leeq.core.context import ExperimentContext
from leeq.core.primitives.built_in.simple_drive import SimpleDispersiveMeasurement
from leeq.experiments.experiments import ExperimentManager
from leeq.experiments.x6y3 import (
    X6Y3Plan, acquire_plan, amplitude_plan, analyze_readout, analyze_sweep,
    compile_plan, ramsey_plan, readout_plan, t1_plan,
)
from leeq.experiments.x6y3_validation import verify_plan_parity
from leeq.setups.huracan import create_huracan_setup
from leeq.setups.x6y3 import X6Y3Calibration

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures'


@pytest.fixture(autouse=True)
def isolated_offline_manager(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail('X6Y3 tests must not contact the network')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)
    manager = ExperimentManager()
    monkeypatch.setattr(manager, '_setups', {})
    monkeypatch.setattr(manager, '_default_setup', None)


@pytest.fixture
def calibration():
    return X6Y3Calibration(FIXTURES / 'x6y3_calibration.yaml')


@pytest.fixture
def setup():
    return create_huracan_setup(FIXTURES / 'x6y3_channel_config.json')


@pytest.mark.parametrize('qubit', [0, 1])
def test_archived_protocols_match_qcal(calibration, setup, qubit):
    plans = [readout_plan(qubit, blocks=1),
             amplitude_plan(qubit, scales=[0.7, 1, 1.3], n_gates=4),
             amplitude_plan(qubit, scales=[1], n_gates=1),
             ramsey_plan(qubit, delays_us=[0, 0.123, 1], detunings_mhz=[-2.5, 2.5]),
             t1_plan(qubit, delays_us=[0, 17.25, 350]),
             X6Y3Plan('independent_X', qubit, [{'operations': [{'gate': 'X'}]}], 4)]
    for plan in plans:
        assert verify_plan_parity(calibration, setup, plan)['pass']


def test_independent_gates_and_calibrated_demodulation(calibration, setup):
    plan = X6Y3Plan('X90_and_X', 0, [{'operations': [{'gate': 'X90'}, {'gate': 'X'}]}], 4)
    program = compile_plan(calibration, setup, plan)[0]['instructions']
    pulses = [p for p in program if p['name'] == 'pulse']
    assert pulses[0]['twidth'] == 35e-9
    assert pulses[1]['twidth'] == 70e-9
    assert pulses[1]['amp'] != 2 * pulses[0]['amp']
    assert pulses[0]['freq'] == pytest.approx(5490403426.796941, abs=1e-6, rel=0)
    assert pulses[-1]['phase'] == -98.7618  # qcal forwards radians unchanged
    assert pulses[-1]['twidth'] == pytest.approx(656.67e-9)
    assert {'name': 'delay', 't': 398.57e-9, 'scope': ['Q0.rdlo']} in program


def test_config_and_snapshot_are_not_mutated(calibration, setup):
    before = calibration.path.read_bytes()
    snapshot = calibration.snapshot
    snapshot['single_qubit'][0]['GE']['freq'] = 1
    compile_plan(calibration, setup, amplitude_plan(scales=[0.8, 1.2]))
    assert calibration.path.read_bytes() == before
    assert calibration.snapshot['single_qubit'][0]['GE']['freq'] != 1


@pytest.mark.parametrize('mutation', ['LO', 'active', 'esp', 'channel', 'amplitude', 'delay'])
def test_invalid_config_rejected(calibration, tmp_path, mutation):
    config = calibration.snapshot
    if mutation == 'LO':
        config['hardware']['qubit_LO'] = 15e9
    elif mutation == 'active':
        config['reset']['active']['enable'] = True
    elif mutation == 'esp':
        config['readout']['esp']['enable'] = True
    elif mutation == 'channel':
        config['single_qubit'][0]['GE']['X']['pulse'][0]['channel'] = 'Q7.qdrv'
    elif mutation == 'amplitude':
        config['readout'][0]['amp'] = 2
    else:
        config['readout'][0]['demod']['delay'] = -1
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        X6Y3Calibration(path)


def test_metadata_mismatch_rejected(calibration, setup):
    setup.channel_metadata['Q0.qdrv']['elem_params']['interp_ratio'] = 8
    with pytest.raises(ValueError, match='sample rate'):
        compile_plan(calibration, setup, readout_plan())


@pytest.mark.parametrize('make', [lambda: readout_plan(qubit=2), lambda: readout_plan(shots=0),
                                lambda: amplitude_plan(n_gates=2), lambda: amplitude_plan(scales=[np.nan]),
                                lambda: ramsey_plan(delays_us=[-1]), lambda: t1_plan(delays_us=[])])
def test_invalid_plans_rejected(make):
    with pytest.raises(ValueError):
        make()


def test_legacy_measurement_defaults_unchanged(setup):
    measurement = SimpleDispersiveMeasurement('legacy', dict(channel=1, freq=100, width=0.5,
        amp=0.1, phase=0.3, shape='square', distinguishable_states=[0, 1]))
    context = ExperimentContext('legacy')
    setup._compiler.compile_lpb(context, measurement)
    drive, delay, demod = context.instructions['circuits']
    assert delay == {'name': 'delay', 't': 200e-9, 'scope': ['Q0']}
    assert demod['phase'] == drive['phase'] == 0.3
    assert demod['twidth'] == drive['twidth']
    assert demod['env'][0]['env_func'] == 'square'


def test_demodulation_frequency_change_requests_reload(calibration, setup):
    measurement = calibration.measurement(0)
    setup._compiler.compile_lpb(ExperimentContext('first'), measurement)
    context = ExperimentContext('second')
    setup._compiler.compile_lpb(context, measurement)
    assert not any(context.instructions['dirtiness'].values())
    params = measurement.get_parameters()['demodulation']
    params['freq'] += 1.0
    measurement.update_parameters(demodulation=params)
    setup._compiler.compile_lpb(context, measurement)
    assert all(context.instructions['dirtiness'].values())


@pytest.mark.parametrize('key,value', [('width', 0), ('delay', -1), ('phase', np.nan), ('amp', 2)])
def test_invalid_demodulation_rejected(calibration, setup, key, value):
    measurement = calibration.measurement(0)
    params = measurement.get_parameters()['demodulation']
    params[key] = value
    measurement.update_parameters(demodulation=params)
    with pytest.raises(ValueError, match='demodulation'):
        setup._compiler.compile_lpb(ExperimentContext('invalid'), measurement)


def test_native_engine_raw_iq_and_persistence(calibration, setup, tmp_path):
    ExperimentManager().register_setup(setup)
    previous = setup.status.get_parameters()
    plan = readout_plan(shots=8, blocks=1)
    arrays = [np.arange(8).reshape(8, 1) + 3j, -np.arange(8).reshape(8, 1) - 4j]
    runner = Mock(side_effect=[[{'Q0.rdlo': value}] for value in arrays])
    setup._runner = SimpleNamespace(run_circuit_batch=runner)
    result = acquire_plan(calibration, setup, plan, tmp_path / 'result')
    np.testing.assert_array_equal(result['iq'], np.stack(arrays)[:, :, 0])
    assert runner.call_count == 2
    assert setup.status.get_parameters() == previous
    manifest = json.loads((result['output_dir'] / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['completed_points'] == 2
    assert manifest['calibration_updated'] is False
    assert (result['output_dir'] / 'calibration.yaml').read_bytes() == calibration.path.read_bytes()
    assert runner.call_args.kwargs['reads_per_shot'] == 1


def test_failure_keeps_completed_points_without_retry(calibration, setup, tmp_path):
    ExperimentManager().register_setup(setup)
    previous = setup.status.get_parameters()
    runner = Mock(side_effect=[[{'Q0.rdlo': np.ones((8, 1), complex)}], ConnectionError('stopped')])
    setup._runner = SimpleNamespace(run_circuit_batch=runner)
    output = tmp_path / 'failure'
    with pytest.raises(ConnectionError):
        acquire_plan(calibration, setup, readout_plan(shots=8), output)
    assert runner.call_count == 2
    assert setup.status.get_parameters() == previous
    assert (output / 'point-0000.npz').exists()
    assert not (output / 'point-0001.npz').exists()
    assert json.loads((output / 'manifest.json').read_text())['completed_points'] == 1
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'failed'


def test_held_out_assignment_and_t1_fit():
    rng = np.random.default_rng(123)
    plan = readout_plan(shots=500)
    iq = np.array([p['prepared_state'] * (3 + 2j)
                   + 0.05 * (rng.normal(size=500) + 1j * rng.normal(size=500)) for p in plan.points])
    report = analyze_readout({'iq': iq, 'manifest': {'points': plan.points}})
    assert report['accuracy'] > 0.99
    assert report['training_blocks'] == [0]
    assert report['held_out_blocks'] == [1]
    plan = t1_plan()
    iq = np.array([(3 + 2j) * np.exp(-p['delay_us'] / 52)
                   + 0.05 * (rng.normal(size=500) + 1j * rng.normal(size=500)) for p in plan.points])
    fit = analyze_sweep({'iq': iq, 'manifest': {'points': plan.points, 'experiment': 't1'}}, report)
    assert fit['fits'][0]['T1_us'] == pytest.approx(52, abs=1)
    assert fit['calibration_updated'] is False
