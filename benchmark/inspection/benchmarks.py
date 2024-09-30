import numpy as np
import json

from mllm import display_chats

import leeq
import json
from simulated_setup import *  # Change to your customized setup file
import numpy as np
from scipy import optimize as so
from leeq.experiments.builtin import *
import plotly.graph_objects as go
from labchronicle import log_and_record, register_browser_function

from leeq.utils.compatibility import *
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.experiments.builtin import *
from pprint import pprint
from simulated_setup import *


benchmark_config = {
    "enable_few_shot": True
}

dut_dict = {
    'Q1': {'Active': True, 'Tuneup': False, 'FromLog': False, 'Params': configuration_a},
    'Q2': {'Active': True, 'Tuneup': False, 'FromLog': False, 'Params': configuration_b}
}

duts_dict = {}
for hrid, dd in dut_dict.items():
    if (dd['Active']):
        if (dd['FromLog']):
            dut = TransmonElement.load_from_calibration_log(dd['Params']['hrid'])
        else:
            dut = TransmonElement(name=dd['Params']['hrid'], parameters=dd['Params'])

        dut.print_config_info()
        duts_dict[hrid] = dut

dut = None

# Global configuration for different benchmark settings
EXPERIMENT_CONFIGS = {
    'run_benchmark_res_spec': {
        'success': {'offset': 0, 'quiescent_state_distribution': None, 'amp': 0.001},
        'failure': {'offset': 1000, 'quiescent_state_distribution': np.array([0.5, 0.5, 0.0, 0.0]), 'amp': 0.00}
    },
    'run_single_benchmark_gmm': {
        'success': {'amp_increment': 0.05, 'freq_increment': 0, 'freq_variation': 0.5, 'amp': 0.2},
        'failure': {'amp_increment': 0.05, 'freq_increment': 1, 'freq_variation': 0.5, 'amp': 0.2,
                    'quiescent_state_distribution': np.array([0.34, 0.33, 0.33, 0.0])}
    },
    'run_single_benchmark_rabi': {
        'success': {'offset': 2, 'quiescent_state_distribution': None, 'amp_range': 0.2, "amp": 0.1},
        'failure': {'offset': 1002, 'quiescent_state_distribution': np.array([0.5, 0.5, 0.0, 0.0]), 'amp_range': 0.2,
                    "amp": 0.1}
    },
    'run_single_benchmark_drag': {
        'success': {'sign': 1, 'quiescent_state_distribution': np.array([0.95, 0.05, 0.0, 0.0]), 'center': -0.005},
        'failure': {'sign': -1, 'quiescent_state_distribution': np.array([0.51, 0.49, 0.0, 0.0]), 'center': -0.005}
    }
}


def setup_simulation(readout_frequency, config):
    simulation_setup(readout_frequency=readout_frequency,
                     quiescent_state_distribution=config.get('quiescent_state_distribution', None))
    setup().status().set_param("AIAutoInspectPlots", config.get('AIAutoInspectPlots', False))
    ExperimentManager().status().set_parameter("Plot_Result_In_Jupyter", config.get('Plot_Result_In_Jupyter', False))


def save_results(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def run_benchmarks(save_path_prefix, num_samples, benchmark_func, config_key, extra_config={}):
    print(num_samples)
    inspections = {}
    try:
        with open(f"{save_path_prefix}.json", 'r') as f:
            inspections = json.load(f)
    except:
        pass
    for i in range(len(inspections), num_samples):
        np.random.seed(i)
        config = EXPERIMENT_CONFIGS[benchmark_func.__name__][config_key]
        config.update(extra_config)

        exps = benchmark_func(i, config, inspections)

        for exp_type, exp in exps.items():

            print('exp._plot_function_images.items()', exp._plot_function_images.items())

            for key, image in exp._plot_function_images.items():
                image.save(f"./images/{save_path_prefix}_{i}_{key}_{exp_type}.png")
                print('image saved to ' + f"{save_path_prefix}_{i}_{key.split('.')[-1]}_{exp_type}.png")

        pprint(inspections[i])
        save_results(f"./result/{save_path_prefix}.json", inspections)


# Benchmark functions
def run_benchmark_res_spec(index, config, inspections):
    readout_frequency = 500 * np.random.random() + 9000
    setup_simulation(readout_frequency=readout_frequency, config=config)

    readout_sweep_width = 10 + 1000 * np.random.random()
    readout_scan_offset = config['offset'] + readout_sweep_width / 3 * np.random.random()
    center_frequency = readout_frequency + readout_scan_offset

    scan_params = {
        'start': center_frequency - readout_sweep_width / 2,
        'stop': center_frequency + readout_sweep_width / 2,
        'step': readout_sweep_width / 500,
        'num_avs': 100,
        'rep_rate': 0.0,
        'mp_width': 1,
        'amp': config['amp']
    }

    # pprint(config)
    # pprint(scan_params)

    dut = duts_dict['Q1']

    inspections[index] = {}
    exps = {}

    res_raw = ResonatorSweepTransmissionRaw(dut_qubit=dut, **scan_params)
    inspections[index]['image_zero_shot'] = extract_results_from_experiment(res_raw)
    exps['raw'] = res_raw
    if benchmark_config['enable_few_shot']:
        res_image_prompt = ResonatorSweepTransmissionImageFewShot(dut_qubit=dut, **scan_params)
        inspections[index]['image_few_shot'] = extract_results_from_experiment(res_image_prompt)
        exps['image_few_shot'] = res_image_prompt
    return exps


def run_single_benchmark_gmm(index, config, inspections):
    setup_simulation(readout_frequency=9645, config=config)

    scan_params = {
        'amp': config['amp'],
        'freq': 9645.5 + config['freq_increment'] + 0.5 * np.random.random()
    }

    dut = duts_dict['Q1']

    inspections[index] = {}
    exps = {}

    res_raw = MeasurementCalibrationMultilevelGMMRaw(dut=dut, **scan_params)
    inspections[index]['image_zero_shot'] = extract_results_from_experiment(res_raw)
    exps['raw'] = res_raw
    if benchmark_config['enable_few_shot']:
        res_image_prompt = MeasurementCalibrationMultilevelGMMImageFewShot(dut=dut, **scan_params)
        inspections[index]['image_few_shot'] = extract_results_from_experiment(res_image_prompt)
        exps['image_few_shot'] = res_image_prompt
    return exps



def run_single_benchmark_rabi(index, config, inspections):
    readout_frequency = 500 * np.random.random() + 9000
    setup_simulation(readout_frequency, config)

    scan_params = {
        'amp': config['amp'] + 0.2 * np.abs(np.random.random(1)),
        'stop': 0.5,
        'step': 0.01
    }

    dut = duts_dict['Q1']
    inspections[index] = {}
    exps = {}
    res_raw = NormalisedRabiDataValidityCheckRaw(dut_qubit=dut, **scan_params)
    inspections[index]['image_zero_shot'] = extract_results_from_experiment(res_raw)
    exps['raw'] = res_raw
    if benchmark_config['enable_few_shot']:
        res_image_prompt = NormalisedRabiDataValidityCheckImageFewShot(dut_qubit=dut, **scan_params)
        inspections[index]['image_few_shot'] = extract_results_from_experiment(res_image_prompt)
        exps['image_few_shot'] = res_image_prompt
    return exps


def run_single_benchmark_drag(index, config, inspections):
    readout_frequency = 500 * np.random.random() + 9000
    setup_simulation(readout_frequency, config)

    span_width = span_width = 0.006 + 0.002 * np.random.random(1)[0]
    center = config['center'] + span_width * 0.2 * np.random.random(1)[0]

    scan_params = {
        'inv_alpha_start': center - span_width / 2,
        'inv_alpha_stop': center + span_width / 2
    }

    dut = duts_dict['Q1']
    inspections[index] = {}
    exps = {}
    res_raw = DragCalibrationSingleQubitMultilevelRaw(dut=dut, **scan_params)
    inspections[index]['image_zero_shot'] = extract_results_from_experiment(res_raw)
    exps['raw'] = res_raw
    if benchmark_config['enable_few_shot']:
        with display_chats(1):
            res_image_prompt = DragCalibrationSingleQubitMultilevelImageFewShot(dut=dut, **scan_params)
            inspections[index]['image_few_shot'] = extract_results_from_experiment(res_image_prompt)
            exps['image_few_shot'] = res_image_prompt
    return exps