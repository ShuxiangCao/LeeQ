# Quick Start Guide: 10 Minutes to Launch

This guide is designed to walk you through the essential setup required to begin utilizing LeeQ, complete with built-in experiments. It is based on the premise that you are employing the [QubiC](https://lbl-qubic.gitlab.io/software/) system as the backend framework. For those without access to the physical hardware, there is an option to establish a high-level simulated backend to commence your experimentation.

## Installation Process

LeeQ operates alongside LabChronicle for the retention of experiment data. To get started, you need to install [LabChronicle](https://github.com/ShuxiangCao/LabChronicle) followed by the installation of LeeQ.

- To install LabChronicle: `pip install git+https://github.com/ShuxiangCao/LabChronicle`
- To install LeeQ: `pip install git+https://github.com/ShuxiangCao/LeeQ`

### Optional Configuration

There are two optional environment variables that can be set for further customization. If these are not specified, the software defaults to creating a directory within the working folder for the storage of experiment and calibration logs.

- `LAB_CHRONICLE_LOG_DIR`: Specifies the storage location for experiment log files.
- `LEEQ_CALIBRATION_LOG_PATH`: Determines the storage path for calibration logs, which include parameters like qubit frequency and pulse settings.

## Setting Up Your Experiment

Create a new file named `experiment_setup.py` in your working directory and include the following content:

```python

from leeq.setups.qubic_lbnl_setups import QubiCSingleBoardRemoteRPCSetup
from labchronicle import Chronicle
from leeq.experiments import setup

Chronicle().start_log()


class QubiCDemoSetup(QubiCSingleBoardRemoteRPCSetup):

    def __init__(self):
        super().__init__(
            name='qubic_demo_setup',
            rpc_uri='http://192.168.1.80:9095' # The RPC address for QubiC system
        )


q1_params = {
    'hrid':'Q1',
    'lpb_collections':{
        # Here we use f01 refer to the transition between the 0 and 1 state. 
        # Multiple different pulses can be defined here.
        'f01': {
            'type': 'SimpleDriveCollection',
            'freq': 4888.20, # in MHz
            'channel': 0, # Please refer to the QubiC LeeQ channel map for the detailed explanation
            'shape': 'blackman_drag',
            'amp': 0.21,
            'phase': 0., # in radius
            'width': 0.05, # in microseconds
            'alpha': 1e9,
            'trunc': 1.2
        },
        'f12': {
            'type': 'SimpleDriveCollection',
            'freq': 4888.205889560021-200,
            'channel': 0,
            'shape': 'blackman_drag',
            'amp': 0.21 / 5 * 4,
            'phase': 0.,
            'width': 0.025,
            'alpha': 1e9,
            'trunc': 1.2
        }
    },
    'measurement_primitives': {
        # Multiple different measurement pulse can be defined here.
        '0': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9997.6, # in MHz
            'channel': 1, # Please refer to the QubiC LeeQ channel map for the detailed explanation
            'shape': 'square',
            'amp': 0.06,
            'phase': 0., # in radius
            'width': 8, # in microseconds
            'trunc': 1.2,
            'distinguishable_states': [0, 1]
        },
        '1': {
            'type': 'SimpleDispersiveMeasurement',
            'freq': 9997.55,
            'channel': 1,
            'shape': 'square',
            'amp': 0.06,
            'phase': 0.,
            'width': 8,
            'trunc': 1.2,
            'distinguishable_states': [0, 1, 2]
        }
    }
}

# To configure more qubits, define more dictionary similar to the one above.  

setup_ = QubiCDemoSetup()
setup().register_setup(setup_)
```

## Load the environment into a jupyter notebook

Now create a new jupyter notebook, with the following code to initialize the notebook environment.

```python
import leeq
import sys
from scipy import optimize as so
from leeq.experiments.builtin import *
from leeq.utils.compatibility import *
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
```


Create the `TransmonElement` object for the parameters defined in the python file.

```python
# setup().start_live_monitor() # If the experiment code support live plot, uncomment this line to start the live monitor
setup().status().set_param("Shot_Number", 500)
setup().status().set_param("Shot_Period", 500)

dut_dict = {
    'Q1': {'Active': True, 'Tuneup': False,'FromLog':False, 'Params': q1_params}
} 

duts_dict = {}
for hrid, dd in dut_dict.items():
    if (dd['Active']):
        if (dd['FromLog']):
            dut = TransmonElement.load_from_calibration_log(dd['Params']['hrid'])
        else:
            dut = TransmonElement(name=dd['Params']['hrid'],parameters=dd['Params'])
            
        if (dd['Tuneup']):
            # Define your own tune up procedures.
            dut.save_calibration_log()
        else:
            # Run measurement calibration to create a GMM . 
            lpb_scan = (dut.get_c1('f01')['I'], dut.get_c1('f01')['X'])
            calib = MeasurementCalibrationMultilevelGMM(dut, mprim_index=0,sweep_lpb_list=lpb_scan)
        dut.print_config_info()
        duts_dict[hrid] = dut

dut = duts_dict['Q1']
```
## Start running built in experiments

Now we are ready to run built in experiment. For the available built in experiment please refer to the related
document page, and the example notebook for a full qubit tune up. As an example for resonator spectroscopy, we can run

```python
ResonatorSweepTransmissionWithExtraInitialLPB(dut,
            start = 9220.6-10,
            stop  = 9220.6+10,
            step = 0.5,
            num_avs = 10000,
            rep_rate = 0.0,
            mp_width = 8,
            amp=0.03
)
```
