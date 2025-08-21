Here are paths to import and loading relevant modules and classes in LeeQ:

```python
from leeq.chronicle import log_and_record, register_browser_function
from leeq import Experiment, Sweeper, SweepParametersSideEffectFactory, basic_run, setup
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlock
from leeq.setups.built_in.setup_simulation_high_level import HighLevelSimulationSetup
from leeq.utils.compatibility import prims
from leeq import basic_run as basic
from leeq import Experiment, Sweeper, basic_run
from leeq.core.elements.built_in.qudit_transmon import TransmonElement
from leeq.utils.compatibility import *
import matplotlib.pyplot as plt
from leeq.core.primitives.logical_primitives import LogicalPrimitiveBlockSerial, LogicalPrimitiveBlock
from leeq import Experiment, Sweeper, SweepParametersSideEffectFactory, ExperimentManager
from leeq.theory.fits import fit_1d_freq, fit_exp_decay_with_cov, fit_1d_freq_exp_with_cov, fit_2d_freq_with_cov
```