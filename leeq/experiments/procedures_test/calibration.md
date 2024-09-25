
# Two-qubit calibration on `duts`
## Background
This procedure should not be used when only calibrate a specific aspect of the qubit, such as frequency or amplitude.
## Steps
- Do an Iterative two-qubit amplitude test at `7000` on `duts`. Proceed to Completion no matter the experiment fails or succeeds.



# Iterative Two-qubit Amplitude test at `frequency` on `duts`
## Background
This experiment searches for the optimal amplitude for the two-qubit gate at `frequency` on `duts`. This is not a single step experiment, but an iterative one.
## Steps
- Run ConditionalStarkTwoQubitGateAIParameterSearchAmplitude with duts=duts, frequency=`frequency`. Proceed to the next no matter the experiment fails or succeeds.
- Run ConditionalStarkTwoQubitGateAIParameterSearchAmplitude with duts=duts, frequency=`frequency` and the amplitude equals the new_amplitude. If failed 5 times, the experiment is considered failed.
## Results
Whether there is a success experiment or not. If so, what is the amplitude.