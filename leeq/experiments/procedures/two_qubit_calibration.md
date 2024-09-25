# Two-qubit Frequency test at `frequency` on `duts`
## Background
This experiment searches for the optimal amplitude for the two-qubit gate at `frequency` on `duts`.
## Steps
- Run ConditionalStarkTwoQubitGateAIParameterSearchAmplitude with duts=duts, frequency=`frequency`. Proceed to the next no matter the experiment fails or succeeds.
- Run ConditionalStarkTwoQubitGateAIParameterSearchAmplitude with duts=duts, frequency=`frequency` and the new amplitude. If failed 5 times, the experiment is considered failed.
## Results
Whether there is a success experiment or not. If so, what is the amplitude.