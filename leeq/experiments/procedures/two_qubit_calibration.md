# Two level Two-qubit calibration on `duts`
## Background
This procedure should not be used when only calibrate a specific aspect of the qubit, such as frequency or amplitude.
## Steps
Stage 1: Run ConditionalStarkTwoQubitGateFrequencyAdvise with `duts`. Proceed to the next stage whatever the result is.
Stage 2: Do an Iterative two-qubit amplitude test at the advised frequency on `duts`. If failed, go back the Stage 1. If failed 5 times, proceed to the Fail stage. If succeeded, proceed to Complete.



# Iterative Two-qubit Amplitude test at `frequency` on `duts`
## Background
This experiment searches for the optimal amplitude for the two-qubit gate at `frequency` on `duts`. This is not a single step experiment, but an iterative one.
## Steps
Stage 1: Run ConditionalStarkTwoQubitGateAmplitudeAdvise with `frequency` and `duts`. Proceed to the next stage whatever the result is.
Stage 2: Run ConditionalStarkTwoQubitGateAmplitudeAttempt with duts=`duts`, frequency=`frequency`, amplitude = new_amplitude_to_try. If failed, go back the Stage 1. If failed 5 times, proceed to the Fail stage. If succeeded, proceed to Complete.
## Results
Whether there is a success experiment or not. If so, what is the amplitude.