# Two level Two-qubit calibration on `duts`
## Background
This procedure should not be used when only calibrate a specific aspect of the qubit, such as frequency or amplitude.
## Steps
- Run ConditionalStarkTwoQubitGateFrequencyAdvise with `duts`. Proceed to the next stage if exp_continue. If not exp_continue, go to Complete stage if there is a best frequency and go to Failed state if there is not.
- Do an Iterative two-qubit amplitude test at the advised frequency on `duts`. Go back the Stage 1 whether the result is.



# Iterative Two-qubit Amplitude test at `frequency` on `duts`
## Background
This experiment searches for the optimal amplitude for the two-qubit gate at `frequency` on `duts`. This is not a single step experiment, but an iterative one.
## Steps
- Run ConditionalStarkTwoQubitGateAmplitudeAdvise with `frequency` and `duts`. Proceed to the next stage if exp_continue. If not exp_continue, go to Complete stage if there is a best frequency and go to Failed state if there is not.
- Run ConditionalStarkTwoQubitGateAmplitudeAttempt with duts=`duts`, frequency=`frequency`, amplitude = new_amplitude_to_try. Go back the Stage 1 whether the result is.
## Results
Whether there is a success experiment or not. If so, what is the amplitude.