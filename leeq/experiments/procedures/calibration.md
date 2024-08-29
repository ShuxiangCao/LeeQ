
# Complete Calibrating Single Qubit `dut`

## Background

The process of recalibrating a single qubit is crucial to maintaining optimal quantum computation performance. This recalibration process involves three key steps: frequency calibration, amplitude calibration, and DRAG parameter calibration. Each step is aimed at refining different aspects of the qubit's operational parameters to ensure accurate and reliable qubit control. The sequence of these steps is critical, as each builds on the stability achieved in the previous step.

## Steps

- Do gate frequency calibration on `dut`
- Do gate amplitude calibration on `dut`
- Do DRAG Calibration on `dut`


# Gate frequency calibration on `dut`

## Background

Gate frequency calibration is one of the essential steps when calibrating a qubit.

Ramsey experiment can predict the qubit frequency different to the frequency I am driving it. First I guess a qubit frequency (which already set in the system), and assume the difference is no more than 10 MHz. Therefore I run a ramsey experiment with frequency offset 10 MHz. Then I wish to do a more accurate calibration by increase the experiment time, and reduce the offset to 1MHz. If this experiment failed and show a value more than 3 MHz its likely that the initial guess is more than 10MHz away from the qubit. Therefore we go back and run experiment at 20MHz offset again. After it succeeded, we do a fine calibration with offset 0.1MHz.

## Steps

- Run simple Ramsey experiment on `dut`, with frequency offset 10 MHz, stop at 0.3us, step 0.005us.
- Run Ramsey experiment on `dut`, with frequency offset 1 MHz, stop at 3us, step 0.05us
- If the second experiment obtained a frequency offset more than 3MHz, go back to the first step, set frequency offset to 20MHz. and try again.
- Run Ramsey experiment on `dut`, with frequency offset 0.1 MHz, stop at 30us, step 0.5us.

# Gate Amplitude Calibration on `dut`

## Background

Gate Amplitude calibration is one of the essential steps when calibrating a qubit.

To accurately calibrate the amplitude of the control pulses for our qubit gates, we start with a Rabi oscillation experiment. This experiment helps determine the amplitude required to perform a full rotation on the Bloch sphere. Successful determination of the Rabi frequency from these measurements will indicate the optimal amplitude setting for the qubit gates.

After successfully calibrating the Rabi frequency, we proceed to Pingpong amplitude calibration using the default parameters. This secondary calibration further refines our control over the qubit by adjusting the amplitudes based on the results from the Rabi experiment, ensuring more precise and reliable gate operations.

## Steps

- Conduct a Rabi experiment to determine the Rabi frequency: Start pulse duration at 0.01 microseconds, step 0.001 microseconds, stop at 0.15 microseconds.
- If the above experiment failed, re-do it and adjust parameters based on visual instructions.
- Upon the successful completion of the Rabi experiment, run Pingpong amplitude calibration with default parameters.

# DRAG Parameter Calibration on `dut`

## Background

DRAG calibration is one of the essential steps in when calibrating a qubit.

Derivative Removal by Adiabatic Gate (DRAG) parameter calibration is essential for minimizing errors in single qubit gates due to the non-idealities of the control pulses, such as leakage to non-computational states. We will run the calibration using default parameters for the single qubit pulse.  If the visual inspection fails, reset the alpha value to 0 and try again. if it failed again call for human assistance.

## Steps

- Begin by running the DRAG parameter calibration on `dut` using default settings: Execute the calibration sequence with predefined parameters.
- If the calibration fails: Set the alpha parameter to 0 and try again.
- If failed again, call for human assistance.

