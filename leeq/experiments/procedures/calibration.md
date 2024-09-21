# Two-qubit Calibration

## Background

## Steps


# Two-qubit gate amplitude calibration
## Background
## Steps
- Pick a 
## Report
Report the 

# Two qubit gate frequency calibration
## Background
## Steps



# Full calibration of Single Qubit `dut`

## Background

This procedure should not be used when only calibrate a specific aspect of the qubit, such as frequency or amplitude.

## Steps

- Do full gate frequency calibration on `dut`
- Do full gate amplitude calibration on `dut`
- Do DRAG Calibration on `dut`

# Full Gate frequency calibration on `dut`

## Background

Gate frequency calibration is one of the essential steps when calibrating a qubit. This procedure is applicable to frequency
calibration of single qubit gates only.

This procedure is used to calibrate the frequency of the qubit gates. It should be used as a reference when the frequency calibration 
parameters are not known or need to be recalibrated.

## Steps

- Run simple Ramsey experiment on `dut`, with frequency offset 10 MHz, stop at 0.3us, step 0.005us.
- Run Ramsey experiment on `dut`, with frequency offset 1 MHz, stop at 3us, step 0.05us
- Run Ramsey experiment on `dut`, with frequency offset 0.1 MHz, stop at 30us, step 0.5us.

# Full Gate Amplitude Calibration on `dut`

## Background

This procedure is used to calibrate the amplitude of the qubit gates. It should be used as a reference when the amplitude calibration parameters are not known or
need to be recalibrated. It is only applicable to single qubit gates amplitude.

## Steps

- Conduct a Rabi experiment to determine the Rabi rate for rough amplitude calibration.
- Upon the successful completion of the Rabi experiment, run Pingpong experiment to do fine calibration. If failed directly goto Failure.