# Full calibration of Single Qubit `dut`

## Background

This procedure should not be used when only calibrate a specific aspect of the qubit, such as frequency or amplitude.

## Steps

- Full gate frequency calibration on `dut`
- Full gate amplitude calibration on `dut`
- DRAG Calibration on `dut`

# Full Gate frequency calibration on `dut`

## Background

This procedure is applicable to frequency calibration of single qubit gates only.

## Steps

- Run simple Ramsey experiment on `dut`, with frequency offset 10 MHz, stop at 0.3us, step 0.005us. Retry 3 times if failed.
- Run Ramsey experiment on `dut`, with frequency offset 1 MHz, stop at 3us, step 0.05us. Retry 3 times if failed.
- Run Ramsey experiment on `dut`, with frequency offset 0.1 MHz, stop at 30us, step 0.5us. Retry 3 times if failed.

# Full Gate Amplitude Calibration on `dut`

## Background

This procedure is used to calibrate the amplitude of the qubit gates. It should be used as a reference when the amplitude calibration parameters are not known or
need to be recalibrated. It is only applicable to single qubit gates amplitude.

## Steps

- Conduct a Rabi experiment with amp=1.0 to determine the Rabi rate for rough amplitude calibration. If failed, adjust the parameters and retry 3 times.
- Run Pingpong experiment. If failed directly goto Failure.