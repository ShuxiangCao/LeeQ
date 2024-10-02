# Bell state preparation on `duts`

## Steps
- Run Full calibration of Single Qubit `duts[0]`
- Run Full calibration of Single Qubit `duts[1]`
- Run Two level Two-qubit calibration on `duts`
- Implement Bell state Tomography on `duts`
 

# GHZ state preparation on `duts`

## Steps
- Run Full calibration of Single Qubit `duts[0]`
- Run Full calibration of Single Qubit `duts[1]`
- Run Full calibration of Single Qubit `duts[2]`
- Run Two level Two-qubit calibration on `duts[0:1]`
- Run Two level Two-qubit calibration on `duts[1:2]`
- Implement three qubits GHZ state Tomography `duts` 