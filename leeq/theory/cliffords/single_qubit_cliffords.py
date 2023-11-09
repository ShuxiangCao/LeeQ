import numpy as np
from leeq.utils import setup_logging

logger = setup_logging(__name__)

_single_qubit_clifford_map = {
    0: ['I'],
    1: ['X'],
    2: ['Y'],
    3: ['Ym'],
    4: ['Yp'],
    5: ['Xp'],
    6: ['Xm'],
    7: ['Y', 'X'],
    8: ['Yp', 'X'],
    9: ['Xp', 'Ym'],
    10: ['Xm', 'Ym'],
    11: ['Ym', 'X'],
    12: ['Xm', 'Yp'],
    13: ['Xp', 'Yp'],
    14: ['Ym', 'Xm'],
    15: ['Yp', 'Xm'],
    16: ['Y', 'Xm'],
    17: ['Ym', 'Xp'],
    18: ['Y', 'Xp'],
    19: ['Yp', 'Xp'],
    20: ['Xm', 'Yp', 'Xp'],
    21: ['Xm', 'Ym', 'Xp'],
    22: ['Xp', 'Yp', 'Xp'],
    23: ['Xp', 'Ym', 'Xp'],
}


def get_clifford_from_id(index: int):
    """
    Get the clifford from the clifford id.

    Parameters:
        index (int): The clifford id.

    Returns:
        list: The composition of clifford.
    """

    assert index in _single_qubit_clifford_map, f'Clifford {index} is not supported.'
    return _single_qubit_clifford_map[index]


# Define type alias for a complex floating number array
CFloatArray = np.ndarray

# Define Pauli matrices
pI = np.array([[1, 0], [0, 1]], dtype=complex)
pX = np.array([[0, 1], [1, 0]], dtype=complex)
pY = np.array([[0, -1j], [1j, 0]], dtype=complex)
pZ = np.array([[1, 0], [0, -1]], dtype=complex)

# Define a tolerance for identifying the inverse gate
TOLERANCE = 1e-9

import numpy as np
import scipy.linalg as sl

# Define type alias for a complex floating number array
CFloatArray = np.ndarray

# Define Pauli matrices
pI = np.array([[1, 0], [0, 1]], dtype=complex)
pX = np.array([[0, 1], [1, 0]], dtype=complex)
pY = np.array([[0, -1j], [1j, 0]], dtype=complex)
pZ = np.array([[1, 0], [0, -1]], dtype=complex)


def create_clifford_gates(pauli_gate: CFloatArray, phase: float) -> CFloatArray:
    """
    Create a Clifford gate using the exponential of a Pauli matrix.

    Parameters:
    pauli_gate (CFloatArray): The Pauli matrix to use for creating the gate.
    phase (float): The phase angle for the exponential.

    Returns:
    CFloatArray: The resulting Clifford gate.
    """
    return sl.expm(-0.5j * phase * np.pi * pauli_gate)


# Generate the Clifford gates for X, Y, and Z with different phases
C1p_X = create_clifford_gates(pX, 1.0)
C1p_Y = create_clifford_gates(pY, 1.0)
C1p_Xp = create_clifford_gates(pX, 0.5)
C1p_Xm = create_clifford_gates(pX, -0.5)
C1p_Yp = create_clifford_gates(pY, 0.5)
C1p_Ym = create_clifford_gates(pY, -0.5)
C1p_Z = create_clifford_gates(pZ, 1.0)
C1p_Zp = create_clifford_gates(pZ, 0.5)
C1p_Zm = create_clifford_gates(pZ, -0.5)

# Initialize an array to hold 24 Clifford gates
NC1 = 24
C1 = np.zeros(shape=(NC1, 2, 2), dtype=complex)

# Populate the array with Clifford gates based on the Pauli matrices
C1[0] = pI
C1[1] = C1p_X
C1[2] = C1p_Y
C1[3] = C1p_Ym
C1[4] = C1p_Yp
C1[5] = C1p_Xp
C1[6] = C1p_Xm
C1[7] = C1p_X.dot(C1p_Y)
C1[8] = C1p_X.dot(C1p_Yp)
C1[9] = C1p_Ym.dot(C1p_Xp)
C1[10] = C1p_Ym.dot(C1p_Xm)
C1[11] = C1p_X.dot(C1p_Ym)
C1[12] = C1p_Yp.dot(C1p_Xm)
C1[13] = C1p_Yp.dot(C1p_Xp)
C1[14] = C1p_Xm.dot(C1p_Ym)
C1[15] = C1p_Xm.dot(C1p_Yp)
C1[16] = C1p_Xm.dot(C1p_Y)
C1[17] = C1p_Xp.dot(C1p_Ym)
C1[18] = C1p_Xp.dot(C1p_Y)
C1[19] = C1p_Xp.dot(C1p_Yp)
C1[20] = C1p_Xp.dot(C1p_Yp).dot(C1p_Xm)
C1[21] = C1p_Xp.dot(C1p_Ym).dot(C1p_Xm)
C1[22] = C1p_Xp.dot(C1p_Yp).dot(C1p_Xp)
C1[23] = C1p_Xp.dot(C1p_Ym).dot(C1p_Xp)


# The same process is applied to create a different set of gates (VZC1)
# similar to the creation of C1 gates, with the dot products involving
# different combinations of the gates, and using Z phase gates as well.
def get_c1(i: int) -> CFloatArray:
    """
    Retrieve a Clifford gate from the precomputed set based on its index.

    Parameters:
    i (int): The index of the Clifford gate to retrieve.

    Returns:
    CFloatArray: The requested Clifford gate.
    """
    return C1[i]


def create_clifford_gates(pauli_gate: CFloatArray, phase: float) -> CFloatArray:
    """
    Create a Clifford gate using the exponential of a Pauli matrix.

    Parameters:
    - pauli_gate (CFloatArray): The Pauli matrix to use for creating the gate.
    - phase (float): The phase angle for the exponential.

    Returns:
    - CFloatArray: The resulting Clifford gate.
    """
    return sl.expm(-0.5j * phase * np.pi * pauli_gate)


def find_inverse_C1(a: CFloatArray, cliff_set: str) -> int:
    """
    Find the index of the Clifford gate that is the inverse of the given matrix.

    Parameters:
    - a (CFloatArray): The matrix to invert, should be a member of the Clifford set.
    - cliff_set (str): The set of Clifford gates to use ('XY' or 'VZX').

    Returns:
    - int: The index of the inverse Clifford gate if found, otherwise None.
    """

    if cliff_set not in ['XY']:
        msg = "Invalid Clifford set. Must be 'XY' or 'VZX'."
        logger.error(msg)
        raise ValueError(msg)

    if cliff_set == 'VZX':
        msg = "Clifford set 'VZX' not yet implemented."
        logger.error(msg)
        raise NotImplementedError(msg)

    min_diff = float('inf')
    for i in range(NC1):
        # Select the appropriate set of gates
        test = a.dot(get_c1(i) if cliff_set == 'XY' else get_VZc1(i))

        # Normalize if the first element is not zero to avoid division by zero
        if not np.isclose(test[0, 0], 0):
            test /= test[0, 0]

        # Calculate the deviation from the identity matrix
        test -= pI
        deviation = np.sum(np.square(test.real)) + np.sum(np.square(test.imag))
        if deviation < min_diff:
            min_diff = deviation
            if min_diff < TOLERANCE:
                return i

    # If no inverse is found within the tolerance, print the minimum deviation found
    msg = "Inverse search failed. Minimum deviation: {}".format(min_diff)
    logger.error(msg)
    raise ValueError(msg)


def append_inverse_C1(c1_index_list: list, clifford_set: str = 'XY', test_inverse: bool = False) -> list:
    """
    Append the index of the inverse Clifford gate to the given sequence of gates.

    Parameters:
    - c1_index_list (list): The sequence of gate indices to append the inverse to.
    - clifford_set (str): The set of Clifford gates to use ('XY' or 'VZX').
    - test_inverse (bool): If True, test the resulting matrix for being the inverse.

    Returns:
    - list: The sequence with the appended index of the inverse gate.
    """
    U = pI
    for index in c1_index_list:
        U = (get_c1(index) if clifford_set == 'XY' else get_VZc1(index)).dot(U)

    inverse_index = find_inverse_C1(U, cliff_set=clifford_set)

    return list(c1_index_list) + [inverse_index]
