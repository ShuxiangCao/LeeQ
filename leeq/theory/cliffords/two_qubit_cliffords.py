import numpy as np
import scipy.linalg as sl

from leeq.utils import setup_logging

logger = setup_logging(__name__)

pI = np.array([[1., 0.], [0., 1.]], dtype='cfloat')
pX = np.array([[0, 1], [1, 0]], dtype='cfloat')
pY = np.array([[0, -1j], [1j, 0]], dtype='cfloat')
pZ = np.array([[1, 0], [0, -1]], dtype='cfloat')
pII = np.kron(pI, pI)
pIX = np.kron(pI, pX)
pIY = np.kron(pI, pY)
pIZ = np.kron(pI, pZ)
pXI = np.kron(pX, pI)
pXX = np.kron(pX, pX)
pXY = np.kron(pX, pY)
pXZ = np.kron(pX, pZ)
pYI = np.kron(pY, pI)
pYX = np.kron(pY, pX)
pYY = np.kron(pY, pY)
pYZ = np.kron(pY, pZ)
pZI = np.kron(pZ, pI)
pZX = np.kron(pZ, pX)
pZY = np.kron(pZ, pY)
pZZ = np.kron(pZ, pZ)

uI = pI  # I
uXp = sl.expm(-0.25j * np.pi * pX)  # X + pi/2
uYm = sl.expm(0.25j * np.pi * pY)  # Y - pi/2
uX = sl.expm(-0.5j * np.pi * pX)  # X + pi/2
uII = np.kron(uI, uI)
uIXp = np.kron(uI, uXp)
uIYm = np.kron(uI, uYm)
uIX = np.kron(uI, uX)
uXpI = np.kron(uXp, uI)
uXpXp = np.kron(uXp, uXp)
uXpYm = np.kron(uXp, uYm)
uXpX = np.kron(uXp, uX)
uYmI = np.kron(uYm, uI)
uYmXp = np.kron(uYm, uXp)
uYmYm = np.kron(uYm, uYm)
uYmX = np.kron(uYm, uX)
uXI = np.kron(uX, uI)
uXXp = np.kron(uX, uXp)
uXYm = np.kron(uX, uYm)
uXX = np.kron(uX, uX)

C1p_I = pI
C1p_X = sl.expm(-0.50j * np.pi * pX)
C1p_Y = sl.expm(-0.50j * np.pi * pY)
C1p_Xp = sl.expm(-0.25j * np.pi * pX)
C1p_Xm = sl.expm(+0.25j * np.pi * pX)
C1p_Yp = sl.expm(-0.25j * np.pi * pY)
C1p_Ym = sl.expm(+0.25j * np.pi * pY)
# def Z gates
C1p_Z = sl.expm(-0.50j * np.pi * pZ)
C1p_Zp = sl.expm(-0.25j * np.pi * pZ)
C1p_Zm = sl.expm(+0.25j * np.pi * pZ)

NC1 = 24
C1 = np.zeros(shape=(NC1, 2, 2), dtype='cfloat')
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

# Constants for the size and limit of the Clifford group operations
NC2 = 11520
NC2_lim = 24 * 24

C2 = None

# Definitions of specific Clifford operations using matrix multiplication
# and Kronecker products
ZXm = sl.expm(+0.25j * np.pi * pZX)
CNOTlike = ZXm
iSWAPlike = ZXm.dot(np.kron(C1p_Ym, C1p_Ym)).dot(ZXm)
SWAPlike = ZXm.dot(
    np.kron(
        C1p_Xp,
        C1p_Ym.dot(C1p_Xp))).dot(ZXm).dot(
    np.kron(
        C1p_Ym,
        C1p_Ym)).dot(ZXm)

# Initialize a placeholder for a set of two-qubit Clifford operations
C2p = np.zeros(shape=(4, 4, 4), dtype='complex')
C2p[0] = pII
C2p[1] = CNOTlike
C2p[2] = iSWAPlike
C2p[3] = SWAPlike

# ZX construction of cliffords - for implimenting virtual Z gates
# may be more efficient construction available!
VZC1 = np.zeros(shape=(NC1, 2, 2), dtype='cfloat')
VZC1[0] = pI
VZC1[1] = C1p_Xp.dot(C1p_Zp)
VZC1[2] = C1p_Zm.dot(C1p_Xm)
VZC1[3] = C1p_X
VZC1[4] = C1p_Zm.dot(C1p_Xm).dot(C1p_Zp).dot(C1p_Xm)
VZC1[5] = C1p_Zm.dot(C1p_Xm).dot(C1p_Z)
VZC1[6] = C1p_Zm.dot(C1p_X).dot(C1p_Zp)
VZC1[7] = C1p_Zm.dot(C1p_Xm).dot(C1p_Zp).dot(C1p_Xp)
VZC1[8] = C1p_Xp.dot(C1p_Zm).dot(C1p_Xp).dot(C1p_Zp)
VZC1[9] = C1p_Z
VZC1[10] = C1p_Z.dot(C1p_Xp).dot(C1p_Zp)
VZC1[11] = C1p_Zp.dot(C1p_Xm)
VZC1[12] = C1p_Zp.dot(C1p_Xp).dot(C1p_Zp)
VZC1[13] = C1p_Xm
VZC1[14] = C1p_Zp
VZC1[15] = C1p_Zm.dot(C1p_Xm).dot(C1p_Zp)
VZC1[16] = C1p_Xp
VZC1[17] = C1p_X.dot(C1p_Zp)
VZC1[18] = C1p_Zm.dot(C1p_Xm).dot(C1p_Zp).dot(C1p_X)
VZC1[19] = C1p_Xp.dot(C1p_Zm).dot(C1p_X).dot(C1p_Zp)
VZC1[20] = C1p_Xm.dot(C1p_Z).dot(C1p_Xp).dot(C1p_Zp)
VZC1[21] = C1p_Zm.dot(C1p_Xp).dot(C1p_Zp)
VZC1[22] = C1p_Xm.dot(C1p_Zm).dot(C1p_X).dot(C1p_Zp)
VZC1[23] = C1p_Zm


def get_VZc1(i):
    return VZC1[i]


def get_c2_vz(n: int) -> np.ndarray:
    """
    Retrieves a Clifford operation from the VZ set based on the index.

    Args:
        n (int): The index of the Clifford operation to retrieve.

    Returns:
        np.ndarray: The Clifford operation as a 4x4 complex matrix.
    """
    global C2VZ
    if C2VZ is None:
        # Build the C2VZ group if it hasn't been initialized yet
        C2VZ = np.zeros(shape=(NC2, 4, 4), dtype='complex')
        for i in range(NC2):
            c2type, q1c, q2c, q1s, q2s = get_c2_info(i)
            C2VZ[i] = np.kron(VZC1[q1s], VZC1[q2s]
                              ) @ C2p[c2type] @ np.kron(VZC1[q1c], VZC1[q2c])
    return C2VZ[n]


def get_c2_xy(n: int) -> np.ndarray:
    """
    Retrieves a Clifford operation from the XY set based on the index.

    Args:
        n (int): The index of the Clifford operation to retrieve.

    Returns:
        np.ndarray: The Clifford operation as a 4x4 complex matrix.
    """
    global C2
    if C2 is None:
        # Build the C2 group if it hasn't been initialized yet
        C2 = np.zeros(shape=(NC2, 4, 4), dtype='complex')
        for i in range(NC2):
            c2type, q1c, q2c, q1s, q2s = get_c2_info(i)
            C2[i] = np.kron(
                C1[q1s], C1[q2s]) @ C2p[c2type] @ np.kron(C1[q1c], C1[q2c])
    return C2[n]


def get_c2(n: int, cliff_set: str = 'XY') -> np.ndarray:
    """
    Retrieves a Clifford operation based on the index and the specified set (XY or VZX).

    Args:
        n (int): The index of the Clifford operation to retrieve.
        cliff_set (str): The set from which to retrieve the Clifford operation ('XY' or 'VZX').

    Returns:
        np.ndarray: The Clifford operation as a 4x4 complex matrix.
    """
    if cliff_set == 'XY':
        return get_c2_xy(n)
    elif cliff_set == 'VZX':
        return get_c2_vz(n)
    else:
        raise ValueError("Invalid Clifford set specified. Use 'XY' or 'VZX'.")


def get_c2_info(i: int) -> tuple:
    """
    Computes the configuration of a two-qubit Clifford operation based on its index.

    Args:
        i (int): The index of the Clifford operation.

    Returns:
        tuple: A tuple containing information about the Clifford operation (type, control qubit configs, swap configs).
    """
    Q2C = i % 24
    Q1C = (i // 24) % 24
    tmp = 19 - (i // (24 * 24)) % 20
    if tmp == 19:
        # Single-qubit Clifford operations
        c2type = 0
        Q2S = 0
        Q1S = 0
    elif tmp == 18:
        # SWAP-like operation
        c2type = 3
        Q2S = 0
        Q1S = 0
    elif tmp < 9:
        # CNOT-like operation
        c2type = 1
        Q2S = tmp % 3
        Q1S = tmp // 3 % 3
    else:
        # iSWAP-like operation
        c2type = 2
        Q2S = tmp % 3
        Q1S = tmp // 3 % 3
    # Adjust the swap configurations
    Q1S = 0 if Q1S == 0 else 14 if Q1S == 1 else 13
    Q2S = 0 if Q2S == 0 else 14 if Q2S == 1 else 13
    return c2type, Q1C, Q2C, Q1S, Q2S


CNOT_INDEX = 11429
CZ_INDEX = 10299
TOLERANCE = 1e-9
inverse_c2_mem = {}


def find_inverse_C2(a: np.ndarray, cliff_set: str = 'XY') -> int:
    """
    Finds the index of the Clifford operation that is the inverse of the given matrix.

    Args:
        a (np.ndarray): The matrix for which to find the inverse Clifford operation.
        cliff_set (str): The set from which to retrieve the Clifford operation ('XY' or 'VZX').

    Returns:
        int: The index of the inverse Clifford operation.
    """
    for i in range(NC2):
        inv_matrix = get_c2(i, cliff_set=cliff_set)
        diff = a @ inv_matrix
        diff = diff * np.exp(-1.j * np.angle(diff[0, 0]))
        if np.allclose(diff, np.eye(4), atol=TOLERANCE):
            return i
    raise AssertionError()
    return None


def append_inverse_C2(
        c2_index_list: list,
        cliff_set: str = 'XY') -> np.ndarray:
    """
    Appends the index of the inverse Clifford operation to a list of indices.

    Args:
        c2_index_list (list): A list of indices of Clifford operations.
        cliff_set (str): The set from which to retrieve the Clifford operation ('XY' or 'VZX').

    Returns:
        np.ndarray: An updated list of indices including the inverse operation.
    """
    U = pII
    for index in c2_index_list:
        U = get_c2(index, cliff_set=cliff_set) @ U
    inverse_index = find_inverse_C2(U, cliff_set=cliff_set)
    result = np.append(c2_index_list, inverse_index)
    return result
