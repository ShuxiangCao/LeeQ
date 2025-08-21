import itertools as it
from typing import List, Tuple, Union

import numpy as np


def generate_gellmann_matrix(j: int, k: int, dimension: int) -> np.ndarray:
    """
    Generate a Gellmann matrix of a specified size.

    Parameters:
        j (int): Index determining the row to modify in the matrix.
        k (int): Index determining the column to modify in the matrix.
        dimension (int): The dimension of the matrix (number of rows and columns).

    Returns:
        np.ndarray: A complex matrix representing the Gellmann matrix for the given indices.
    """

    if j > k:
        coordinates = [[j - 1, k - 1], [k - 1, j - 1]]
        values = [1, 1]  # Symmetric matrix element setup
    elif k > j:
        coordinates = [[j - 1, k - 1], [k - 1, j - 1]]
        values = [-1j, 1j]  # Anti-symmetric imaginary unit setup
    elif j == k and j < dimension:
        indices = list(range(j + 1))
        coordinates = [indices, indices]
        scale_factor = np.sqrt(2 / (j * (j + 1)))
        values = np.array(list(it.repeat(1 + 0j, j)) + [-j + 0j]) * scale_factor
    else:
        indices = list(range(dimension))
        coordinates = [indices, indices]
        values = list(it.repeat(1 + 0j, dimension))  # Identity-like matrix

    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for value, row, col in zip(values, *coordinates):
        matrix[row][col] = value

    return matrix


def generate_gellmann_basis(dimension: int, use_sparse: bool = False) -> List[np.ndarray]:
    """
    Generate the full basis of Gellmann matrices for the given dimension.

    Parameters:
        dimension (int): The dimension of the matrices.
        use_sparse (bool): Flag to generate sparse matrices (Not used here).

    Returns:
        List[np.ndarray]: A list of all Gellmann matrices for the specified dimension.
    """
    basis = [generate_gellmann_matrix(j, k, dimension, use_sparse)
             for j, k in it.product(range(1, dimension + 1), repeat=2)]
    return basis
