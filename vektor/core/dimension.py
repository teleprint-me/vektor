"""
vektor/core/dimension.py

I need to know the data type for each item in the matrix if we're instantiating from scratch.

I don't need to know the data type if we're already given the information we need, e.g. the list it self.

I need to instantiate a zeroed out list upon creation if it's empty.

I need to know the shape if nothing is given.
This pattern should hold across other functions as well.

I might want to implement something that allows us to introduce noise as well.
Maybe a helper function for instantiation random values.

  matrix = [
      [
          dtype(random.gauss(0, 1))
          for _ in range(shape[1])
      ]
      for _ in range(shape[0])
  ]

This would be similar to numpy.random.randn or numpy.Generator.normal

  rng = np.random.default_rng()  # Random number generator instance
  size = (self.input_dim, self.output_dim)
  distribution = rng.standard_normal(size=size, dtype=self.dtype)

It's better to start thinking about things like this early on rather than later.
"""
import random
from typing import Callable, List, Optional

from vektor.core.dtype import TMatrix, TScalar, TShape, TVector


def initialize_2d_zero_matrix(
    shape: Optional[TShape] = None,
    dtype: Optional[TScalar] = float,
) -> TMatrix:
    """
    Create and initialize a 2D matrix filled with zeros.

    Parameters:
    - shape (Optional[TShape]): A tuple representing the desired shape of the matrix
                                as (rows, columns). If None, an empty matrix is returned.
    - dtype (Optional[TScalar]): The data type of matrix elements. Defaults to float.

    Returns:
    - TMatrix: A 2D matrix (list of lists) initialized with zeros.

    Raises:
    - ValueError: If the provided shape is not a tuple of two non-negative integers.

    Example:
    >>> initialize_2d_zero_matrix(shape=(2, 3), dtype=int)
    [[0, 0, 0], [0, 0, 0]]

    >>> initialize_2d_zero_matrix()
    []
    """
    if shape is not None:
        if not isinstance(shape, tuple):
            raise ValueError("Shape must be a tuple.")

        if len(shape) != 2:
            raise ValueError("Shape must be a tuple of two elements (rows, columns).")

        rows, cols = shape
        if not (isinstance(rows, int) and rows >= 0):
            raise ValueError("Number of rows must be a non-negative integer.")

        if not (isinstance(cols, int) and cols >= 0):
            raise ValueError("Number of columns must be a non-negative integer.")

    rows, cols = shape if shape is not None else (0, 0)
    return [[dtype(0) for _ in range(cols)] for _ in range(rows)]


def add_noise_to_matrix(
    matrix: TMatrix,
    callback: Optional[Callable] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> TMatrix:
    """
    Add noise to each element of a matrix.

    Parameters:
    - matrix (TMatrix): The input matrix.
    - callback (Optional[Callable]): A callback function for generating noise. Defaults to random.gauss.
    - seed (Optional[int]): The seed for the random number generator. If None, no seed is used.
    - **kwargs: Additional keyword arguments for the noise generation function.

    Returns:
    - TMatrix: A new matrix with noise added to its elements.

    Raises:
    - ValueError: If callback is not a callable function or matrix is empty or improperly structured.

    Example:
    >>> add_noise_to_matrix(matrix, noise_level=0.1, callback=random.gauss, mu=0.0, sigma=1.0)
    """
    if not matrix:
        raise ValueError("Matrix is empty.")

    if not all(len(row) == len(matrix[0]) for row in matrix):
        raise ValueError("All rows in the matrix must have the same length.")

    if seed is not None:
        random.seed(seed)

    if callback is None:
        # Default to random.gauss if no callback is provided
        callback = random.gauss
        kwargs.setdefault("mu", 0.0)
        kwargs.setdefault("sigma", 1.0)

    if not callable(callback):
        raise ValueError("Callback must be a callable function.")

    noisy_matrix = [[item + callback(**kwargs) for item in row] for row in matrix]
    return noisy_matrix


def create_column_matrix_from_vector(
    vector: TVector, dtype: TScalar = float
) -> TMatrix:
    """
    Convert a 1D vector into a column matrix.

    Parameters:
    - vector (TVector): A list of scalar values.
    - dtype (TScalar): The data type of matrix elements. Defaults to float.

    Returns:
    - TMatrix: A column matrix (2D list) representing the vector.

    Example:
    >>> create_column_matrix_from_vector([1, 2, 3])
    [[1], [2], [3]]
    """
    return [[dtype(item)] for item in vector]


def create_row_matrix_from_vector(vector: TVector, dtype: TScalar = float) -> TMatrix:
    """
    Convert a 1D vector into a row matrix.

    Parameters:
    - vector (TVector): A list of scalar values.
    - dtype (TScalar): The data type of matrix elements. Defaults to float.

    Returns:
    - TMatrix: A row matrix (2D list) representing the vector.

    Example:
    >>> create_row_matrix_from_vector([1, 2, 3])
    [[1, 2, 3]]
    """
    return [list(map(dtype, vector))]


def create_2d_matrix(
    data: Optional[TMatrix] = None,
    shape: Optional[TShape] = None,
    dtype: TScalar = float,
) -> TMatrix:
    """
    Create a 2D matrix from a list of lists or by initializing a matrix of a given shape.

    Parameters:
    - data (Optional[TMatrix]): List of lists representing the matrix.
                                If provided, overrides shape.
    - shape (Optional[TShape]): Shape of the matrix to initialize if data is not provided.
    - dtype (TScalar): The data type of matrix elements. Defaults to float.

    Returns:
    - TMatrix: A 2D matrix.

    Raises:
    - ValueError: If both data and shape are provided, or if neither is provided.
                  If data rows are of inconsistent lengths.

    Example:
    >>> create_2d_matrix(data=[[1, 2], [3, 4]])
    [[1.0, 2.0], [3.0, 4.0]]

    >>> create_2d_matrix(shape=(2, 2))
    [[0.0, 0.0], [0.0, 0.0]]
    """
    if data is not None and shape is not None:
        raise ValueError("Cannot specify both data and shape.")

    if data is None and shape is None:
        raise ValueError("Must specify either data or shape.")

    if data is not None:
        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows in the data must have the same length.")
        return [[dtype(item) for item in row] for row in data]

    return initialize_2d_zero_matrix(shape, dtype)


def create_3d_matrix(
    data: List[TMatrix],
    shape: Optional[TShape],
    dtype: TScalar = float,
) -> TMatrix:
    """
    Create a 3D matrix from a list of lists of lists.

    :param data: List of lists of lists representing the matrix.
    :return: 3D matrix.
    """
    return data


def reshape_matrix(matrix: TMatrix, shape: TShape, new_shape: TShape) -> TMatrix:
    """
    Reshape a matrix to a new shape if possible.

    :param matrix: The matrix to reshape.
    :param new_shape: A tuple representing the desired new shape.
    :raises ValueError: If the new shape is not compatible with the current number of elements.
    :return: Reshaped matrix.
    """
    # Calculate total elements for the new shape
    new_total_elements = new_shape[0] * new_shape[1]

    # Calculate current total elements
    current_total_elements = len(matrix) * len(matrix[0])

    if new_total_elements != current_total_elements:
        raise ValueError(
            "New shape is not compatible with the current number of elements"
        )

    # Create a new matrix with the new shape
    new_matrix = [
        [matrix[j][i] for j in range(new_shape[0])] for i in range(new_shape[1])
    ]

    return new_matrix
