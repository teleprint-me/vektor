"""
Module: vektor.core.dtype

Description: This module defines custom data types used in the Vektor library. These types
             are designed to represent the basic building blocks of vector and matrix
             operations, especially in the context of transformer models.

Types:
    - TScalar: Represents a scalar value. Scalars are the basic numerical entities used in
               mathematical operations. In Vektor, a scalar can be a boolean, integer,
               floating-point number, or a complex number. This flexibility allows Vektor
               to handle a wide range of mathematical computations.

    - TVector: Represents a vector. In the context of Vektor, a vector is a list of scalars.
               Vectors are fundamental to many mathematical operations, especially in
               representing data and performing linear algebraic computations.

    - TMatrix: Represents a matrix. A matrix in Vektor is defined as a list of vectors.
               Matrices are two-dimensional arrays, crucial for complex mathematical
               operations and transformations.

    - TTensor: Represents a 3D tensor. In Vektor, a 3D tensor is conceptualized as a list
               of matrices. Tensors are used in more complex data structures and operations,
               particularly in the fields of machine learning and computer vision.

    - TTensorND: Represents an N-dimensional tensor. While Vektor primarily focuses on up to
                 3D tensors, this type is provided for potential future extensions and for
                 users who might need to represent higher-dimensional data structures.
                 The N-dimensional tensor is represented as a list of 'Any' type, offering
                 flexibility at the expense of specificity.

    - TShape: Represents the shape of a tensor. The shape is defined as a tuple of integers,
              each integer representing the size of the tensor in that dimension. This type
              is crucial for operations that need to be aware of the tensor's dimensions,
              such as reshaping or slicing.

Note: The 'T' prefix in type names is used to differentiate these custom types from potential
      class names in the Vektor library, ensuring clarity and avoiding naming conflicts.
"""
from typing import Any, List, Tuple, Union

# Basic scalar types
TScalar = Union[bool, int, float, complex]

# Vector, Matrix, and general Tensor
TVector = List[TScalar]
TMatrix = List[TVector]
TTensor = List[TMatrix]
TTensorND = List[Any]  # Can be extended for higher dimensions

# Shape of a tensor (n-dimensional)
TShape = Tuple[int, ...]
