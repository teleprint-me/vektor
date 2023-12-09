# Vektor Core Data Types

The `vektor/core/dtype.py` module defines essential data types within the Vektor library. These types are designed to facilitate vector and matrix operations, particularly in transformer models.

## Types

### TScalar 
Represents a scalar value, such as boolean, integer, floating-point number, or complex number. This versatility enables Vektor to handle a wide range of mathematical computations.

### TVector
Represents a vector, which is essentially a list of scalars. Vectors are fundamental for mathematical operations and data representation.

### TMatrix
Represents a matrix, defined as a list of vectors. Matrices play a pivotal role in complex mathematical operations and transformations.

### TTensor
Represents a 3D tensor, conceptualized as a list of matrices. Tensors are vital for more advanced data structures, commonly used in machine learning and computer vision.

### TTensorND
Represents an N-dimensional tensor, offering flexibility for potential future extensions and handling higher-dimensional data structures.

### TShape
Represents the shape of a tensor, defined as a tuple of integers. It's crucial for operations involving tensor dimensions, such as reshaping or slicing.

## Definitions

```python
from typing import Any, List, Tuple, Union


TScalar = Union[bool, int, float, complex]
TVector = List[TScalar]
TMatrix = List[TVector]
TTensor = List[TMatrix]
TTensorND = List[Any]  # Can be extended for higher dimensions
TShape = Tuple[int, ...]
```

## Usage Examples

```python
from vektor.core.dtype import TVector, TMatrix, TScalar


def add_vectors(v1: TVector, v2: TVector) -> TVector:
    return [a + b for a, b in zip(v1, v2)]


def multiply_matrix_scalar(m: TMatrix, s: TScalar) -> TMatrix:
    return [[element * s for element in row] for row in m]
```
