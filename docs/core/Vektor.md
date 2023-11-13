
# Vektor Class Documentation

## Overview
The `Vektor` class is a fundamental component of the Vektor package, designed to facilitate core matrix operations, particularly for applications in neural networks. It provides a flexible and efficient way to represent and manipulate matrices with various operations.

## Initialization
`Vektor` can be initialized either by passing an existing matrix or by specifying the shape and data type.

### Syntax
```python
Vektor(matrix: Optional[List[List[float]]] = None, shape: Optional[Tuple[int, int]] = None, dtype: Type = float)
```

- `matrix`: A list of lists representing the matrix.
- `shape`: A tuple representing the matrix's dimensions (rows, columns).
- `dtype`: The data type of the matrix elements.

## Core Methods

### Matrix Addition (`__add__`)
Performs elementwise addition with another `Vektor`.

### Matrix Subtraction (`__sub__`)
Performs elementwise subtraction with another `Vektor`.

### Elementwise Multiplication (`__mul__`)
Performs elementwise multiplication with another `Vektor`.

### Matrix Multiplication (`__matmul__`)
Performs matrix multiplication (dot product) with another `Vektor`.

### Transpose (`T`)
Property to get the transpose of the matrix.

### Shape
Property to get the shape (dimensions) of the matrix.

## Example Usage
```python
from vektor.core.vektor import Vektor

# Initialize with shape
A = Vektor(shape=(2, 3))

# Initialize with existing matrix
B = Vektor(matrix=[[1, 2], [3, 4], [5, 6]])

# Elementwise addition
C = A + B

# Matrix multiplication
D = A @ B.T
```

## Conclusion
The `Vektor` class provides a robust and intuitive way to handle matrix operations, making it an essential tool for various applications, especially in the field of neural networks.
