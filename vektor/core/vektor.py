"""
vektor/core/vektor.py

The beating heart of the Vektor package.
"""
import random
from typing import Callable, List, Optional, Tuple, Type


class Vektor:
    def __init__(
        self,
        matrix: Optional[List[List[float]]] = None,
        shape: Optional[Tuple[int, int]] = None,
        dtype: Type = float,
    ) -> None:
        """
        Initialize the Vektor instance with a given shape, data type, or an existing matrix.

        :param matrix: A list of lists representing a matrix. Used if shape is None.
        :param shape: A tuple representing the dimensions of the matrix. Used if matrix is None.
        :param dtype: The data type of the matrix elements, default is float.
        """
        if matrix is not None:
            self.matrix = matrix
            self.dtype = type(matrix[0][0])
        else:
            self.matrix = [
                [dtype(random.gauss(0, 1)) for _ in range(shape[1])]
                for _ in range(shape[0])
            ]
            self.dtype = dtype

    def __repr__(self) -> str:
        """
        String representation of the matrix.
        """
        return "\n".join(["\t".join(map(str, row)) for row in self.matrix])

    def __matmul__(self, other) -> "Vektor":
        result = [
            [sum(a * b for a, b in zip(row, col)) for col in zip(*other.matrix)]
            for row in self.matrix
        ]
        return Vektor(result)

    def transpose(self) -> "Vektor":
        """
        Transpose the matrix.

        :return: A new Vektor instance representing the transposed matrix.
        """
        return Vektor(
            [
                [self.matrix[j][i] for j in range(len(self.matrix))]
                for i in range(len(self.matrix[0]))
            ],
            self.dtype,
        )

    def dot(self, other: "Vektor") -> "Vektor":
        """
        Perform a dot product with another Vektor.

        :param other: The Vektor to dot with.
        :return: A new Vektor instance representing the result of the dot product.
        """
        if len(self.matrix[0]) != len(other.matrix):
            raise ValueError("Shapes are not aligned for dot product")
        return Vektor(
            [
                [sum(a * b for a, b in zip(row, col)) for col in zip(*other.matrix)]
                for row in self.matrix
            ],
            self.dtype,
        )

    def elementwise(
        self,
        other: "Vektor",
        operation: Callable[[float, float], float] = lambda x, y: x + y,
    ) -> "Vektor":
        """
        Perform an elementwise operation with another Vektor.

        :param other: The Vektor to operate with.
        :param operation: A callable taking two arguments and returning the result of an elementwise operation.
        :return: A new Vektor instance representing the result of the elementwise operation.
        """
        if len(self.matrix) != len(other.matrix) or len(self.matrix[0]) != len(
            other.matrix[0]
        ):
            raise ValueError(
                "Matrix shapes must be identical for elementwise operations"
            )
        return Vektor(
            [
                [
                    operation(self.matrix[i][j], other.matrix[i][j])
                    for j in range(len(self.matrix[0]))
                ]
                for i in range(len(self.matrix))
            ],
            self.dtype,
        )

    def shape(self) -> Tuple[int, int]:
        """
        Get the shape of the matrix.

        :return: A tuple representing the dimensions of the matrix.
        """
        return len(self.matrix), len(self.matrix[0])


# Example usage:
if __name__ == "__main__":
    nn_ops = Vektor(shape=(2, 3))
    print("Matrix:\n", nn_ops)
    print("Transpose:\n", nn_ops.transpose())
    print("Transpose:\n", nn_ops.transpose())
