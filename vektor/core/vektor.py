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

    def __elementwise__(
        self, other: "Vektor", operation: Callable[[float, float], float]
    ) -> "Vektor":
        if self.shape != other.shape:
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
            dtype=self.dtype,
        )

    def __add__(self, other: "Vektor") -> "Vektor":
        return self.__elementwise__(other, lambda x, y: x + y)

    def __sub__(self, other: "Vektor") -> "Vektor":
        return self.__elementwise__(other, lambda x, y: x - y)

    def __mul__(self, other: "Vektor") -> "Vektor":
        return self.__elementwise__(other, lambda x, y: x * y)

    def __matmul__(self, other: "Vektor") -> "Vektor":
        if len(self.matrix[0]) != len(other.matrix):
            raise ValueError("Shapes are not aligned for matrix multiplication")

        return Vektor(
            [
                [sum(a * b for a, b in zip(row, col)) for col in zip(*other.matrix)]
                for row in self.matrix
            ],
            dtype=self.dtype,
        )

    @property
    def T(self) -> "Vektor":
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

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get the shape of the matrix.

        :return: A tuple representing the dimensions of the matrix.
        """
        return len(self.matrix), len(self.matrix[0])


# Example usage:
if __name__ == "__main__":
    import math

    class Neuron:
        def __init__(self, num_inputs: int):
            """
            Initialize the neuron with random weights and a bias.

            :param num_inputs: Number of input connections to the neuron.
            """
            self.weights = Vektor(shape=(1, num_inputs))  # Random weights
            self.bias = Vektor(matrix=[[random.gauss(0, 1)]])  # Random bias

        def activate(
            self, inputs: Vektor, activation_function: Callable[[float], float]
        ) -> float:
            """
            Activate the neuron with the given inputs and activation function.

            :param inputs: Input values as a Vektor.
            :param activation_function: A function to apply to the weighted sum.
            :return: The output of the neuron.
            """
            # This was self.weights @ inputs + self.bias
            weighted_sum = inputs @ self.weights.T + self.bias
            return activation_function(
                weighted_sum.matrix[0][0]
            )  # Applying the activation function

    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    # Create a neuron with 3 inputs
    neuron = Neuron(num_inputs=3)

    # Example input
    inputs = Vektor(matrix=[[0.5, -1, 0.2]])

    # Activate the neuron
    output = neuron.activate(inputs, sigmoid)
    print("Neuron output:", output)
