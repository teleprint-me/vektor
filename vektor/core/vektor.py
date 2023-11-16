"""
vektor/core/vektor.py

The beating heart of the Vektor package.
"""
import random
from typing import Callable, List, Optional, Tuple, Type, Union

from vektor.core.dtype import TMatrix, TScalar, TShape, TTensor, TVector


class Vektor:
    def __init__(
        self,
        data: Optional[TTensor] = None,
        shape: Optional[TShape] = None,
        dtype: Type = float,
    ) -> None:
        """
        Initialize the Vektor instance with a given shape, data type, or an existing matrix.
        """
        self._dtype = dtype

        if data is not None:
            if isinstance(data, dtype):  # Scalar
                self._matrix = [[dtype(data)]]
            elif all(isinstance(item, dtype) for item in data):  # 1D Vector
                self._matrix = [[dtype(item)] for item in data]  # Column vector
            else:  # 2D Matrix
                self._matrix = [[dtype(item) for item in row] for row in data]
        else:
            # Initialize with zeros if shape is given
            self._matrix = (
                [[dtype(0) for _ in range(shape[1])] for _ in range(shape[0])]
                if shape
                else []
            )

    def __getitem__(self, key) -> TMatrix:
        return self._matrix[key]

    def __setitem__(self, index: int, data: TMatrix) -> None:
        if index >= len(self._matrix):
            self._matrix.append(data)
        else:
            self._matrix[index] = data

    def __len__(self) -> int:
        return len(self._matrix)

    def __contains__(self, data: TMatrix) -> bool:
        return data in self._matrix

    def __repr__(self) -> str:
        """
        String representation of the matrix.
        """
        # NOTE: Keep the old line just in case.
        # return "\n".join(["\t".join(map(str, row)) for row in self.matrix])
        return "\n".join(str(row) for row in self.matrix)

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
    def dtype(self) -> TScalar:
        return self._dtype

    @property
    def matrix(self) -> TMatrix:
        return self._matrix

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
            dtype=self.dtype,
        )

    @property
    def shape(self) -> TShape:
        """
        Get the shape of the matrix.

        :return: A tuple representing the dimensions of the matrix.
        """
        return len(self.matrix), len(self.matrix[0])

    @shape.setter
    def shape(self, new_shape: TShape) -> None:
        """
        Reshape the vector to a new shape if possible.

        :param new_shape: A tuple representing the desired new shape.
        :raises ValueError: If the new shape is not compatible with the current number of elements.
        """
        # Calculate the total elements for the new shape
        new_total_elements = new_shape[0] * new_shape[1]

        # Calculate the current total elements
        current_total_elements = self.shape[0] * self.shape[1]

        if new_total_elements != current_total_elements:
            raise ValueError(
                "New shape is not compatible with the current number of elements"
            )

        # Create a new matrix with the new shape
        new_matrix = [
            [self.matrix[j][i] for j in range(new_shape[0])]
            for i in range(new_shape[1])
        ]

        # Update the matrix and shape attributes
        self._matrix = new_matrix
        self._shape = new_shape


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
