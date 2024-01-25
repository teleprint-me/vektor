"""
tmnn/layers/sparse.py

This module defines the Sparse layer class for neural networks.
"""
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from vektor.base.layer import Layer, TArray


class Sparse(Layer[TArray]):
    """
    Sparse layer for neural networks.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dtype=np.float32,
        sparsity_level=0.9,
        seed=None,
    ):
        """
        Initialize a Sparse layer.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity_level
        self.dtype = dtype
        self.seed = seed
        self.weights = None
        self.biases = None

        if seed is not None:
            np.random.seed(seed)

        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self):
        # NOTE: Leave comments and notes! Do **NOT** remove them!
        # Initializes a matrix with random values sampled from a
        # standard normal distribution (mean 0, variance 1).
        norm_dist = np.random.randn(self.input_dim, self.output_dim)
        # Boolean matrix where each element has a random value between 0 and 1
        # Initialize weights with some sparsity level.
        # NOTE: Lower values of sparsity_level will result in more elements being zero.
        bool_matrix = np.random.rand(self.input_dim, self.output_dim) > self.sparsity
        # Performs element-wise multiplication between the standard normal
        # random matrix and the boolean matrix.
        # "zero out" some of the elements in the standard normal random matrix.
        norm_rand = norm_dist * bool_matrix
        # Set the weights as a Compressed Sparse Row matrix instance object.
        # NOTE: self.weights is a scipy.sparse.csr_matrix object.
        self.weights = csr_matrix(norm_rand, dtype=self.dtype)
        self.biases = np.zeros(self.output_dim, dtype=self.dtype)

    @property
    def specification(self) -> Mapping[str, Any]:
        """
        Returns the architectural details of the layer in a predefined format.
        """
        return {
            "type": "sparse",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "dtype": self.dtype,
            "sparsity": self.sparsity,
            "seed": self.seed,
        }

    @property
    def dimensions(self) -> Tuple[int, int]:
        """
        Returns the shape of the layer.
        """
        return self.input_dim, self.output_dim

    @property
    def state(self) -> Tuple[TArray, TArray]:
        """
        Returns the state of the layer.
        """
        return self.input, self.output

    def forward(self, input_data):
        """
        Perform a forward pass through the Sparse layer.

        Args:
            input_data (numpy.ndarray): The input data to the layer.

        Returns:
            numpy.ndarray: The output data produced by the layer.
        """
        # NOTE: input_data is presumed to be dense.
        self.input = input_data
        # Sparse dot product
        self.output = self.weights.dot(self.input) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Perform a backward pass for gradient computation.

        Args:
            output_gradient (numpy.ndarray): The gradient of the loss with respect to the layer's output.
            learning_rate (float): The learning rate used for gradient descent.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the layer's input.
        """
        # NOTE: output_gradient is presumed to be dense.
        # Gradient computation with sparse matrices
        input_gradient = self.weights.T.dot(output_gradient)
        # Gradient update; convert to dense for updating, then back to sparse
        weight_gradient = self.input.T.dot(csr_matrix(output_gradient))

        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, dtype=self.dtype)

        return input_gradient

    def get_params(self) -> Dict[str, Union[TArray, np.floating]]:
        """
        Get the current weights and biases of the layer.
        """
        return {
            "weights": self.weights,
            "biases": self.biases,
            "dtype": self.dtype,
        }

    def set_params(
        self, weights: TArray, biases: TArray, dtype: np.floating = np.float32
    ) -> None:
        """
        Set the weights and biases of the layer.
        """
        self.weights = weights
        self.biases = biases
        self.dtype = dtype
