"""
tmnn/activations/relu.py

This module defines the ReLU activation layer class for neural networks.
"""
from numpy import maximum, ndarray

from vektor.base.activation import Activation


# NOTE: Forward and backward methods are inherited from Activation class
class ReLU(Activation):
    """
    ReLU (Rectified Linear Unit) activation layer for neural networks.

    This class provides the ReLU activation function and its derivative.

    Attributes:
        None

    Methods:
        __init__(): Constructor method.
        relu(x): Compute the ReLU activation function.
        relu_prime(x): Compute the derivative of the ReLU activation function.
    """

    def __init__(self):
        """
        Initializes the ReLU activation layer.
        """
        super().__init__(self.relu, self.relu_prime)

    @staticmethod
    def relu(x: ndarray) -> ndarray:
        """
        Compute the ReLU activation function.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The result of applying the ReLU activation function to the input.
        """
        return maximum(0, x)

    @staticmethod
    def relu_prime(x: ndarray) -> ndarray:
        """
        Compute the derivative of the ReLU activation function.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The derivative of the ReLU activation function applied to the input.
        """
        return (x > 0).astype(float)
