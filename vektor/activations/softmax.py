"""
vektor/activations/softmax.py

This module defines the Softmax activation layer class for neural networks.
"""
import numpy as np

from vektor.base.activation import Activation


# NOTE: Forward and backward methods are inherited from Activation class
class Softmax(Activation):
    """
    Softmax activation layer for neural networks.

    This class provides the Softmax activation function.

    Attributes:
        None

    Methods:
        __init__(): Constructor method.
        softmax(x): Compute the Softmax activation function.
    """

    def __init__(self):
        """
        Initializes the Softmax activation layer.
        """
        super().__init__(self.softmax)

    @staticmethod
    def softmax(x):
        """
        Compute the Softmax activation function.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The result of applying the Softmax activation function to the input.
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
