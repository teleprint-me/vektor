"""
vektor/activations/softmax.py

This module defines the Softmax activation layer class for neural networks.
"""

from typing import Any, Mapping

import numpy as np

from vektor.base.activation import Activation, TArray
from vektor.base.exceptions import NotSupportedError


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

    Note:
        The `softmax_prime` method is included to maintain consistency with the Activation class
        pattern, but it is not used since Softmax does not have a simple analytical derivative.
        Users may handle the derivative separately when using Softmax.
    """

    def __init__(self):
        """
        Initializes the Softmax activation layer.
        """
        super().__init__(self.softmax, self.softmax_prime)

    @staticmethod
    def softmax(x: TArray) -> TArray:
        """
        Compute the Softmax activation function.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The result of applying the Softmax activation function to the input.
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def softmax_prime(x: TArray) -> TArray:
        """
        Placeholder for the derivative of the Softmax activation function.

        Note:
            The `softmax_prime` method is included to maintain consistency with the Activation class
            pattern, but it is not used since Softmax does not have a simple analytical derivative.
            Users may handle the derivative separately when using Softmax.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: Placeholder for the derivative (not used).
        """
        pass

    def backward(self, output_gradient: TArray, **kwargs: Mapping[str, Any]) -> TArray:
        raise NotSupportedError("The backward method is not supported for Softmax.")
