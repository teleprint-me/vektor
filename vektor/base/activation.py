"""
vektor/base/activation.py

This module defines the base class for all activation layers used in neural networks.
"""
from typing import Any, Callable, Mapping, Optional

from numpy import multiply, ndarray

from vektor.base.layer import Layer


class Activation(Layer):
    """
    Base class for activation layers in neural networks.
    """

    def __init__(
        self,
        activation: Callable[[ndarray], ndarray],
        activation_prime: Callable[[ndarray], ndarray],
    ):
        """
        Initializes the Activation layer.

        Args:
            activation (Callable[[ndarray], ndarray]): The activation function.
            activation_prime (Callable[[ndarray], ndarray]): The derivative of the activation function.
        """
        self.activation = activation
        self.activation_prime = activation_prime

        self.input: Optional[ndarray] = None
        self.output: Optional[ndarray] = None

    def forward(self, input_data: ndarray) -> ndarray:
        """
        Perform a forward pass through the activation layer.

        Args:
            input_data (numpy.ndarray): The input data to the layer.

        Returns:
            numpy.ndarray: The output data from the layer, after applying the activation function.
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(
        self, output_gradient: ndarray, **kwargs: Mapping[str, Any]
    ) -> ndarray:
        """
        Perform a backward pass to compute the gradient through the activation layer.

        Args:
            output_gradient (numpy.ndarray): The gradient of the loss with respect to the output.
            kwargs (Mapping[str, Any]): Additional keyword arguments for more specialized layers.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input, after applying the derivative of the activation function.
        """
        if self.input is None:
            raise ValueError("forward must be called before backward.")

        return multiply(output_gradient, self.activation_prime(self.input))
