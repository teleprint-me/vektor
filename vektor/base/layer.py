"""
vektor/base/layer.py

This module defines the base class for all layers used in neural networks.
"""
from typing import Any, Mapping, Protocol

from numpy import ndarray


class Layer(Protocol):
    """
    Protocol defining the mandatory methods that any neural network layer must implement.
    """

    def forward(self, input_data: ndarray) -> ndarray:
        """
        Perform a forward pass through the layer.

        Args:
            input_data (numpy.ndarray): The input data to the layer.

        Returns:
            numpy.ndarray: The output data from the layer.
        """
        ...

    def backward(
        self, output_gradient: ndarray, **kwargs: Mapping[str, Any]
    ) -> ndarray:
        """
        Perform a backward pass to compute the gradient.

        Args:
            output_gradient (numpy.ndarray): The gradient of the loss with respect to the output.
            kwargs (Mapping[str, Any]): Additional keyword arguments for more specialized layers.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input.
        """
        ...
