"""
vektor/base/activation.py

This module defines the base class for all activation layers used in neural networks.
"""
from typing import Any, Callable, Mapping, Optional

from numpy import multiply, ndarray
from scipy.sparse import sparray

from vektor.base.layer import Layer, TArray


class Activation(Layer[TArray]):
    """
    Base class for activation layers in neural networks.
    """

    def __init__(
        self,
        activation: Callable[[TArray], TArray],
        activation_prime: Callable[[TArray], TArray],
    ):
        """
        Initializes the Activation layer.

        Args:
            activation (Callable[[TArray], TArray]): The activation function.
            activation_prime (Callable[[TArray], TArray]): The derivative of the activation function.
        """
        self.activation = activation
        self.activation_prime = activation_prime
        self.input: Optional[TArray] = None
        self.output: Optional[TArray] = None

    @property
    def architecture(self) -> Mapping[str, Any]:
        """
        Returns the architectural details of the layer in a predefined format.
        """
        raise NotImplementedError(
            f"Activation architecture is missing for {self.__class__.__name__}."
        )

    def forward(self, input_data: TArray) -> TArray:
        """
        Perform a forward pass through the activation layer.

        Args:
            input_data (TArray): The input data to the layer.

        Returns:
            TArray: The output data from the layer, after applying the activation function.
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient: TArray, **kwargs: Mapping[str, Any]) -> TArray:
        """
        Perform a backward pass to compute the gradient through the activation layer.

        Args:
            output_gradient (TArray): The gradient of the loss with respect to the output.
            kwargs (Mapping[str, Any]): Additional keyword arguments for more specialized layers.

        Returns:
            TArray: The gradient of the loss with respect to the input, after applying the derivative of the activation function.
        """
        if self.input is None:
            raise ValueError("forward must be called before backward.")

        # Ensure the multiplication is compatible with both dense and sparse arrays
        if isinstance(output_gradient, sparray):
            # If output_gradient is sparse, use the multiplication that's appropriate for sparse matrices
            return output_gradient.multiply(self.activation_prime(self.input))
        elif isinstance(output_gradient, ndarray):
            # Use numpy's multiply for dense arrays
            return multiply(output_gradient, self.activation_prime(self.input))
        else:
            raise ValueError(
                f"Unknown array type. Expected ndarray or sparray. Got {type(output_gradient)} instead."
            )
