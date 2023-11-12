"""
vektor/base/layer.py

This module defines the base class for all layers used in neural networks.

NOTE:

SciPy is switching to an array interface, compatible with NumPy arrays, from the older matrix interface. SciPy recommends using array objects (csr_array, bsr_array, coo_array, etc.) for all new work.

sparray is the new base class for SciPy array-like objects which are compatible with ndarray's.

Source: https://docs.scipy.org/doc/scipy-1.11.3/reference/sparse.html
"""
from typing import Any, Mapping, Protocol, TypeVar

from numpy import ndarray
from scipy.sparse import sparray

# Define a type variable that can be ndarray or any subtype of sparray (like csr_array, coo_array, etc.)
TArray = TypeVar("TArray", ndarray, sparray)


class Layer(Protocol[TArray]):
    """
    Protocol defining the mandatory methods that any neural network layer must implement.
    """

    @property
    def architecture(self) -> Mapping[str, Any]:
        """
        Returns the architectural details of the layer in a predefined format.
        """
        ...

    def forward(self, input_data: TArray) -> TArray:
        """
        Perform a forward pass through the layer.

        Args:
            input_data (TArray): The input data to the layer.

        Returns:
            TArray: The output data from the layer.
        """
        ...

    def backward(self, output_gradient: TArray, **kwargs: Any) -> TArray:
        """
        Perform a backward pass to compute the gradient.

        Args:
            output_gradient (TArray): The gradient of the loss with respect to the output.
            kwargs (Mapping[str, Any]): Additional keyword arguments for more specialized layers.

        Returns:
            TArray: The gradient of the loss with respect to the input.
        """
        ...

    def get_params(self) -> Mapping[str, Any]:
        """Return the parameters of the layer."""
        ...

    def set_params(self, **params: Any) -> None:
        """Set the parameters of the layer."""
        ...
