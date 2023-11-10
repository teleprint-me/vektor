"""
vektor/base/neural_network.py

This module defines the base class for all neural network layers.
"""
from typing import List, Tuple, Type, Union

import numpy as np
from scipy.sparse import sparray

from vektor.base.layer import Layer


class NeuralNetwork:
    """
    Base class for a modular neural network, with support for various layer types.

    Attributes:
        layers (List[Layer]): A list of layers that make up the network.
        dtype (np.dtype): The data type for network parameters, default to float16.
    """

    def __init__(self, layers: List[Type[Layer]], dtype: np.dtype = np.float16):
        self.layers = [
            layer() for layer in layers
        ]  # Instantiate layers from the provided layer classes
        self.dtype = dtype

    def forward(
        self, input_data: Union[np.ndarray, sparray]
    ) -> Union[np.ndarray, sparray]:
        """
        Perform a forward pass through each layer in the network.
        """
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(
        self, gradient: Union[np.ndarray, sparray]
    ) -> Union[np.ndarray, sparray]:
        """
        Perform a backward pass through each layer in the network, in reverse order.
        """
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def add_layer(self, layer: Type[Layer]) -> None:
        """
        Add a new layer to the network.
        """
        self.layers.append(layer())

    def get_params(self) -> List[Tuple[Union[np.ndarray, sparray], np.ndarray]]:
        """
        Get parameters from all layers in the network.
        """
        return [
            layer.get_params() for layer in self.layers if hasattr(layer, "get_params")
        ]

    def set_params(
        self, params: List[Tuple[Union[np.ndarray, sparray], np.ndarray]]
    ) -> None:
        """
        Set parameters for all layers in the network.
        """
        for layer, param in zip(self.layers, params):
            if hasattr(layer, "set_params"):
                layer.set_params(*param)
