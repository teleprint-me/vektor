"""
tests/base/test_layer.py
"""
import numpy as np
import pytest

from vektor.base.layer import Layer


class StubLayer(Layer):
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        # A simple pass-through for testing purposes
        return input_data

    def backward(self, output_gradient: np.ndarray, **kwargs) -> np.ndarray:
        # For testing, we'll simply return the output_gradient
        return output_gradient


@pytest.fixture
def layer_input():
    return np.array([[1, 2, 3], [4, 5, 6]], dtype=float)


@pytest.fixture
def layer_gradient():
    return np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)


def test_layer_forward(layer_input):
    layer = StubLayer()
    output = layer.forward(layer_input)
    np.testing.assert_array_equal(output, layer_input)


def test_layer_backward(layer_input, layer_gradient):
    layer = StubLayer()
    output = layer.backward(layer_gradient)
    np.testing.assert_array_equal(output, layer_gradient)
