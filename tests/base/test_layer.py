"""
tests/base/test_layer.py
"""
import numpy as np
import pytest
from scipy.sparse import csr_array

from vektor.base.layer import Layer, TArray


class StubLayer(Layer[TArray]):
    def forward(self, input_data: TArray) -> TArray:
        # A simple pass-through for testing purposes
        return input_data

    def backward(self, output_gradient: TArray, **kwargs) -> TArray:
        # For testing, we'll simply return the output_gradient
        return output_gradient


@pytest.fixture
def stub_layer():
    return StubLayer()


@pytest.fixture(params=[np.array, csr_array])
def layer_input(request):
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    return request.param(data)


@pytest.fixture(params=[np.array, csr_array])
def layer_gradient(request):
    data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
    return request.param(data)


def test_layer_forward(stub_layer, layer_input):
    output = stub_layer.forward(layer_input)
    if isinstance(output, np.ndarray):
        np.testing.assert_array_equal(output, layer_input)
    else:  # it's a sparse matrix
        np.testing.assert_array_equal(output.toarray(), layer_input.toarray())


def test_layer_backward(stub_layer, layer_input, layer_gradient):
    output = stub_layer.backward(layer_gradient)
    if isinstance(output, np.ndarray):
        np.testing.assert_array_equal(output, layer_gradient)
    else:  # it's a sparse matrix
        np.testing.assert_array_equal(output.toarray(), layer_gradient.toarray())
