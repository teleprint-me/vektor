"""
tests/base/test_layer.py

NOTE: New code should use the `standard_normal` method of a `Generator` instance instead.

# Old way: using np.random.randn
samples = np.random.randn(output_size, input_size)

# New way: using rng.standard_normal
rng = np.random.default_rng()
samples = rng.standard_normal(size=(output_size, input_size), dtype=np.float64)

# Specifying Mean and Standard Deviation
mu = ...  # your mean value
sigma = ...  # your standard deviation value
samples = mu + sigma * rng.standard_normal(size=...)

Source:
- https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html#numpy.random.randn
- https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html#numpy.random.Generator.standard_normal
"""
import numpy as np
import pytest
from scipy.sparse import csr_array, sparray

from vektor.base.layer import Layer, TArray


class MockLayer(Layer[TArray]):
    def __init__(self, input_dim: int, output_dim: int, dtype=float):
        self.rng = np.random.default_rng()
        self.weights = self.rng.standard_normal(
            size=(output_dim, input_dim), dtype=dtype
        )
        self.biases = self.rng.standard_normal(size=(output_dim,), dtype=dtype)
        self.dtype = dtype

    def forward(self, input_data: TArray) -> TArray:
        return np.dot(input_data, self.weights.T) + self.biases

    def backward(self, output_gradient: TArray, **kwargs) -> TArray:
        return np.dot(output_gradient, self.weights)

    @property
    def specification(self) -> dict:
        return {
            "type": "MockDense",
            "class": MockDense,  # Self-reference to the class
            "input_dim": self.weights.shape[1],
            "output_dim": self.weights.shape[0],
            "dtype": self.dtype,
        }

    def get_params(self) -> dict:
        return {
            "weights": self.weights,
            "biases": self.biases,
            "dtype": self.dtype,
        }

    def set_params(self, weights: TArray, biases: TArray, dtype=np.float32) -> None:
        self.weights = weights
        self.biases = biases
        self.dtype = dtype


@pytest.fixture
def mock_layer():
    return MockLayer(input_dim=3, output_dim=2)


@pytest.fixture(params=[np.array, csr_array])
def layer_input(request):
    # Generates input data of shape (2, 3) for testing the forward method
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    return request.param(data)


@pytest.fixture(params=[np.array, csr_array])
def layer_gradient(request):
    # Generates gradient data of shape (2, 2) for testing the backward method
    # Ensure the shape is compatible with the MockLayer's output
    data = np.array([[0.1, 0.2], [0.4, 0.5]], dtype=float)
    return request.param(data)


def test_layer_forward(mock_layer, layer_input):
    # Convert to dense if input is sparse
    dense_input = (
        layer_input.toarray() if isinstance(layer_input, sparray) else layer_input
    )

    # Calculate expected output
    expected_output = np.dot(dense_input, mock_layer.weights.T) + mock_layer.biases

    # Actual output from the mock_layer
    output = mock_layer.forward(dense_input)

    # Check if the actual output matches the expected output
    np.testing.assert_array_almost_equal(output, expected_output)


def test_layer_backward(mock_layer, layer_input, layer_gradient):
    # Prepare inputs
    dense_input = (
        layer_input.toarray() if isinstance(layer_input, sparray) else layer_input
    )
    dense_gradient = (
        layer_gradient.toarray()
        if isinstance(layer_gradient, sparray)
        else layer_gradient
    )

    # Ensuring forward pass is done before backward pass
    mock_layer.forward(dense_input)

    # Expected output for the backward method
    expected_output = np.dot(dense_gradient, mock_layer.weights)

    # Actual output from the mock_layer
    output = mock_layer.backward(dense_gradient)

    # Check if the actual output matches the expected output
    np.testing.assert_array_almost_equal(output, expected_output)
