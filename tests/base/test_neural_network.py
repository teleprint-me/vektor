"""
tests/base/test_neural_network.py
"""
import numpy as np
import pytest

from tests.base.test_activation import activation_layer
from tests.base.test_layer import stub_layer
from tests.base.test_loss_function import mock_loss
from vektor.base.neural_network import NeuralNetwork


class MockDenseLayer:
    """
    New code should use the `standard_normal` method of a `Generator` instance instead.

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

    def __init__(self, input_size, output_size):
        rng = np.random.default_rng()  # Random number generator instance
        self.dtype = np.float32  # Override default of 64-bit
        self.weights = rng.standard_normal(
            size=(output_size, input_size), dtype=self.dtype
        )
        self.biases = rng.standard_normal(size=output_size, dtype=self.dtype)

    def forward(self, input_data):
        return np.dot(self.weights, input_data) + self.biases

    def backward(self, output_gradient):
        # This is a simplified example.
        # In a real scenario, we would also update weights
        # and compute the gradient with respect to the input.
        return np.dot(self.weights.T, output_gradient)

    def get_params(self):
        return {
            "weights": self.weights,
            "biases": self.biases,
            "dtype": self.dtype,
        }

    def set_params(self, weights, biases, dtype):
        self.weights = weights
        self.biases = biases
        self.dtype = dtype


@pytest.fixture
def true_values():
    return np.array([1, 2, 3])


@pytest.fixture
def predicted_values():
    return np.array([1.1, 1.9, 3.1])


@pytest.fixture
def mock_dense_layer():
    input_size = 3  # example size, adjust as needed
    output_size = 2  # example size, adjust as needed
    return MockDenseLayer(input_size, output_size)


@pytest.fixture
def neural_network(stub_layer, activation_layer):
    return NeuralNetwork(layers=[stub_layer, activation_layer])


def test_network_initialization_with_no_layers():
    # Test initializing an empty network
    network = NeuralNetwork()
    assert len(network.layers) == 0


def test_network_initialization_with_layers(neural_network):
    assert len(neural_network.layers) == 2


def test_adding_layers_to_network(stub_layer, activation_layer):
    neural_network = NeuralNetwork()
    neural_network.add_layer(stub_layer)
    neural_network.add_layer(activation_layer)
    assert len(neural_network.layers) == 2


def test_forward_propagation(neural_network, true_values):
    output = neural_network.forward(true_values)
    np.testing.assert_array_equal(output, true_values)


def test_backward_propagation(
    neural_network,
    mock_loss,
    true_values,
    predicted_values,
):
    # Perform a forward pass
    neural_network.forward(true_values)
    # Calculate the gradient of the loss
    loss_gradient = mock_loss.prime(true_values, predicted_values)
    # Perform a backward pass
    backward_output = neural_network.backward(loss_gradient)
    # Assert the backward pass is equal to the gradient of the loss
    np.testing.assert_array_equal(backward_output, loss_gradient)


def test_network_parameter_retrieval(neural_network, mock_dense_layer):
    neural_network.add_layer(mock_dense_layer)
    params = neural_network.get_params()

    for param in params:
        if param is not None:  # Only check layers that have parameters
            assert "weights" in param and "biases" in param


def test_network_parameter_setting(neural_network, mock_dense_layer):
    rng = np.random.default_rng()
    new_weights = rng.standard_normal(
        size=mock_dense_layer.weights.shape, dtype=mock_dense_layer.dtype
    )
    new_biases = rng.standard_normal(
        size=mock_dense_layer.biases.shape, dtype=mock_dense_layer.dtype
    )
    new_dense_layer = MockDenseLayer(3, 2)
    new_dense_layer.set_params(new_weights, new_biases, neural_network.dtype)
    parameters = neural_network.get_params()
    parameters.append(new_dense_layer)
    neural_network.set_params(parameters)

    updated_params = neural_network.get_params()
    np.testing.assert_array_equal(updated_params["weights"], new_weights)
    np.testing.assert_array_equal(updated_params["biases"], new_biases)
