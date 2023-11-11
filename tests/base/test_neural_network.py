"""
tests/base/test_neural_network.py
"""
import numpy as np
import pytest

from tests.base.test_activation import activation_layer
from tests.base.test_layer import stub_layer
from tests.base.test_loss_function import mock_loss
from vektor.base.neural_network import NeuralNetwork


@pytest.fixture
def true_values():
    return np.array([1, 2, 3])


@pytest.fixture
def predicted_values():
    return np.array([1.1, 1.9, 3.1])


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


def test_network_parameter_retrieval():
    # Test retrieving parameters from the network
    ...


def test_network_parameter_setting():
    # Test setting parameters for the network
    ...
