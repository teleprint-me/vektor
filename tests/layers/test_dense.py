"""
tests/layers/test_dense.py

Test suite for the Dense layer implementation.
"""
import numpy as np
import pytest

from vektor.layers.dense import Dense


def test_initialization():
    input_dim = 5
    output_dim = 3
    layer = Dense(input_dim, output_dim)

    assert layer.weights.shape == (
        input_dim,
        output_dim,
    ), "Incorrect weight matrix shape"
    assert layer.biases.shape == (1, output_dim), "Incorrect bias vector shape"
    assert np.all(layer.biases == 0), "Biases should be initialized to zero"


def test_forward_propagation():
    layer = Dense(5, 3)
    sample_input = np.random.rand(1, 5)
    output = layer.forward(sample_input)

    assert output.shape == (1, 3), "Incorrect output shape from forward propagation"


def test_backward_propagation():
    layer = Dense(5, 3)
    sample_input = np.random.rand(1, 5)

    # Performing a forward pass to set the input attribute in the layer
    output = layer.forward(sample_input)

    sample_output_grad = np.random.rand(1, 3)

    # Performing the backward pass
    input_grad = layer.backward(sample_output_grad, learning_rate=0.01, lambda_=0.01)

    # Asserting that the shape of the input gradient matches the shape of the sample input
    assert input_grad.shape == sample_input.shape, "Incorrect input gradient shape"


def test_parameter_updating():
    layer = Dense(5, 3)
    new_weights = np.random.rand(5, 3)
    new_biases = np.random.rand(1, 3)

    layer.set_params(new_weights, new_biases)
    params = layer.get_params()

    assert np.array_equal(
        params["weights"], new_weights
    ), "Weights were not updated correctly"
    assert np.array_equal(
        params["biases"], new_biases
    ), "Biases were not updated correctly"


@pytest.mark.parametrize("activation_fn", ["relu", "tanh", "sigmoid", None])
def test_activation_function_support(activation_fn):
    layer = Dense(5, 3, activation_fn=activation_fn)
    sample_input = np.random.rand(1, 5)
    output = layer.forward(sample_input)

    assert (
        output is not None
    ), f"Forward propagation failed with activation function: {activation_fn}"
