"""
tests/base/test_activation.py

Remember to adjust actual activation functions for more complex ones and to test various edge cases as needed.
"""
import sys

import numpy as np
import pytest
from scipy.sparse import csr_array, sparray

from vektor.base.activation import Activation


# Define a linear activation function and its derivative for testing purposes
def linear_activation(x: np.ndarray) -> np.ndarray:
    return x


def linear_activation_prime(x: np.ndarray) -> np.ndarray:
    # NOTE: Explicitly set `dtype` to `float` to guarantee sparse arrays. Otherwise, it returns None instead.
    return np.ones_like(x, dtype=float)


@pytest.fixture
def activation_layer():
    return Activation(
        activation=linear_activation, activation_prime=linear_activation_prime
    )


@pytest.fixture(params=[np.array, csr_array])
def input_data(request):
    data = np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]], dtype=float)
    return request.param(data)


@pytest.fixture(params=[np.array, csr_array])
def output_gradient(request):
    data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
    return request.param(data)


def test_activation_forward(activation_layer, input_data):
    output = activation_layer.forward(input_data)
    # Convert sparse matrices to dense for comparison
    if isinstance(output, sparray):
        output = output.toarray()
        input_data = input_data.toarray()
    np.testing.assert_array_equal(
        output,
        input_data,
        "Forward pass should apply the linear activation function",
    )


def test_activation_backward(activation_layer, input_data, output_gradient):
    # Perform a forward pass to set the input state
    activation_layer.forward(input_data)
    # Calculate the backward pass which should apply the derivative of the activation function
    backward_output = activation_layer.backward(output_gradient)
    # Convert sparse matrices to dense for comparison
    if isinstance(backward_output, sparray):
        backward_output = backward_output.toarray()
        output_gradient = output_gradient.toarray()
    # For a linear activation, the backward output should be the same as the gradient
    np.testing.assert_array_equal(
        backward_output,
        output_gradient,
        "Backward pass should apply the derivative of the linear activation function",
    )


def test_activation_backward_without_forward(activation_layer, output_gradient):
    # Attempting a backward pass without a forward pass should raise an error
    with pytest.raises(ValueError):
        activation_layer.backward(output_gradient)
