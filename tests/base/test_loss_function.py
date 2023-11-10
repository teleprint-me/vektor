"""
tests/base/test_loss_function.py
"""
from typing import Union

import numpy as np
import pytest
from scipy.sparse import csr_array

from vektor.base.loss_function import LossFunction


class MockLoss(LossFunction):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # A mock loss calculation (mean squared error for simplicity)
        return np.mean((y_true - y_pred) ** 2)

    def prime(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # The derivative of the mock loss (gradient of mean squared error)
        return 2 * (y_pred - y_true) / y_true.size

    def regularize(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: Union[np.ndarray, csr_array],
        lambda_: float,
    ) -> np.ndarray:
        # A mock regularization (L2 regularization for simplicity)
        if isinstance(weights, csr_array):
            weights = weights.toarray()
        return lambda_ * np.sum(weights**2)


@pytest.fixture
def mock_loss():
    return MockLoss()


@pytest.fixture
def true_values():
    return np.array([1, 2, 3])


@pytest.fixture
def predicted_values():
    return np.array([1.1, 1.9, 3.1])


@pytest.fixture
def weights():
    # Return csr_array instead of numpy array
    return csr_array([0.1, 0.2, 0.3])


def test_loss_call(mock_loss, true_values, predicted_values):
    # Test the loss calculation
    loss = mock_loss(true_values, predicted_values)
    expected_loss = np.mean((true_values - predicted_values) ** 2)
    np.testing.assert_almost_equal(loss, expected_loss)


def test_loss_prime(mock_loss, true_values, predicted_values):
    # Test the loss derivative calculation
    loss_prime = mock_loss.prime(true_values, predicted_values)
    expected_prime = 2 * (predicted_values - true_values) / true_values.size
    np.testing.assert_array_almost_equal(loss_prime, expected_prime)


def test_loss_regularize(mock_loss, true_values, predicted_values, weights):
    # Test the regularization calculation
    lambda_ = 0.1
    regularized_loss = mock_loss.regularize(
        true_values, predicted_values, weights, lambda_
    )
    expected_regularization = lambda_ * np.sum(weights.toarray() ** 2)
    np.testing.assert_almost_equal(regularized_loss, expected_regularization)
