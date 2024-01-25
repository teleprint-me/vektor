"""
vektor/errors/mse.py

This module defines the Mean Squared Error (MSE) loss function.
"""
import numpy as np

from vektor.base.loss_function import LossFunction


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error (MSE) loss function.

    This class represents the Mean Squared Error (MSE) loss function used in neural networks.
    """

    def __call__(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error (MSE) loss.
        """
        return np.mean(np.square(y_true - y_pred))

    def prime(self, y_true, y_pred):
        """
        Calculate the derivative of the Mean Squared Error (MSE) loss.
        """
        return 2 * (y_pred - y_true) / y_true.size

    def regularized(self, y_true, y_pred, weights, lambda_=1e-5):
        """
        Calculate the regularized Mean Squared Error (MSE) loss with L2 regularization.
        """
        mse_loss = np.mean(np.square(y_true - y_pred))
        l2_penalty = lambda_ * np.sum(np.square(weights))
        return mse_loss + l2_penalty
