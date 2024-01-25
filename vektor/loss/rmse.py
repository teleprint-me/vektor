"""
vektor/errors/rmse.py

This module defines the Root Mean Squared Error (RMSE) loss function.
"""
import numpy as np

from vektor.base.loss_function import LossFunction


class RootMeanSquaredError(LossFunction):
    """
    Root Mean Squared Error (RMSE) loss function.

    This class represents the RMSE loss function, which calculates the root mean squared
    error between the true values (y_true) and predicted values (y_pred).
    """

    def __call__(self, y_true, y_pred):
        """
        Calculate the root mean squared error (RMSE) between y_true and y_pred.
        """
        return np.sqrt(np.mean(np.square(y_true - y_pred)))

    def prime(self, y_true, y_pred):
        """
        Calculate the derivative of the RMSE loss.
        """
        return (y_pred - y_true) / (
            y_true.size * np.sqrt(np.mean(np.square(y_true - y_pred)))
        )

    def regularized(self, y_true, y_pred, weights, lambda_=1e-5):
        """
        Calculate the regularized RMSE loss with L2 regularization.
        """
        rms_loss = np.sqrt(np.mean(np.square(y_true - y_pred)))
        l2_penalty = lambda_ * np.sum(np.square(weights))
        return rms_loss + l2_penalty
        return rms_loss + l2_penalty
