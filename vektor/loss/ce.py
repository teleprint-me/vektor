"""
vektor/errors/ce.py

This module defines the Cross-Entropy (CE) loss function.
"""
import numpy as np

from vektor.base.layer import TArray
from vektor.base.loss_function import LossFunction


class CrossEntropy(LossFunction):
    """
    Cross-Entropy loss function for binary and multi-class classification.

    This class represents the Cross-Entropy loss function, which is commonly used
    for both binary and multi-class classification problems.
    """

    def __call__(self, y_true: TArray, y_pred: TArray) -> float:
        """
        Calculate the Cross-Entropy loss between y_true and y_pred.
        """
        epsilon = 1e-15  # To prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def prime(self, y_true: TArray, y_pred: TArray) -> TArray:
        """
        Calculate the derivative of the Cross-Entropy loss.
        """
        epsilon = 1e-15  # To prevent division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

    def regularized(
        self,
        y_true: TArray,
        y_pred: TArray,
        weights: TArray,
        lambda_: float = 1e-5,
    ) -> float:
        """
        Calculate the regularized Cross-Entropy loss with L2 regularization.
        """
        cross_entropy_loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        l2_penalty = lambda_ * np.sum(np.square(weights))
        return cross_entropy_loss + l2_penalty
