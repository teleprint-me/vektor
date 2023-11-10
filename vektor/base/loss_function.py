"""
vektor/base/loss_function.py

This module defines the base class for all loss functions used in neural networks.
"""
from abc import ABC, abstractmethod
from typing import Union

from numpy import ndarray
from scipy.sparse import sparray


class LossFunction(ABC):
    """
    Base class for all loss functions.

    This class serves as the base for all loss functions used in neural networks.

    Attributes:
        None

    Methods:
        __call__(self, y_true, y_pred): Calculate the loss.
        prime(self, y_true, y_pred): Calculate the derivative of the loss.
        regularize(self, y_true, y_pred, weights, lambda_): Calculate the regularized loss.
    """

    @abstractmethod
    def __call__(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        """
        Calculate the loss.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            numpy.ndarray: The calculated loss.
        """
        ...

    @abstractmethod
    def prime(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        """
        Calculate the derivative of the loss.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            numpy.ndarray: The derivative of the loss.
        """
        ...

    @abstractmethod
    def regularize(
        self,
        y_true: ndarray,
        y_pred: ndarray,
        weights: Union[ndarray, sparray],
        lambda_: float,
    ) -> ndarray:
        """
        Calculate the regularized loss.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted values.
            weights (Union[numpy.ndarray, sparray]): The model's weights.
            lambda_ (float): The regularization parameter.

        Returns:
            numpy.ndarray: The regularized loss.

        NOTE: Multiplication of sparse arrays using the * operator now performs element-wise multiplication, just like NumPy arrays, instead of matrix multiplication. For matrix multiplication, the @ operator should be used.
        """
        ...
