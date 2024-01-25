"""
vektor/loss/__init__.py
"""
from vektor.errors.ce import CrossEntropy
from vektor.errors.mse import MeanSquaredError
from vektor.errors.rmse import RootMeanSquaredError

__all__ = [CrossEntropy, MeanSquaredError, RootMeanSquaredError]
