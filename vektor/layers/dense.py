"""
vektor/layers/dense.py

This module defines the Dense layer class for neural networks.
"""
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from vektor.base.layer import Layer, TArray


class Dense(Layer[TArray]):
    """
    Dense layer for neural networks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dtype: np.floating = np.float32,
        activation_fn: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initializes the Dense layer.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        self.dtype = dtype
        self.seed = seed
        self.weights = None
        self.biases = None

        if seed is not None:
            np.random.seed(seed)

        self._initialize_weights_and_biases()

    def _get_normal_distribution(self) -> TArray:
        """
        Initialize a random normal distribution.
        """
        rng = np.random.default_rng()  # Random number generator instance
        size = (self.input_dim, self.output_dim)
        distribution = rng.standard_normal(size=size, dtype=self.dtype)
        return distribution

    def _get_normal_initialization(self) -> float:
        """
        Initialize a scaling factor for a random normal distribution.
        """
        # NOTE: DO NOT REMOVE NOTES!
        if self.activation_fn == "relu":
            # NOTE: He Initialization: Var(W) = mu + (2 / n)
            # The weights are sampled from a Gaussian distribution
            # with mean 0 and variance 2/n, where n is the number of input
            # neurons feeding into the layer.
            scaling_factor = np.sqrt(2.0 / self.input_dim)
        elif self.activation_fn in ["tanh", "sigmoid"]:
            # NOTE: Glorot and Bengio initialization:
            #   Var(W) = mu + (2 / (n_in + n_out))
            # This initialization maintains the variance of activations
            # across layers. 6 or 2 may be used as the coefficient of the
            # expression. Averaging the dimensions and then dividing the
            # coefficient by the average is a experimental possibility as well.
            scaling_factor = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        else:  # NOTE: Fallback to random initialization
            scaling_factor = 1.0
        return scaling_factor

    def _initialize_weights_and_biases(self) -> None:
        """
        Initialize layer weights based on the activation function.
        """
        initialization = self._get_normal_initialization()
        distribution = self._get_normal_distribution()
        self.weights = initialization * distribution
        self.biases = np.zeros((1, self.output_dim), dtype=self.dtype)

    @property
    def specification(self) -> Mapping[str, Any]:
        """
        Returns the architectural details of the layer in a predefined format.
        """
        return {
            "type": "dense",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "dtype": self.dtype,
            "activation_fn": self.activation_fn,
            "seed": self.seed,
        }

    @property
    def dimensions(self) -> Tuple[int, int]:
        """
        Returns the shape of the layer.
        """
        return self.input_dim, self.output_dim

    @property
    def state(self) -> Tuple[TArray, TArray]:
        """
        Returns the state of the layer.
        """
        return self.input, self.output

    def forward(self, input_data: TArray) -> TArray:
        """
        Perform a forward pass through the layer.
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(
        self, output_gradient: TArray, learning_rate: float, lambda_: float
    ) -> TArray:
        """
        Perform a backward pass through the layer for gradient computation.
        """
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)

        # Update parameters with regularization
        self.weights -= learning_rate * (weights_gradient + 2 * lambda_ * self.weights)
        self.biases -= learning_rate * np.sum(
            output_gradient,
            axis=0,
            keepdims=True,
            dtype=self.dtype,
        )

        return input_gradient

    def get_params(self) -> Dict[str, Union[TArray, np.floating]]:
        """
        Get the current weights and biases of the layer.
        """
        return {
            "weights": self.weights,
            "biases": self.biases,
            "dtype": self.dtype,
        }

    def set_params(
        self, weights: TArray, biases: TArray, dtype: np.floating = np.float32
    ) -> None:
        """
        Set the weights and biases of the layer.
        """
        self.weights = weights
        self.biases = biases
        self.dtype = dtype
