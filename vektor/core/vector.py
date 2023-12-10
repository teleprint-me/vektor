"""
Module: vektor.core.vector

The beating heart of the Vektor package.
"""

from typing import Tuple

from vektor.core.dtype import TScalar, TVector


class Vector:
    """
    Represents a mathematical vector.

    Args:
        *coordinates (TScalar): Variable number of scalar coordinates for the vector.

    Attributes:
        coordinates (Tuple[TScalar, ...]): Tuple of scalar coordinates.
    """

    def __init__(self, *coordinates: TScalar) -> None:
        self.coordinates = coordinates

    def __repr__(self) -> str:
        """
        String representation of the vector in R^n.
        """
        return f"Vector{self.coordinates}"

    def __add__(self, other: "Vector") -> "Vector":
        """
        Element-wise addition of two vectors.
        """
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Vectors must have the same dimension for addition")
        return Vector(*(a + b for a, b in zip(self.coordinates, other.coordinates)))

    def __sub__(self, other: "Vector") -> "Vector":
        """
        Element-wise subtraction of two vectors.
        """
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Vectors must have the same dimension for subtraction")
        return Vector(*(a - b for a, b in zip(self.coordinates, other.coordinates)))

    def __mul__(self, other: "Vector") -> "Vector":
        ...

    def __div__(self, other: "Vector") -> "Vector":
        ...

    @property
    def length(self) -> float:
        """
        Calculate the Euclidean norm (length) of the vector in R^n.

        Returns:
            float: The length of the vector.
        """
        return sum(x**2 for x in self.coordinates) ** 0.5

    @property
    def unit(self) -> Tuple[TScalar, ...]:
        """
        Find the unit vector in the direction of the vector in R^n.

        Returns:
            Tuple[TScalar, ...]: The unit vector.

        Raises:
            ValueError: If the vector is a zero vector (magnitude is 0).
        """
        magnitude = self.length  # Compute only once!
        if magnitude == 0:
            raise ValueError("Cannot compute the unit vector of a zero vector")
        return tuple(x / magnitude for x in self.coordinates)

    # Other vector-specific methods...
