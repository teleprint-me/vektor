"""
Module: vektor.core.vector.py
"""
from vektor.core.dtype import TScalar, TVector


class Vector:
    def __init__(self, *coordinates: TScalar) -> None:
        self.coordinates = coordinates

    def __repr__(self) -> str:
        return f"Vector{self.coordinates}"

    def __add__(self, other: "Vector") -> "Vector":
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Vectors must have the same dimension for addition")
        return Vector(*(a + b for a, b in zip(self.coordinates, other.coordinates)))

    def __sub__(self, other: "Vector") -> "Vector":
        if len(self.coordinates) != len(other.coordinates):
            raise ValueError("Vectors must have the same dimension for subtraction")
        return Vector(*(a - b for a, b in zip(self.coordinates, other.coordinates)))

    # Additional dunder methods like __mul__ for dot product, scalar multiplication, etc.

    @property
    def length(self) -> float:
        return sum(x**2 for x in self.coordinates) ** 0.5

    # Other vector-specific methods...
