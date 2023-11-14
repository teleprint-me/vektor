"""
tests/core/test_dtype.py
"""
import math

import pytest

from vektor.core.dtype import float32


@pytest.fixture
def float32_value() -> int:
    """
    float32 is a stateless, callable, class with a static method enabling converting back to float from int.
    NOTE: This value has more precision than Float32 can handle
    """
    return float32(123.456789)


def test_conversion_of_standard_float_to_float32(float32_value):
    """
    Test conversion of a standard float to Float32.
    """
    value = 123.456  # NOTE: This is specific to 32-bit
    # Ensure the static method is used for conversion
    assert float32.to_float(float32_value) == pytest.approx(value)


def test_float32_precision(float32_value):
    """
    Test the precision of the Float32 type.

    NOTE: Pay special attention to this test, as this is a challenging aspect to get right.
    """
    value = 123.456  # NOTE: This is specific to 32-bit
    # Using the static method for conversion and checking for approximation
    assert float32.to_float(float32_value) != value
    assert float32.to_float(float32_value) == pytest.approx(value, rel=1e-6)


def test_float32_special_values():
    """
    Test special values like zero, infinities, and NaN.
    """
    # Use static method to convert back to float and check values
    assert float32.to_float(float32(0)) == 0
    assert float32.to_float(float32(float("inf"))) == float("inf")
    assert float32.to_float(float32(float("-inf"))) == float("-inf")
    assert math.isnan(float32.to_float(float32(float("nan"))))


# Add more tests for other cases as needed
