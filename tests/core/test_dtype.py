"""
tests/core/test_dtype.py
"""
import math

import pytest

from vektor.core.dtype import float32


def integer_is_close(a: int, b: int, tolerance: int) -> bool:
    """
    Check if two integer values are close within a specified tolerance.

    Args:
        a (int): The first integer value.
        b (int): The second integer value.
        tolerance (int): The tolerance within which to consider values as close.

    Returns:
        bool: True if the absolute difference between 'a' and 'b' is less than or equal to 'tolerance', False otherwise.
    """
    return abs(a - b) <= tolerance


def float_is_close(a, b, relative=1e-03, absolute=0.0):
    """
    Check if two floating-point values are approximately equal within specified tolerances.

    Args:
        a (float): The first floating-point value.
        b (float): The second floating-point value.
        relative (float): The relative tolerance for comparing values (default: 1e-03).
        absolute (float): The absolute tolerance for comparing values (default: 0.0).

    Returns:
        bool: True if the absolute difference between 'a' and 'b' is within the tolerance bounds, False otherwise.
    """
    return abs(a - b) <= max(relative * max(abs(a), abs(b)), absolute)


@pytest.fixture
def float64_original_value() -> float:
    """
    Fixture for the source floating-point value used for testing 64-bit precision.
    """
    return 123.456789


@pytest.fixture
def float64_expected_value() -> float:
    """
    Fixture for the expected 64-bit floating-point value.
    """
    return 123.456


@pytest.fixture
def float32_expected_value() -> int:
    """
    Fixture for the expected 32-bit float value represented as an integer.
    """
    return 1_123_477_984


@pytest.fixture
def float32_value(float64_original_value) -> float:
    """
    Fixture for a 32-bit float value representing 123.456789.

    Float32 is a stateless, callable class with a static method that allows conversion
    from int back to float.

    Args:
        float64_original_value (float): The 64-bit floating-point value to convert to 32-bit.

    Returns:
        float: The 32-bit float representation of 'float64_original_value'.

    IMPORTANT! Notes:
        64-bit Float representation: 123.456789.
        32-bit Integer representation: 1123477984
        32-bit Hex representation: 0x42f6e9e0
        32-bit Binary representation: 0b01000010111101101110100111100000
    """
    return float32(float64_original_value)


def test_conversion_of_standard_float_to_float32(float32_value, float32_expected_value):
    """
    Test the conversion of a standard float to a 32-bit float.

    This test checks if the converted 32-bit float value is close enough to the expected value
    within an acceptable tolerance.

    Args:
        float32_value (float): The 32-bit float value obtained from converting a standard float.
        float32_expected_value (int): The expected 32-bit float value represented as an integer.
    """
    tolerance = 500_000  # Adjust the tolerance based on your precision requirements
    assert integer_is_close(float32_value, float32_expected_value, tolerance)


def test_conversion_of_standard_float32_to_float(float32_value, float64_expected_value):
    """
    Test the conversion of a standard 32-bit float to a 64-bit float.

    This test checks if the converted 64-bit float value is close enough to the expected value
    within an acceptable tolerance.

    Args:
        float32_value (float): The 32-bit float value to convert to a 64-bit float.
        float64_expected_value (float): The expected 64-bit float value.
    """
    float32_as_float64 = float32.to_float(float32_value)
    assert float_is_close(float32_as_float64, float64_expected_value, 1e-03)


def test_float32_precision(float32_value, float64_expected_value):
    """
    Test the precision of the 32-bit float type.

    This test checks if the 32-bit float representation is close enough to the expected 64-bit float
    value within an acceptable relative tolerance.

    Args:
        float32_value (float): The 32-bit float value to test for precision.
        float64_expected_value (float): The expected 64-bit float value.
    """
    expected_value = float64_expected_value

    converted_value = float32.to_float(float32_value)

    assert converted_value != expected_value
    assert float_is_close(converted_value, expected_value, 1e-03)


def test_float32_special_values():
    """
    Test special values like zero, infinities, and NaN for the 32-bit float type.
    """
    assert float32.to_float(float32(0)) == 0
    assert float32.to_float(float32(float("inf"))) == float("inf")
    assert float32.to_float(float32(float("-inf"))) == float("-inf")
    assert math.isnan(float32.to_float(float32(float("nan"))))


# Add more tests for other cases as needed
