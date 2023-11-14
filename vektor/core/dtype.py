"""
vektor/core/dtype.py

              32-BIT FLOATING POINT FORMAT IEEE-754

SIGN BIT          8-BIT EXPONENT              23-BIT MANTISSA
   1                 10000011           0111100 | 01100000 | 00000000
1 = NEGATIVE      EXAMPLE: -23.546875
0 = POSITIVE      HEX: 0xc1bc6000

           value = (-1)^sign · 2^(exponent - bias) · mantissa

NOTE: This is a WIP (Work in Progress).
"""
import math
from abc import ABCMeta, abstractmethod
from typing import Protocol


class TFloat(Protocol, metaclass=ABCMeta):
    def __init__(self, sign, exponent, mantissa, bias: int):
        ...

    @abstractmethod
    def __call__(self, value: float) -> "TFloat":
        ...

    @staticmethod
    @abstractmethod
    def to_float(value: "TFloat") -> float:
        ...


class Float64(TFloat):
    # TODO: Implement arithmetic operations with appropriate precision handling
    ...


class Float32(TFloat):
    # NOTE: Hexadecimal values are always type `int` in Python.

    def __call__(self, value: float) -> int:
        # Handle Special Cases
        if value == 0:
            return 0x00000000
        if math.isnan(value):
            return 0x7FC00000  # Common representation of NaN
        if math.isinf(value):
            return (
                0x7F800000 if value > 0 else 0xFF800000
            )  # Positive and negative infinity

        # Extract the Sign
        sign = 0 if value >= 0 else 1
        value = abs(value)

        # Normalize the Number
        mantissa, exponent = math.frexp(value)

        # Adjust the Exponent for 32-bit float and Add Bias
        bias = 127
        exponent = min(max(int(exponent + bias), 0), 255)

        # Convert Mantissa to 23-bit format
        mantissa = int((mantissa - 1.0) * (2**23)) & 0x7FFFFF

        # Combine into a 32-bit format
        return (sign << 31) | (exponent << 23) | mantissa

    @staticmethod
    def to_float(value: int) -> float:
        # Implement conversion from custom 32-bit to 64-bit float
        ...


class Float16:
    def __init__(self, value):
        # Store the value with float16 precision
        pass

    # Implement methods for arithmetic operations, str, etc.


class Float8:
    def __call__(self, value: float) -> int:
        """
        Convert a value from float64 to float8.
        """
        # 1. Handle Special Cases:
        # Zero, infinity, and NaN (Not a Number) need special handling.
        if value == 0:
            return 0  # Represent zero as 0x00
        if math.isnan(value):
            return 0xFF  # Example representation for NaN
        if math.isinf(value):
            # Differentiate between positive and negative infinity
            return 0x7F if value > 0 else 0xFF

        # 2. Extract the Sign (1 bit):
        # Determine if the number is positive or negative.
        sign = 0 if value > 0 else 1
        value = abs(value)

        # 3. Normalize the Number:
        # Get the mantissa and exponent by normalizing the number.
        # frexp: Return the mantissa and exponent of x as the pair (m, e)
        mantissa, exponent = math.frexp(value)

        # 4. Scale exponent to fit into 4 bits and adjust for bias
        exponent_bias = 15  # Example bias for 4-bit exponent
        exponent = min(max(int(exponent + exponent_bias), 0), 15)

        # 5. Quantize mantissa to fit into 3 bits
        mantissa = int((mantissa - 0.5) * 8) & 0x07

        # 6. Pack the Bits: Combine the sign, exponent, and mantissa into an 8-bit format.
        return (sign << 7) | (exponent << 3) | mantissa

    @staticmethod
    def to_float(value: int) -> float:
        """
        Convert a value from 8-bit custom floating point to 64-bit standard floating point.

        NOTE: value = (-1)^sign · 2^(exponent - bias) · mantissa
        """
        # Extract components from 8-bit float
        sign = (value >> 7) & 0x01  # is the number negative or positive?
        exponent = (value >> 3) & 0x0F  # exponent is 4-bit.
        mantissa = value & 0x07  # Dequantize mantissa?

        # Adjust exponent for bias
        exponent_bias = 15  # The same bias used in float_to_float8
        exponent = exponent - exponent_bias

        # Convert mantissa to fraction in the range [0.5, 1)
        fraction = 0.5 + (mantissa / 8.0)

        # Construct the float
        result = fraction * (2**exponent)

        # Apply the sign
        if sign == 1:
            result = -result

        return result


class Float4:
    def __init__(self, value):
        # Store the value with float4 precision
        pass

        # Implement methods for arithmetic operations, str, etc.
        # Store the value with float4 precision
        pass

    # Implement methods for arithmetic operations, str, etc.
