"""
NOTE: This is a WIP (Work in Progress).

vektor/core/dtype.py

## 32-BIT FLOATING POINT FORMAT IEEE-754

SIGN BIT          8-BIT EXPONENT              23-BIT MANTISSA
   1                 10000011           0111100 | 01100000 | 00000000

EXAMPLE: -23.546875    HEX: 0xc1bc6000

value = (-1)^sign · 2^(exponent - bias) · mantissa

### Understanding IEEE-754

1. **Components**: A floating-point number in IEEE-754 format consists of three parts:
    - **Sign bit**: Indicates positive or negative.
    - **Exponent**: Determines the range of the number.
    - **Mantissa (or significand)**: The precision part of the number.

2. **Representation**: The standard representation for a floating-point number is:

    (-1)^sign · 2^(exponent - bias) · mantissa

    The exponent is 'biased' to handle both positive and negative exponents.

3. **Special Values**: NaN, infinity, and zero are represented using specific patterns in the exponent and mantissa.

### Conversion Between Python Int and Floating-Point values

1. **From Float to Int (Custom Precision)**:
   - **Extract Components**: Use `math.frexp()` to break the float into a mantissa and exponent. 
   - **Normalize and Scale**: Adjust the mantissa and exponent according to your custom precision. This involves scaling and possibly rounding the mantissa, and adjusting the exponent with a custom bias.
   - **Pack into Int**: Combine the sign, exponent, and mantissa into an integer using bitwise operations.

2. **From Int to Float**:
   - **Extract Components**: Use bitwise operations to extract the sign, exponent, and mantissa from the integer.
   - **De-normalize**: Adjust the exponent and mantissa back to their original scale. This involves adding the bias back to the exponent and converting the mantissa back to a fraction.
   - **Construct the Float**: Combine these components back into a floating-point number.
"""
import math
from abc import ABCMeta, abstractmethod
from typing import List, Protocol, Tuple, Union

TBool = bool
TInt = int
TFloat = float
TScalar = Union[TBool, TInt, TFloat]
TVector = List[TScalar]
TMatrix = List[TVector]
TShape = Tuple[int, int]


class DType(Protocol, metaclass=ABCMeta):
    """
    Abstract base class representing a data type, designed for converting between
    floating point and integer representations with custom precision. This class
    serves as a protocol for subclasses to implement specific conversion logic
    for different data types.
    """

    @abstractmethod
    def __call__(self, value: float) -> int:
        """
        Convert a floating-point number to its corresponding integer representation
        based on the custom precision defined by the subclass.

        Parameters:
        value (float): The floating-point number to be converted.

        Returns:
        int: The integer representation of the floating-point number.
        """
        ...

    @staticmethod
    @abstractmethod
    def to_float(value: int) -> float:
        """
        Convert an integer back to its floating-point representation based on the
        custom precision defined by the subclass.

        Parameters:
        value (int): The integer value to be converted.

        Returns:
        float: The floating-point representation of the integer.
        """
        ...


class float32(DType):  # WIP
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
        ...  # TODO


class float16(DType):  # TODO
    def __call__(self, value: float) -> int:
        ...

    @staticmethod
    def to_float(value: int) -> float:
        ...


class float8(DType):  # WIP
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


class float4(DType):  # TODO
    def __call__(self, value: float) -> int:
        ...

    @staticmethod
    def to_float(value: int) -> float:
        ...
