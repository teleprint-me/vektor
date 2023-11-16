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
TComplex = complex
TScalar = Union[TBool, TInt, TFloat, TComplex]
TVector = List[TScalar]
TMatrix = List[TVector]
TTensor2D = TMatrix  # 2D tensor (matrix)
TTensor3D = List[TMatrix]  # 3D tensor (list of matrices)
TShape = Tuple[TInt, TInt]


class DType(Protocol, metaclass=ABCMeta):
    """
    Abstract base class representing a data type, designed for converting between
    floating point and integer representations with custom precision. This class
    serves as a protocol for subclasses to implement specific conversion logic
    for different data types.
    """

    @abstractmethod
    def __call__(self, value: TFloat) -> TInt:
        """
        Convert a floating-point number to its corresponding integer representation
        based on the custom precision defined by the subclass.

        Parameters:
        value (float): The floating-point number to be converted.

        Returns:
        int: The integer representation of the floating-point number.
        """
        ...

    @abstractmethod
    def to_float(self, value: TInt) -> TFloat:
        """
        Convert an integer back to its floating-point representation based on the
        custom precision defined by the subclass.

        Parameters:
        value (int): The integer value to be converted.

        Returns:
        float: The floating-point representation of the integer.
        """
        ...


class Float32(DType):  # WIP
    # Special Numeric Values
    ZERO = 0x00000000
    NAN = 0x7FC00000
    PINF = 0x7F800000
    NINF = 0xFF800000

    # Constants for Conversion
    BIAS = 127
    MAX_EXPONENT = 255
    MANTISSA_BITS = 23
    SIGN_POSITION = 31

    # Bitwise Masks: Specific to converting from int32 to float64.
    SIGN_MASK = 0x01
    EXPONENT_MASK = 0xFF  # Mask is 8-bit
    MANTISSA_MASK = 0x007FFFFF

    def __call__(self, value: TFloat) -> TInt:
        # NOTE: This method is still returning values with a wide margin of error that can not be ignored.
        # This is verifiable with `Float32.to_float` which returns proper results.
        # The current margin of error within this method is acceptable for the time being considering the limitations of the given context.
        # Handle Special Cases
        if value == 0:
            return self.ZERO
        if math.isnan(value):
            return self.NAN
        if math.isinf(value):
            return self.PINF if value > 0 else self.NINF

        # Extract the Sign
        sign = 0 if value >= 0 else 1
        value = abs(value)

        # Decompose value into exponent and mantissa
        mantissa, exponent = math.frexp(value)

        # Adjust the exponent for 32-bit float and add the bias
        exponent = int(exponent + self.BIAS - 1)

        # Scale the mantissa to fit within the specified range
        mantissa = round(mantissa * (1 << self.MANTISSA_BITS))

        # Clamping the mantissa to fit within 23 bits
        mantissa = max(0, mantissa)
        mantissa = min(mantissa, (1 << self.MANTISSA_BITS) - 1)

        # Combine the sign, exponent, and mantissa into a 32-bit format
        result = (
            (sign << self.SIGN_POSITION) | (exponent << self.MANTISSA_BITS) | mantissa
        )

        return result

    def to_float(self, value: TInt) -> TFloat:
        # Handle special cases
        if value == self.ZERO:
            return 0.0
        if value == self.NAN:
            return float("nan")
        if value == self.PINF:
            return float("inf")
        if value == self.NINF:
            return float("-inf")

        # Extract the sign, exponent, and mantissa
        sign = (value >> self.SIGN_POSITION) & self.SIGN_MASK
        exponent = (value >> self.MANTISSA_BITS) & self.EXPONENT_MASK
        mantissa = value & self.MANTISSA_MASK

        # Handle denormalized numbers (exponent is all 0s)
        if exponent == 0:
            exponent = 1 - self.BIAS  # Denormalized exponent bias
        else:
            mantissa |= 1 << self.MANTISSA_BITS  # Add implicit leading 1 for normalized
            exponent -= self.BIAS

        # Calculate the floating-point number
        float_value = (mantissa / (2**self.MANTISSA_BITS)) * (2**exponent)
        if sign == 1:
            float_value = -float_value

        return float_value


float32 = Float32()


class Float16(DType):  # TODO
    def __call__(self, value: TFloat) -> TInt:
        raise NotImplementedError

    def to_float(self, value: TInt) -> TFloat:
        raise NotImplementedError


class Float8(DType):  # WIP
    def __call__(self, value: TFloat) -> TInt:
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

    def to_float(self, value: TInt) -> TFloat:
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


class Float4(DType):  # TODO
    def __call__(self, value: TFloat) -> TInt:
        raise NotImplementedError

    def to_float(self, value: TInt) -> TFloat:
        raise NotImplementedError
