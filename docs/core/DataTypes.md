# Vektor Data Types: Custom Data Types for ML/AI in Python

## **1. Introduction**

### Purpose
This document explores the importance of custom data types in Machine Learning and Artificial Intelligence, particularly in Python. Custom data types are vital for various purposes including simulations, educational tools, and addressing specific computational needs in the ML/AI domain. They allow for more controlled and efficient data handling, essential in fields where precision and optimization are key.

### Scope
Our focus will be on pure Python implementations, suitable for understanding and experimenting with advanced concepts like Neural Networks and Transformers. This approach enables a deeper comprehension of the underlying mechanisms, fostering a better understanding of these complex systems.

### Goal
The primary goal is to demonstrate how to apply a bit mask to a floating-point number in Python, thereby reducing its precision. This method is instrumental in understanding and manipulating data at a binary level, a crucial aspect in many ML/AI algorithms.

### Process
The process involves converting a float to its binary integer representation, applying a mask to modify specific bits, and then converting it back to a floating-point number. This approach mimics low-level operations in higher-level Python environment.

### Challenges
One significant challenge in Python is its handling of floats. Python uses a double-precision 64-bit binary format for floats, which doesn't directly expose its binary representation for manipulation, unlike languages like C or C++.

## Step-by-step guide for this approach:

1. **Start with a Float Value**:
   - Begin with an initial float value that represents the data you wish to manipulate.

2. **Convert to an Integer Representation**:
   - Accurately convert the float to an integer to access its binary representation, a key step in understanding and manipulating the data at a binary level.

3. **Apply the Mask**:
   - Design the mask considering the binary structure of a float. This process determines which bits of the data will be retained or altered.

4. **Convert Back to Float**:
   - The conversion back to float involves reversing the previous steps. This stage is critical and highlights the intricacies of binary manipulation in Python.

#### Practical Example

```python
import struct

def apply_mask_to_float(value, mask):
    # Convert float to binary representation (as a 64-bit double)
    binary_repr = struct.unpack('Q', struct.pack('d', value))[0]

    # Apply mask
    masked_binary = binary_repr & mask

    # Convert back to float
    return struct.unpack('d', struct.pack('Q', masked_binary))[0]

# Example Usage
value = 0.1232314325324
mask = 0xFFF0000000000000  # Example mask (significant for 64-bit float)
masked_value = apply_mask_to_float(value, mask)
print(masked_value)
```

#### Detailed Explanation and Insights

- We utilize the `struct` module to transition a float into its binary form (`'d'` for double) and then unpack it as a long long integer (`'Q'` for unsigned long long).
- The application of the mask is a key step, allowing for selective bit manipulation. In the example, the mask `0xFFF0000000000000` retains the most significant bits, including the exponent, setting the less significant bits to zero.
- The reverse process converts the binary data back into a floating-point number, demonstrating the flexibility of Python in handling such low-level operations.

#### Enhancements and Further Exploration

- **Use-Cases in ML/AI**: In Machine Learning, such precision manipulation can be crucial for tasks like model quantization, which reduces model size and speeds up inference, particularly on resource-constrained devices.
- **Visual Aids**: Incorporating diagrams illustrating the binary structure of floats and the effect of different masks can aid in understanding.
- **Additional Mask Examples**: Exploring different masks, such as those focusing on mantissa bits, can provide insights into various aspects of precision and rounding behavior in floating-point arithmetic.
- **Limitations and Cautions**: While this method offers powerful insights and capabilities, it's important to be aware of potential pitfalls, such as precision loss and the introduction of numerical instability in certain calculations.
- **Comparative Analysis with Lower-level Languages**: A brief comparison with similar operations in languages like C or C++ could highlight the nuances and potential advantages of using Python for such tasks.

## **2. Standard Floating-Point Formats**

### Overview of IEEE 754 Standard
The IEEE 754 Standard is the cornerstone for floating-point computation in modern computing. It defines formats for representing floating-point numbers and sets rules for arithmetic operations, rounding, and handling special values. Understanding this standard is crucial for manipulating precision in ML/AI applications, as it directly influences how data is represented and processed.

### Python's `struct` Module
Python's `struct` module is an essential tool for handling binary data. It becomes particularly significant when controlling precision, allowing for direct manipulation of floating-point numbers at the binary level. This module bridges the gap between Python's high-level abstraction and the low-level binary representation of data.

### Formats
- **64-bit (Double Precision, `'d'`)**: This is the default floating-point format in Python, offering high precision and is widely used in general-purpose computations.
- **32-bit (Single Precision, `'f'`)**: With reduced precision, this format is useful in many ML/AI applications where memory efficiency is crucial, and the high precision of 64-bit floats is not necessary.
- **16-bit (Half Precision, `'e'`)**: Increasingly popular in deep learning, this format significantly reduces memory and computational requirements, facilitating the deployment of models on resource-constrained devices.

### Approach
While bit-masking offers fine-grained control, converting to a lower precision format using `struct` is more straightforward for simulating lower precision in Python. This method is less manual and leverages Python's inherent capabilities to handle different floating-point representations.

### Methodology
1. **Convert to Lower Precision**: Utilize `struct` to pack the float into a lower precision format (e.g., 32-bit) and then unpack it to obtain a float with reduced precision.
2. **Convert Back to Higher Precision**: Repack this lower precision float into a higher precision format (e.g., 64-bit) and unpack it to retrieve the final value.

This process inherently reduces precision due to the fewer bits available in the lower precision format.

#### Example Implementation:

```python
import struct

def lower_precision(value, from_format='d', to_format='f'):
    # Convert from higher precision to lower
    lower_precision_binary = struct.pack(to_format, value)
    lower_precision_value = struct.unpack(to_format, lower_precision_binary)[0]

    # Convert back to higher precision
    return struct.unpack(from_format, struct.pack(from_format, lower_precision_value))[0]

# Example Usage
value = 0.1232314325324
lower_precision_value = lower_precision(value)
print(lower_precision_value)
```

In this example, `from_format='d'` and `to_format='f'` denote double precision (64-bit) and single precision (32-bit) floats, respectively.

### Considerations
- **Simplicity**: This method simplifies precision control, making it more accessible than bit-masking.
- **Precision Loss**: The precision loss is controlled and predictable, as it adheres to standard floating-point formats.
- **Performance**: Leveraging Python's built-in capabilities for binary data handling, this method strikes a balance between performance and ease of use.
- **Practicality**: Ideal for simulations or experiments in ML/AI where observing the effects of reduced precision is necessary without delving into detailed bit manipulation.

### Enhanced Insights
- **Visual Representation**: Including diagrams showing the memory representation of different formats can provide a clearer understanding of how precision varies with each format.
- **Performance Analysis**: A brief analysis of performance trade-offs between these formats can guide practitioners in choosing the right precision level for their specific application.
- **Real-world Applications**: Illustrating specific ML/AI scenarios where precision plays a critical role, such as in model quantization or deploying models on mobile devices, can highlight the practical relevance of this knowledge.

Using `struct` for direct conversion between different floating-point precisions is a practical and effective approach, especially in educational or experimental settings within data science and machine learning domains.

## **3. Custom Precision Formats Using Bit-Masking**

### Concept Overview
Bit-masking is a powerful technique for precision control in numerical computations. It allows for custom manipulation of the binary representation of data, enabling the creation of non-standard precision formats. This section explores how bit-masking can be applied to create custom precision levels in floating-point numbers, especially useful in ML/AI applications.

### Methodology
1. **Converting Floats to Binary Representation**:
   - Utilize Python's `struct` module to convert floats into their binary form, allowing direct manipulation of their binary components.

2. **Applying Masks for Precision Control**:
   - Design and apply bit masks to the binary representation to achieve desired precision levels. This process can finely tune the amount of information retained in a floating-point number.

### Custom Formats
- **8-bit Precision**: 
  - **Design**: Tailored for specific applications where memory efficiency is paramount, but precision is less critical.
  - **Use Cases**: Useful in embedded systems or IoT devices where memory is limited.
  - **Limitations**: Significant precision and range limitations; suitable for specific contexts where these trade-offs are acceptable.

- **4-bit Precision**: 
  - **Nature**: Highly experimental, pushing the boundaries of precision.
  - **Potential Applications**: Could be explored in advanced ML/AI algorithms where model size and speed are more critical than precision.
  - **Challenges**: Pronounced precision and range limitations; requires careful consideration and thorough testing.

### Strategy
Employing direct conversions for standard floating-point formats (`f`, `d`, `e`) and resorting to bit-masking for custom formats like 8-bit and 4-bit precision offers a versatile approach. This strategy takes advantage of Python's built-in capabilities for standard formats while allowing creativity and flexibility for custom precisions.

### Direct Conversions for Standard Formats
Python's `struct` module efficiently handles standard floating-point formats:

- **`'f'` (32-bit floating-point number)**: Single precision, common in many ML/AI applications.
- **`'d'` (64-bit floating-point number)**: Double precision, the default in Python.
- **`'e'` (16-bit floating-point number)**: Half precision, increasingly used in deep learning models.

These conversions involve packing and unpacking float values into these formats, a straightforward process beneficial for simulating different precision levels.

### Bit-Masking for Custom Formats
For custom formats like 8-bit and 4-bit precision, bit-masking is employed:

1. **Convert float to binary**.
2. **Apply a custom-designed mask**.
3. **Convert back to float**.

This approach provides granular control over precision but requires a deeper understanding of binary representations and potential trade-offs.

### Enhanced Example Implementation

#### Direct Conversion:

```python
import struct

def convert_precision(value, target_format):
    # Convert to target format and back to double precision
    return struct.unpack('d', struct.pack(target_format, value))[0]

# Usage examples
value = 0.1232314325324
value_32bit = convert_precision(value, 'f')  # 32-bit
value_16bit = convert_precision(value, 'e')  # 16-bit
```

#### Bit-Masking for 8-bit and 4-bit:

```python
def apply_mask(value, mask):
    # Convert to binary, apply mask, and unpack
    packed = struct.pack('d', value)
    masked = int.from_bytes(packed, byteorder='little') & mask
    return struct.unpack('d', masked.to_bytes(8, byteorder='little'))[0]

# Masks for 8-bit and 4-bit precision
mask_8bit = 0xFF00000000000000  # 8-bit mask
mask_4bit = 0xF000000000000000  # 4-bit mask

# Applying masks
value_8bit = apply_mask(value, mask_8bit)
value_4bit = apply_mask(value, mask_4bit)
```

### Expanded Considerations
- **Precision and Range**: Each format has its specific range and precision capabilities. Custom formats require careful design to ensure they meet the intended application's requirements.
- **Performance**: Direct conversions offer efficiency, while bit-masking provides flexibility at the cost of some computational overhead.
- **Numerical Stability**: Lower precision formats, especially 4-bit, can introduce significant challenges in terms of numerical stability and precision. Understanding these limitations is crucial for their effective application.

### Visual Aids and Real-world Applications (To be Developed)
- **Visual Aids**: Future iterations of this document will include diagrams illustrating the effect of bit-masking on the binary representation of floats.
- **Real-world Examples**: Case studies or examples of where these custom precision formats have been effectively applied in ML/AI will be explored to provide practical insights.

Using a mix of direct conversions for standard formats and bit-masking for custom formats offers a comprehensive toolkit for experimenting with precision levels in numerical computations, particularly relevant in the fields of data science and machine learning.

## **4. General Application of the DType Base Class**

### Introduction to the DType Class
This section introduces the `DType` class, a versatile tool for managing different data types and precision levels in Python, particularly valuable in ML/AI applications.

### Code Snippets for Standard Formats
- **Direct Conversion**: 
  - Demonstrates using the `DType` class for converting between standard floating-point formats (`f`, `d`, `e`) using Python's `struct` module.
  - Suitable for scenarios where standard precision levels are adequate.

### Bit-Masking for Non-standard Formats
- **Custom Precision**: 
  - Example code for applying masks to simulate 8-bit and 4-bit precision.
  - This technique is particularly relevant for specialized applications requiring non-standard precision levels.

### Example Usage of DType Class

```python
"""
vektor/core/dtype.py
"""
import struct
from typing import Optional, Union

TScalar = Union[bool, int, float]

# Mapping from struct format codes to Python data types
FROM_MAP = {
    "d": float,  # double precision float
    "f": float,  # single precision float
    "e": float,  # half precision float
    # other format specifiers...
}


class DType:
    def __init__(
        self,
        from_format: str = "d",  # default from float64
        to_format: str = "f",  # default to float32
        precision_bits: Optional[int] = None,
    ):
        self.dtype = FROM_MAP.get(from_format)
        self.from_format = from_format
        self.to_format = to_format
        self.precision_bits = precision_bits
        self.mask = self.calculate_mask() if precision_bits is not None else None

    def convert_precision(self, value: TScalar) -> TScalar:
        # Convert to target format and back to the original format
        packed_value = struct.pack(self.to_format, self.dtype(value))
        intermediate_value = struct.unpack(self.to_format, packed_value)[0]
        repacked_value = struct.pack(self.from_format, intermediate_value)
        return struct.unpack(self.from_format, repacked_value)[0]

    def calculate_mask(self) -> int:
        # Calculate and return the mask based on the specified precision_bits
        total_bits = 64  # Assuming we're dealing with 64-bit representation
        mask_bits = (1 << self.precision_bits) - 1  # Create mask for precision_bits
        return mask_bits << (total_bits - self.precision_bits)

    def mask_precision(self, value: TScalar) -> TScalar:
        packed_value = struct.pack("d", float(value))
        binary_repr = struct.unpack("Q", packed_value)[0]
        masked_binary = binary_repr & self.mask
        binary_packed = struct.pack("Q", masked_binary)
        return struct.unpack("d", binary_packed)[0]


# Example Usage for Floating-Point Precision Control
float64_value = 0.123456789
dtype_32bit = DType(to_format="f")  # Convert to 32-bit float
converted_value = dtype_32bit.convert_precision(float64_value)
print(converted_value)

float64_value = 0.123456789
dtype_8bit = DType(precision_bits=8)  # Dynamic 8-bit precision
masked_value = dtype_8bit.mask_precision(float64_value)
print(masked_value)
```

### Considerations for Floating-Point Precision Control

- **Precision and Range Impact**: It's crucial to understand how different floating-point formats affect the precision and range of values. This awareness is vital in applications where the accuracy of numerical calculations is critical, such as in machine learning algorithms and data analysis.

- **Performance Implications**: The efficiency of direct conversions using the `struct` module is generally high. However, when applying non-standard formats through bit-masking, be mindful of the additional computational overhead. This consideration becomes especially important in large-scale applications or real-time processing scenarios.

- **Numerical Stability and Accuracy**: Manipulating the precision of floating-point numbers can lead to changes in numerical stability and accuracy. This is a key consideration in ML/AI, where the outcomes of algorithms can be sensitive to small changes in input data. It's important to thoroughly test and understand the implications of precision changes on the results.

- **Contextual Application**: The choice between direct conversion and bit-masking should be based on the specific requirements of the application. For instance, standard formats are generally sufficient for most computational tasks, but custom formats might be necessary for specialized applications or to meet specific resource constraints.

- **Practical Use Cases**: When implementing these methods, consider practical scenarios where precision control can enhance performance or efficiency. For example, reducing the precision of data in neural network training can decrease memory usage and computational time, enabling more efficient training processes.

## **5. Integer and Boolean Masking**

### Overview
This section delves into the specifics of applying precision control to integers and booleans in Python. Unlike floating-point numbers, integers and booleans allow for more direct bit manipulation due to their simpler binary representation. Understanding how to apply these techniques can be valuable in certain computational contexts.

### Integer Masking
- **Direct Binary Manipulation**: In Python, integers are represented as binary numbers in memory, enabling direct application of bitwise operations. This characteristic simplifies the process of applying masks to integers, as it doesn't require the conversion steps necessary for floating-point numbers.
- **Practical Applications**: 
  - **Data Compression**: Masking can be used to reduce the size of integer data, which can be essential in scenarios involving large datasets or limited storage capacity.
  - **Security and Privacy**: Applying masks to integers can be a technique for obfuscating data, useful in scenarios where privacy is a concern.
- **Considerations**: 
  - **Range and Precision**: The range and precision of the original integer values should be considered when designing masks. Over-aggressive masking might lead to significant data loss or distortion.
  - **Performance**: While generally efficient, the impact of masking on performance should be evaluated, especially when dealing with very large integers or extensive datasets.

### Boolean Masking
- **Theoretical Perspective**: Booleans in Python are essentially integers with only two states: `True` (1) and `False` (0). Masking booleans is technically feasible but often lacks practical utility due to their binary nature.
- **Potential Use Cases**: 
  - While not common, there could be scenarios in algorithm development or data processing where applying a mask to boolean values might be useful for achieving specific logical outcomes.
- **Considerations**: 
  - **Logical Integrity**: Care must be taken to ensure that masking does not inadvertently alter the logical integrity of boolean operations.
  - **Simplicity Over Complexity**: In most cases, simpler logical operations are preferred over masking for booleans due to clarity and maintainability.

### Example Implementations
#### Integer Masking:
```python
# Integer Masking Example
integer_value = 123456789
mask = 0xFFFF  # Example mask
masked_integer = integer_value & mask
print(masked_integer)
```

#### Boolean Masking (Theoretical):
```python
# Boolean Masking Example (Theoretical)
boolean_value = True
mask = 0x1  # Mask that retains the boolean value
masked_boolean = boolean_value & mask
print(masked_boolean == True)
```

### Summary
While the primary focus in precision control is often on floating-point numbers, understanding how to apply similar concepts to integers and booleans can broaden the toolkit available to Python developers, especially in specialized scenarios. This section provides a foundational understanding of these techniques, emphasizing their practicality and limitations.

## **6. Applying DType Class to Integers and Booleans**

### Simplified Handling for Integers and Booleans
This section presents the application of the `DType` class to integer and boolean types in Python. Given their straightforward binary representation, these data types can be manipulated more directly compared to floating-point numbers, simplifying precision control.

### Integer and Boolean Masking
- **Integers**: 
  - The `DType` class can apply masks directly to integer values, leveraging Python's native binary representation of integers. This process is efficient and does not require the convoluted conversions needed for floating-point numbers.
  - **Practical Application**: In situations where integer data needs to be compressed or obfuscated, direct masking offers an efficient solution.
- **Booleans**: 
  - Masking boolean values is less common due to their binary nature (True/False). However, the `DType` class can theoretically handle boolean masking, though its practical applications are limited.
  - **Theoretical Use**: Could be explored in specialized algorithmic scenarios where altering boolean states in a controlled manner is required.

### Example Implementation for Integer and Boolean Masking

```python
class DType:
    # ... [other parts of the class] ...

    def mask_precision(self, value: TScalar) -> TScalar:
        if isinstance(value, int):
            # Directly apply the mask for integers
            return value & self.mask
        elif isinstance(value, float):
            # Use the existing masking process for floats
            # ... [existing float masking code] ...
        # Additional handling for booleans can be implemented if needed

    # ... other methods ...

# Example Usage for Integer Masking
integer_value = 123456789
dtype_integer = DType(mask=0xFFFF)  # Example mask for integer
masked_integer = dtype_integer.mask_precision(integer_value)
print(masked_integer)
```

### Considerations
- **Precision and Range for Integers**: It's important to design masks that are appropriate for the range and precision of the integer values being processed. Over-masking can lead to significant data loss.
- **Use Case for Booleans**: While boolean masking is not typical, understanding its theoretical basis can inspire innovative uses in specific contexts.
- **Type-Specific Efficiency**: The `DType` class demonstrates the efficiency of direct masking for integers and booleans, offering a contrast to the more nuanced approach needed for floating-point types.

### Summary
The `DType` class's extension to handle integers and booleans illustrates its versatility and adaptability across different data types. While the primary focus is often on floating-point numbers, the ability to work effectively with integers and booleans broadens the class's utility, making it a valuable tool in a wider range of computational scenarios.

## **7. Integration with Neural Network Components**

- **Matrix and Vector Operations**: Adapting common operations in ML/AI (like
  matrix multiplication, vector addition) to work with custom data types.
- **Considerations in Neural Networks**: Discuss the impact of precision on
  training and inference in neural networks, particularly in Transformers.

## **8. Performance and Limitations**

- **Overheads and Efficiency**: Discuss the computational overheads and
  efficiency considerations of using custom data types in Python.
- **Numerical Stability and Precision Loss**: Highlight the trade-offs between
  memory efficiency, computational speed, and numerical accuracy.

## **9. Conclusion and Future Work**

- **Summary**: Recap the key findings and methodologies.
- **Future Directions**: Suggest areas for further exploration, such as
  integration with GPU computing or optimization techniques.

## **10. References**

- **List of Sources**: Include references to any external materials, research
  papers, or documentation that were used or could be useful for further
  reading.
