"""
vektor/encoders/positional.py

https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
"""
import numpy as np


class PositionalEncoder:
    def __init__(self):
        ...

    def position(self):
        ...

    def encode(self):
        ...

    def decode(self):
        ...

    def get_encoding_position(seq_len, dims, freq=10000):
        # Initialize the matrix to hold the positional encodings
        matrix = np.zeros((seq_len, dims))

        # Iterate through each position in the sequence
        for pos in range(seq_len):
            # Iterate over each element (considering pairs for sine and cosine)
            for el in np.arange(int(dims / 2)):
                # Calculate the denominator for the sine and cosine functions
                # This denominator controls the frequency of the waves
                denominator = np.power(freq, 2 * el / dims)

                # Calculate sine value for even indices
                matrix[pos, 2 * el] = np.sin(pos / denominator)
                # Calculate cosine value for odd indices
                matrix[pos, 2 * el + 1] = np.cos(pos / denominator)

        return matrix
