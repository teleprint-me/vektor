"""
vektor/embeddings/cosine.py


arxiv: https://arxiv.org/pdf/2205.05092.pdf
"""

import numpy as np


class Cosine:
    def similarity(self, vector1, vector2):
        """
        Calculate the cosine similarity between two vectors.

        Args:
            vector1 (numpy.ndarray): The first vector.
            vector2 (numpy.ndarray): The second vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        # dot product is zero if vectors form a right angle
        dot_product = np.dot(vector1, vector2)
        # get the magnitude of each vector
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        # handle division by zero if either vector has zero magnitude
        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return similarity


if __name__ == "__main__":
    # Example usage
    cosine = Cosine()

    vector_a = np.array([1, 2, 3])
    vector_b = np.array([50, 5, 20])

    similarity = cosine.similarity(vector_a, vector_b)
    print(f"Cosine Similarity: {similarity}")
