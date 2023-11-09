"""
tests/test_sparse_encoder.py
"""
import pytest

from vektor.sparse_encoder import SparseEncoder
from vektor.tokenizer import Tokenizer


@pytest.fixture
def encoder():
    # A fixture for creating a fresh SparseEncoder instance for each test
    return SparseEncoder(num_bits=16)


def test_quantization(encoder):
    # Test that quantization correctly limits the size of the integer
    assert encoder.quantize(2**16 + 1) == 65535  # Changed from 2**16 to 2**16 + 1
    assert (
        encoder.quantize(2**16 - 1) == 65535
    )  # The value should be equal to max_val
    assert encoder.quantize(0) == 0


def test_unique_token_assignment(encoder):
    # Test that unique tokens get unique indices
    tokens = ["hello", "world", "hello"]
    expected_indices = [0, 1, 0]  # "hello" should get the same index both times
    encoded_indices = encoder.fit_transform(tokens)
    assert encoded_indices == expected_indices


def test_fit_transform(encoder):
    # Test the fit_transform method for a known input and output
    tokenizer = Tokenizer()
    text = "hello world hello"
    tokens = tokenizer.tokenize(text)
    unique_tokens = list(set(tokens))
    encoded_tokens = [tokenizer.encode_to_bytes(token) for token in unique_tokens]
    sparse_encoded_tokens = encoder.fit_transform(encoded_tokens)
    assert len(sparse_encoded_tokens) == len(unique_tokens)
    assert all(isinstance(index, int) for index in sparse_encoded_tokens)


def test_repeated_tokens(encoder):
    # Test that repeated tokens are handled correctly
    tokens = ["repeat", "repeat", "repeat"]
    expected_indices = [0, 0, 0]  # All instances of "repeat" should get index 0
    encoded_indices = encoder.fit_transform(tokens)
    assert encoded_indices == expected_indices


def test_large_numbers(encoder):
    # Test that large numbers are handled according to the quantization rules
    large_number = 2**32
    quantized_number = encoder.quantize(large_number)
    assert quantized_number == encoder.max_val


def test_empty_input(encoder):
    # Test that an empty input doesn't break the encoder and returns an empty list
    encoded_indices = encoder.fit_transform([])
    assert encoded_indices == []


# Additional tests can include handling of non-ASCII characters, very long tokens, etc.
