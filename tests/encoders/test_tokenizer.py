"""
tests/test_tokenizer.py
"""
import pytest

from vektor.encoders.tokenizer import Tokenizer


def test_tokenize():
    # Test the tokenize function with different character sets
    texts = [
        ("Hello, world! 123", ["Hello", ",", "world", "!", "123"]),
        (
            "こんにちは！これはテストです。",
            ["こんにちは", "！", "これはテストです", "。"],
        ),
        ("😊👍", ["😊", "👍"]),
    ]
    for text, expected_tokens in texts:
        assert Tokenizer.tokenize(text) == expected_tokens


@pytest.mark.parametrize(
    "token,expected",
    [
        ("test", 1952805748),
        ("A", 65),
        ("😊", 4036991114),
        ("😊👍", 17338744812909597069),
        ("encode-decode", 8036207974711832502455940113509),
        ("a", 97),
        ("aa", 24929),
        ("aaa", 6381921),
        ("AA", 16705),
        ("AAA", 4276545),
        ("Abc", 4285027),
        ("Hello", 310939249775),
        ("Hello!", 79600447942433),
        ("Hello, world!", 5735816763073854953388147237921),
    ],
)
def test_encode_to_bytes(token, expected):
    # Test encoding a token to bytes
    assert Tokenizer.encode_to_bytes(token) == expected


@pytest.mark.parametrize(
    "encoded,expected",
    [
        (1952805748, "test"),
        (65, "A"),
        (4036991114, "😊"),
        (17338744812909597069, "😊👍"),
        (8036207974711832502455940113509, "encode-decode"),
        (97, "a"),
        (24929, "aa"),
        (6381921, "aaa"),
        (16705, "AA"),
        (4276545, "AAA"),
        (4285027, "Abc"),
        (310939249775, "Hello"),
        (79600447942433, "Hello!"),
        (5735816763073854953388147237921, "Hello, world!"),
    ],
)
def test_decode_from_bytes(encoded, expected):
    # Test decoding bytes back into a token
    assert Tokenizer.decode_from_bytes(encoded) == expected


def test_encode_decode():
    # Test the encode and decode functions together with a variety of tokens
    tokens = ["encode-decode", "テスト", "😊👍"]
    for token in tokens:
        encoded = Tokenizer.encode_to_bytes(token)
        decoded = Tokenizer.decode_from_bytes(encoded)
        assert (
            token == decoded
        ), f"Encoding and then decoding should return the original token for '{token}'"


# No need for tests for error handling if the Tokenizer is expected to handle all UTF-8 characters
