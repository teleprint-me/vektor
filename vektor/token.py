"""
vektor/token.py
"""
import binascii
import re


# TODO: Refine the subword tokenization logic
def subword_tokenize(text, split_length=5):
    # Split the text into subwords of the specified length
    return [text[i : i + split_length] for i in range(0, len(text), split_length)]


# TODO: Refine the word tokenization logic
def word_tokenize(text):
    # Improved regex to handle contractions, URLs, etc.
    tokens = re.findall(r"\w+|[^\w\s]|'\w+", text, re.UNICODE)
    return tokens


# Define a function to tokenize a string into words, subwords, and punctuation as separate tokens
def tokenize(text):
    # First, tokenize by words using the refined word_tokenize function
    word_tokens = word_tokenize(text)
    # Then, for each word token, perform subword tokenization
    tokens = [subword for token in word_tokens for subword in subword_tokenize(token)]
    return tokens


# Add error handling to encoding and decoding functions
def encode_token_to_bytes(token):
    try:
        return int(binascii.hexlify(token.encode()), 16)
    except UnicodeEncodeError as e:
        # Handle the specific error, perhaps log it or re-raise with additional information
        raise UnicodeEncodeError(f"Encoding failed for token: {token}") from e


def decode_bytes_to_token(encoded_token):
    try:
        hex_str = format(encoded_token, "x")  # Convert to hex string
        hex_str = "0" * (len(hex_str) % 2) + hex_str
        return binascii.unhexlify(hex_str).decode()
    except (binascii.Error, ValueError) as e:
        # Handle decoding errors
        raise ValueError(f"Decoding failed for encoded token: {encoded_token}") from e


if __name__ == "__main__":
    # Testing the encoding and decoding
    text = "Ye are a baggage: the Slys are no rogues; look in the chronicles; we came in with Richard Conqueror. Therefore paucas pallabris; let the world slide: sessa!"
    tokens = tokenize(text)
    encoded_tokens = [encode_token_to_bytes(token) for token in tokens]
    decoded_tokens = [decode_bytes_to_token(token) for token in encoded_tokens]

    print(f"Original text: {text}")
    print(f"Original tokens: {tokens}")
    print(f"Encoded tokens: {encoded_tokens}")
    print(f"Decoded tokens: {decoded_tokens}")
